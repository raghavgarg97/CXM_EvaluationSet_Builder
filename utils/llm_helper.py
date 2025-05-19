import os
import json
import aiohttp
import asyncio
import hashlib
import logging
import tiktoken
import sys
import traceback
from copy import deepcopy
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log
)

# Setup logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
encoding = tiktoken.encoding_for_model("gpt-4")

class LLM:
    def __init__(self, log_file_name, config_file_path="./configs/llm_config.json"):
        self.config = self._load_config(config_file_path)
        self.server_url = self.config["server_url"]
        self.headers = self.config["headers"]

        self._call_router_async = retry(
            stop=stop_after_attempt(self.config["retry_config"]["max_retries"]),
            wait=wait_exponential(
                multiplier=self.config["retry_config"]["exponential"],
                max=self.config["retry_config"]["max_delay"],
                exp_base=self.config["retry_config"]["initial_delay"]
            ),
            before_sleep=before_sleep_log(logger, logging.DEBUG),
            retry_error_callback=lambda retry_state: [None for _ in range(len(retry_state.args[1]))]
        )(self._call_router_async_core)

        self.base_payload = self.config["base_payload"]
        self.models_dict = self.config["models_dict"]

        # Setup caching and log file.
        self.task_records = defaultdict(lambda: defaultdict(dict))

        self.log_file = os.path.join("./llm_logs", f"{log_file_name}.txt")
        self._load_records_from_file()

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def chat(self,
                   roles_and_messages,
                   task_name,
                   model_name,
                   perspective="agent",
                   system_message=None,
                   add_prompter=False,
                   use_cache = True,
                   **kwargs):
        """
        Handles a single LLM conversation. Uses caching before issuing API call.
        """
        cache_key = self._make_cache_key(roles_and_messages, system_message)
        cached_response = self._check_cache(task_name, cache_key, model_name)
        if cached_response is not None and use_cache:
            return cached_response

        payload = self._generate_payload(roles_and_messages, system_message, perspective, model_name,
                                         add_prompter, **kwargs)
        try:
            responses = await self._call_router_async([payload])
            response = responses[0]
        except Exception as e:
            raise RuntimeError(f"LLM chat failed: {e}") from e

        encoded_input_length = len(encoding.encode(self._get_inp_string_to_encode(roles_and_messages, system_message)))
        self._store_record(task_name, cache_key, response, model_name, encoded_input_length)
        return response

    async def _chat_batch_core(self,
                     list_of_roles_and_messages,
                     task_name,
                     model_name,
                     system_messages_list,
                     list_of_tools,
                     perspective="agent",
                     add_prompter=False,
                     use_cache=True,
                     **kwargs):
        """
        Handles multiple conversations in batch. For each conversation, if there is a
        cached response, use it. Otherwise, call the LLM and update cache (batch writing the logs).
        """

        assert len(list_of_roles_and_messages) == len(system_messages_list)
        if list_of_tools:
            assert use_cache is False
            assert len(list_of_tools) == len(system_messages_list)

        results = [None] * len(list_of_roles_and_messages)
        to_call = []
        to_call_indices = []
        # Accumulate cache keys for corresponding uncached convs.
        keys_to_call = []

        # 1) Check cache.
        for i, (conv, sys_message) in enumerate(zip(list_of_roles_and_messages, system_messages_list)):
            key = self._make_cache_key(conv, sys_message)
            cached = self._check_cache(task_name, key, model_name)
            if cached is not None and use_cache:
                results[i] = cached
            else:
                to_call.append((sys_message, conv))
                keys_to_call.append(key)
                to_call_indices.append(i)



        to_call_encoded_input_lengths = [self._get_inp_string_to_encode(conv, sys_message) for i, (conv, sys_message) in enumerate(zip(list_of_roles_and_messages, system_messages_list)) if i in to_call_indices]
        to_call_encoded_input_lengths = [len(x) for x in encoding.encode_batch(to_call_encoded_input_lengths)]

        # 2) If everything was cached, return.
        if not to_call:
            return results

        if list_of_tools:
            tools_to_call = [list_of_tools[j] for j in to_call_indices]
            payloads = [
                self._generate_payload(conv, sys_message, perspective, model_name, add_prompter, tools=tools, **kwargs)
                for tools, (sys_message, conv) in zip(tools_to_call, to_call)
            ]
        else:
            payloads = [
                self._generate_payload(conv, sys_message, perspective, model_name, add_prompter, **kwargs)
                        for sys_message, conv in to_call
            ]

        try:
            responses = await self._call_router_async(payloads)
        except Exception as e:
            raise RuntimeError(f"LLM batch chat failed: {e}") from e

        # 4) Update the cache records and results.
        log_entries = []
        for idx, resp, key, inp_len in zip(to_call_indices, responses, keys_to_call, to_call_encoded_input_lengths):
            results[idx] = resp
            log_entries.append({
                "task_name": task_name,
                "cache_key": key,
                "response": resp,
                "model": model_name,
                "input_tokens": inp_len
            })
            # Update the in-memory cache.
            self.task_records[task_name][model_name][key] = resp

        # Write all new log entries in one batch.
        self._write_log_batch(log_entries)
        return results

    async def chat_batch(self,
                    list_of_roles_and_messages,
                    batch_size,
                    task_name,
                    model_name,
                    list_of_tools = None,
                    perspective = "agent",
                    system_message = None,
                    add_prompter = False,
                    use_cache = True,
                    use_tqdm = True,
                    ** kwargs):

        iterator = range(0, len(list_of_roles_and_messages), batch_size)
        assert 'tools' not in kwargs or not list_of_tools, f"Only pass on of - list_of_tools or tools"

        if isinstance(system_message, str) or not system_message:
            system_messages_list = [system_message for _ in range(len(list_of_roles_and_messages))]
        else:
            system_messages_list = system_message

        iterator = tqdm(iterator, total = (len(list_of_roles_and_messages) + batch_size - 1)//batch_size) if use_tqdm else iterator
        results = []
        for i in iterator:
            roles_and_messages_batch = list_of_roles_and_messages[i:i + batch_size]
            system_messages_batch = system_messages_list[i:i + batch_size]
            list_of_tools_batch = list_of_tools[i:i + batch_size] if list_of_tools is not None else None

            curr_res = await self._chat_batch_core(
                list_of_roles_and_messages=roles_and_messages_batch,
                task_name=task_name,
                model_name=model_name,
                perspective=perspective,
                system_messages_list=system_messages_batch,
                add_prompter=add_prompter,
                use_cache=use_cache,
                list_of_tools=list_of_tools_batch,
                **kwargs
            )
            results.extend(curr_res)
        return results


    # @retry(
    #     stop=stop_after_attempt(self.max_retries),
    #     wait=wait_exponential(multiplier=self.exponential, max=self.max_retries, exp_base=self.initial_delay),
    #     before_sleep=before_sleep_log(logger, logging.DEBUG),
    #     retry_error_callback=lambda retry_state: [None for _ in range(len(retry_state.args[1]))]
    # )
    async def _call_router_async_core(self, payloads):
        """
        This private method calls the LLM router endpoint asynchronously.
        It submits a list of payloads and expects each of the returned JSON items to
        contain an answer inside item["choices"][0]["message"]["content"].
        """
        import json  # local import if needed
        async with aiohttp.ClientSession() as session:
            resp = await session.post(url=self.server_url,
                                        headers=self.headers,
                                        data=json.dumps({"payloads": payloads}))
            text = await resp.text()
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}, text={text}")
            result_json = json.loads(text)
            # TODO: adapt extraction logic for tool calls if necessary
            responses = [item["choices"][0]["message"] for item in result_json]
            return [response.get('content', response.get('tool_calls')) for response in responses]

    def _map_role_to_llm_role(self, role, perspective="agent"):
        """
        Map roles based on perspective.
        For perspective=="agent", then 'bot_agent' → 'assistant' and everything else → 'user'.
        """
        if perspective == "agent":
            return "assistant" if role == "bot_agent" else "user"
        else:
            return "user" if role == "bot_agent" else "assistant"

    def _generate_payload(self, roles_and_messages, system_message, perspective, model_name, add_prompter, **kwargs):
        """
        Given conversation information, build the final payload for the LLM request.
        """
        # Allow passing in a single string message.
        if isinstance(roles_and_messages, str):
            roles_and_messages = [('bot_customer', roles_and_messages)]
        # Start with a copy of the base payload.
        payload = deepcopy(self.base_payload)
        model_kwargs = deepcopy(self.models_dict[model_name])
        payload.update(model_kwargs)
        payload.update(kwargs)

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        for role, text in roles_and_messages:
            assert role in ['bot_agent', 'bot_customer', 'assistant', 'user']
            llm_role = self._map_role_to_llm_role(role, perspective) if 'bot' in role else role
            messages.append({"role": llm_role, "content": text})
        payload["messages"] = messages

        # Optional: instruct prompter to return JSON.
        if add_prompter:
            payload["response_format"] = {"type": "json_object"}

        # Adjust maximum tokens for models with specific naming patterns.
        if any(x in payload["model"] for x in ["o1", "o3"]):
            payload["max_completion_tokens"] = payload["max_tokens"]
            payload.pop("max_tokens", None)

        # print("Generated payload is:", payload)
        return payload

    @staticmethod
    def _get_inp_string_to_encode(roles_and_messages, system_message):
        str_to_encode = json.dumps({
            "system": system_message,
            "messages": roles_and_messages
        }, sort_keys=True)
        return str_to_encode

    def _make_cache_key(self, roles_and_messages, system_message):
        """
        Create a cache key based on the JSON representation of the conversation.
        """
        str_to_encode = self._get_inp_string_to_encode(roles_and_messages, system_message)
        return hashlib.sha256(str_to_encode.encode('utf-8')).hexdigest()

    def _check_cache(self, task_name, key, model):
        return self.task_records[task_name][model].get(key)

    def _store_record(self, task_name, key, response, model, encoded_input_length):
        if not response:
            return

        self.task_records[task_name][model][key] = response
        log_entry = {
            "task_name": task_name,
            "cache_key": key,
            "response": response,
            "model": model,
            "input_tokens": encoded_input_length
        }
        self._write_log_batch([log_entry])

    def _write_log_batch(self, log_entries):
        """
        Write one or more log entries in batch to the log file.
        Each log entry should be a dictionary.
        """
        # Ensure the log_file directory exists.
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for entry in log_entries:
                    if entry['response']:
                        f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write log entries: {e}")

    def _load_records_from_file(self):
        """
        Load cached records from the log file.
        Each line in the log file is expected to be a JSON entry.
        """
        if not os.path.exists(self.log_file):
            return
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        task = entry["task_name"]
                        key = entry["cache_key"]
                        resp = entry["response"]
                        model = entry["model"]
                        self.task_records[task][model][key] = resp
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to load cache records from file: {e}")
        # Optionally print a summary.
        for tsk, models in self.task_records.items():
            total = sum(len(entries) for entries in models.values())
            print(f"Restored {total} records for task '{tsk}'")
