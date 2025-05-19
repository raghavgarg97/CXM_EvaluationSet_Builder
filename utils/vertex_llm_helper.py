import os
import json
import copy
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Union

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    ToolConfig,
    Content,
    Part,
)


class vertexLLM:
    """
    Tiny helper around the Vertex‐AI Generative SDK that can be used
    as a drop‐in replacement for “OpenAI‐style” function calling code.
    """

    def __init__(self, service_account_key: str, location: str = "us-central1"):
        if not os.path.exists(service_account_key):
            raise FileNotFoundError(service_account_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
        with open(service_account_key, "r", encoding="utf-8") as f:
            project_id = json.load(f).get("project_id")
        if not project_id:
            raise ValueError("Could not find `project_id` inside the key file.")
        vertexai.init(project=project_id, location=location)

        self._generation_defaults = {
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
        }

    async def chat(
        self,
        messages: List[Tuple[str, str]],
        model_name: str = "gemini-2.0-pro",
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        **extra_generation_kwargs,
    ) -> List[str]:
        res = await self.chat_batch(
            [messages],
            model_name=model_name,
            tools=tools,
            system_message=system_message,
            generation_overrides=generation_overrides,
            batch_size=1,
            **extra_generation_kwargs,
        )
        return res[0]

    def chat_batch_core(
        self,
        batch_messages: List[List[Tuple[str, str]]],
        model_name: str = "gemini-2.0-pro",
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        **extra_generation_kwargs,
    ) -> List[Any]:
        # unchanged synchronous path
        return self._chat_impl(
            batch_messages,
            model_name=model_name,
            tools=tools,
            system_message=system_message,
            generation_overrides=generation_overrides,
            **extra_generation_kwargs,
        )

    async def chat_batch(
        self,
        all_messages: List[List[Tuple[str, str]]],
        model_name: str = "gemini-2.0-pro",
        tools: Optional[
            Union[
                List[Dict[str, Any]],       # one global list
                List[List[Dict[str, Any]]]  # per-conversation lists
            ]
        ] = None,
        system_message: Optional[
            Union[
                str,            # one global
                List[str]       # one per-conversation
            ]
        ] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        batch_size: int = 8,
        **extra_generation_kwargs,
    ) -> List[List[str]]:
        """
        Async‐parallel version of chat_batch: fires up to `batch_size`
        generate_content_async calls in parallel, each with its own
        system_message[i] and tools[i].
        """
        if batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        sem = asyncio.Semaphore(batch_size)
        tasks: List[asyncio.Task] = []

        for i, msgs in enumerate(all_messages):
            # pick per‐conv system_message
            if isinstance(system_message, list):
                sys_i = system_message[i]
            else:
                sys_i = system_message  # same or None

            # pick per‐conv tools
            if not tools:
                tools_i = None
            elif isinstance(tools[0], dict):
                tools_i = tools  # single global list
            else:
                tools_i = tools[i]  # per‐conv list

            # each worker acquires/releases the semaphore
            async def worker(
                conv_msgs=msgs,
                sys_msg=sys_i,
                tools_list=tools_i,
            ) -> List[str]:
                acquired = False
                try:
                    await sem.acquire()
                    acquired = True
                    return await self._chat_single_async(
                        model_name=model_name,
                        messages=conv_msgs,
                        system_message=sys_msg,
                        tools=tools_list,
                        generation_overrides=generation_overrides,
                        **extra_generation_kwargs,
                    )
                finally:
                    if acquired:
                        sem.release()

            tasks.append(asyncio.create_task(worker()))

        # wait for all of them
        return await asyncio.gather(*tasks)

    async def _chat_single_async(
        self,
        model_name: str,
        messages: List[Tuple[str, str]],
        system_message: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        generation_overrides: Optional[Dict[str, Any]],
        **extra_generation_kwargs,
    ) -> List[str]:
        """
        Exactly one async call to generate_content_async for a single convo.
        """
        # 1) build model handle with per‐conv system_instruction
        if system_message:
            model = GenerativeModel(model_name, system_instruction=[system_message])
        else:
            model = GenerativeModel(model_name)

        # 2) translate tools
        vertex_functions = self._openai_tools_to_vertex(tools)
        tool_obj = (
            Tool(function_declarations=vertex_functions)
            if vertex_functions
            else None
        )

        # 3) build generation_config
        gen_cfg = copy.deepcopy(self._generation_defaults)
        if generation_overrides:
            gen_cfg.update(generation_overrides)
        gen_cfg.update(extra_generation_kwargs)

        # 4) build the Content list
        vertex_messages = [
            Content(role=role, parts=[Part.from_text(text)])
            for role, text in messages
        ]

        # 5) fire the async call
        if tool_obj:
            fut = model.generate_content_async(
                vertex_messages,
                generation_config=gen_cfg,
                tools=[tool_obj],
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                    )
                ),
            )
        else:
            fut = model.generate_content_async(
                vertex_messages,
                generation_config=gen_cfg,
            )

        resp = await fut

        # 6) unpack
        try:
            parts = resp.to_dict()["candidates"][0]["content"]["parts"]
        except Exception as e:
            print(resp.to_dict())
            raise e
        return [p.get("text", p.get("function_call")) for p in parts]

    def _chat_impl(
        self,
        batch_messages: List[List[Tuple[str, str]]],
        model_name: str,
        tools: Optional[List[Dict[str, Any]]],
        system_message: Optional[str],
        generation_overrides: Optional[Dict[str, Any]],
        **extra_generation_kwargs,
    ) -> List[Any]:
        """
        unchanged synchronous batch of size=1 under the hood
        """
        # … your existing implementation of _chat_impl …
        raise NotImplementedError("Legacy sync path left unchanged.")

    @staticmethod
    def _openai_tools_to_vertex(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[FunctionDeclaration]]:
        if not tools:
            return None
        vertex_fns: List[FunctionDeclaration] = []
        for tool in tools:
            if tool.get("type") != "function" or "function" not in tool:
                continue
            fn = tool["function"]
            original_params = fn.get("parameters", {})
            if isinstance(original_params, dict):
                cleaned = {
                    k: v for k, v in original_params.items()
                    if k != "additionalProperties"
                }
            else:
                cleaned = original_params or {}
            vertex_fns.append(
                FunctionDeclaration(
                    name=fn.get("name"),
                    description=fn.get("description", ""),
                    parameters=cleaned,
                )
            )
        return vertex_fns or None