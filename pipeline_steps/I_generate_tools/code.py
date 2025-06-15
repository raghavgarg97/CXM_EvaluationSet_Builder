import os
import asyncio
import aiohttp
import pandas as pd
import json
import copy
import pickle as pkl
import logging
from tqdm.auto import tqdm
import random

import tiktoken  # for token counting

from utils.misc import parse_json

# =============================================================================
# Global LLM logs collector
# =============================================================================
prompt_encoding = tiktoken.encoding_for_model("gpt-4")
LLM_LOGS = []

# =============================================================================
# Pipeline step name for prompt loading
# =============================================================================
CURR_PIPELINE_STEP = "I_generate_tools"

def load_prompt(prompt_name: str) -> str:
    path = f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}'
    with open(path, 'r') as file:
        return file.read()

# =============================================================================
# Constants
# =============================================================================
llm_config_path = './configs/llm_config.json'
URL = json.load(open(llm_config_path))['server_url'].replace('batch-chat-completion', 'chat-completion')

MODEL_SKILL      = "gemini-2.0-flash-001"
MODEL_DEDUPE     = "o4-mini"
# MODEL_DEDUPE     = "gemini-2.0-flash-001"
# MODEL_DISAMBIG   = "o4-mini"
MODEL_DISAMBIG   = "o4-mini"
MODEL_TOOL_CREATION = "gpt-4o"
RPS              = 3.0

# for reclustering
N_TOOLS = 150
USE_CASES_THRESHOLD = 20
TEMPERATURE = 0.1

CREATE_TOOLS_PROMPT = '''You are given an external tool and it's definition. You are also given some of the use cases of the tool. Your task is to create a tool for it in the format given below:

Tool Name: {tool_name}
Tool Description: {tool_description}

Use cases corresponding to the tool:
(Some of the use cases might not be for this case, feel free to ignore them and only focus on the ones that align with the Tool Description.)
```
{use_cases_str}
```

# Tool Generation Instructions
When creating a tool schema, follow these guidelines:

1. Core Schema Components:
- Name: Choose a clear, descriptive function name (e.g., "get_weather")
- Description: Provide detailed explanation of when and how to use the function
- Parameters: Define input arguments using JSON schema format

2. Parameter Definition Best Practices:
- Use descriptive property names and types
- Include clear descriptions for each parameter
- Keep only the minimum relevant parameters.
- For this task use only strings for input parameters [IMPORTANT]

3. Design Principles:
- Make functions intuitive and obvious in their purpose
- Provide comprehensive documentation that a new user could understand

Example Schema Format:
{{
    "type": "function", 
    "function": {{
        "name": "get_weather",
        "description": "Retrieves current weather for the given location.",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }},
                "units": {{
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Units the temperature will be returned in."
                }}
            }},
            "required": ["location", "units"],
            "additionalProperties": false
        }},
        "strict": true
    }}
}}

Output only the JSON schema for the tool and nothing else.
'''


# =============================================================================
# LLM helper calls
# =============================================================================
async def call(payload, timeout=300):
    # 1) compute input_tokens

    prompt = "".join(m.get("content", "") for m in payload.get("messages", []))
    input_tokens = len(prompt_encoding.encode(prompt))

    # 2) make the HTTP call
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as sess:
        async with sess.post(URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(text)
                raise Exception(f"{resp.status} status code found")
            res_json = await resp.json()

    # 3) extract the output text
    try:
        output = res_json["choices"][0]["message"]["content"]
    except:
        output = json.dumps(res_json)

    # 4) log this call
    LLM_LOGS.append({
        "model":        payload.get("model"),
        "input_tokens": input_tokens,
        "output":       output
    })

    return res_json

async def sleeped_call(sleep_time, payload, timeout=300):
    await asyncio.sleep(sleep_time)
    return await call(payload, timeout)

async def simulate(rps, payload_batch, sleep_time=0, timeout=300):
    await asyncio.sleep(sleep_time)
    interval = 1.0 / rps
    calls = [sleeped_call(i * interval, pl, timeout) for i, pl in enumerate(payload_batch)]
    return await tqdm.gather(*calls)

# =============================================================================
# Step 1: Identify external steps in each KB
# =============================================================================
async def identify_skills(df: pd.DataFrame) -> list:
    prompt = load_prompt("identify_skills.txt")
    current_pl = {
        "client_identifier": "ml-ca-dev",
        "model": MODEL_SKILL,
        "provider": "GOOGLE_BARD" if 'gemini' in MODEL_SKILL else 'OPEN_AI',
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": ""}],
        "temperature": 0.0,
        "tracking_params": {"release": "ca_research"},
    }

    payloads = []
    for _, row in df.iterrows():
        pl = copy.deepcopy(current_pl)
        pl["messages"][0]["content"] = prompt.format(KNOWLEDGE_BASE=row["kb_article"])
        payloads.append(pl)

    results = await simulate(RPS, payloads)
    identified = []
    for res in results:
        try:
            txt = res["choices"][0]["message"]["content"].strip("```").strip("json")
            identified.append(json.loads(txt))
        except:
            identified.append({})

    external_tools = [x.get('external_step_dictionary', {}) for x in identified]
    all_tools = []
    for i, tool_dict in enumerate(external_tools):
        for tool_name, tool_description in tool_dict.items():
            all_tools.append({
                'tool_name': tool_name,
                'tool_description': tool_description,
                'kb_uID': df.iloc[i]['uID']
            })
    return all_tools

# =============================================================================
# Step 2: Deduplicate / merge the external steps across all KBs
# =============================================================================
def ensure_tmp_dir():
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    # Clear any existing log files before starting new ones
    for log_name in ['skills.log', 'skills_2.log']:
        log_path = os.path.join(tmp_dir, log_name)
        if os.path.exists(log_path):
            os.remove(log_path)
    return tmp_dir

async def dedupe_and_merge(df: pd.DataFrame):
    prompt = load_prompt("dedupe_skills.txt")
    base_pl = {
        "client_identifier": "ml-ca-dev",
        "model": MODEL_DEDUPE,
        "provider": "GOOGLE_BARD" if 'gemini' in MODEL_DEDUPE else 'OPEN_AI',
        "max_completion_tokens": 4096,
        "messages": [{"role": "user", "content": ""}],
        "temperature": 1.0,
        "tracking_params": {"release": "ca_research"},
    }
    ensure_tmp_dir()  # Ensure tmp dir exists
    logger = logging.getLogger("dedupe")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.path.dirname(__file__), 'tmp', 'skills.log')
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path, mode="w"))

    currently_identified = {}
    kb_mapped = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # persist progress
        pkl.dump((currently_identified, kb_mapped), open("results.pkl", "wb"))

        # build current list string
        lst = "\n".join(f"{i+1}. {k} : {v[0]}"
                        for i, (k, v) in enumerate(currently_identified.items())) or "empty"
        pl = copy.deepcopy(base_pl)
        pl["messages"][0]["content"] = prompt.format(lst, f"{row.tool_name} : {row.tool_description}")

        # call LLM synchronously
        res = await call(pl)
        try:
            out = json.loads(res["choices"][0]["message"]["content"]
                             .strip("```").strip("json"))
        except:
            out = {"decision": "ignore"}

        # apply per-decision logic
        new_local_skills = []
        decision = out.get("decision", "").strip().lower()
        if decision == "ignore":
            shared = out.get("shared_step_name")
            if shared:
                new_local_skills.append(shared)
                if shared in currently_identified:
                    currently_identified[shared][1].append(idx)
                else:
                    currently_identified[shared] = ("", [idx])
        elif decision == "create_new":
            name = out.get("new_name")
            desc = out.get("new_desc", "")
            if name:
                new_local_skills.append(name)
                currently_identified[name] = (desc, [idx])
        elif decision == "update_old":
            old = out.get("old_step_name")
            new_name = out.get("new_name")
            desc = out.get("new_desc", "")
            if old in currently_identified and new_name:
                old_idxs = currently_identified[old][1]
                # rename in historical mappings
                for i in old_idxs:
                    if i == len(kb_mapped):
                        new_local_skills = [new_name if t == old else t for t in new_local_skills]
                    else:
                        kb_mapped[i] = [new_name if t == old else t for t in kb_mapped[i]]
                # merge index lists
                merged = list(set(old_idxs + [idx]))
                currently_identified[new_name] = (desc, merged)
                if new_name != old:
                    del currently_identified[old]
                new_local_skills.append(new_name)
        # else: unrecognized decision, ignore

        kb_mapped.append(list(set(new_local_skills)))

    return currently_identified, kb_mapped

# =============================================================================
# Step 3: Disambiguate / split and finalize tools
# =============================================================================
async def disambiguate_tools(currently_identified: dict, df_tools):
    prompt = load_prompt("disambiguate_tools.txt")

    # sort tools by how many KB entries they cover
    sorted_skills = sorted(currently_identified.items(),
                           key=lambda i: len(i[1][1]), reverse=True)
    df2 = pd.DataFrame({
        "tool_name":        [k for k, _  in sorted_skills],
        "tool_description": [v[0] for _, v in sorted_skills],
        "composition":      [v[1] for _, v in sorted_skills],
    }).sample(frac=1).reset_index(drop=True)

    base_pl = {
        "client_identifier": "ml-ca-dev",
        "model": MODEL_DISAMBIG,
        "provider": "GOOGLE_BARD" if 'gemini' in MODEL_DISAMBIG else 'OPEN_AI',
        "max_completion_tokens": 4096,
        "messages": [{"role": "user", "content": ""}],
        "temperature": 1.0,
        "tracking_params": {"release": "ca_research"},
    }
    ensure_tmp_dir()  # Ensure tmp dir exists
    logger = logging.getLogger("disambig")
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.path.dirname(__file__), 'tmp', 'skills_2.log')
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path, mode="w"))

    new_current = {}
    new_mapped  = []

    for idx, row in tqdm(df2.iterrows(), total=len(df2)):
        # persist progress
        pkl.dump((new_current, new_mapped), open("results_2.pkl", "wb"))

        # build current tool list string
        lst = "\n".join(f"{i+1}. {n} : {d[0]}"
                        for i, (n, d) in enumerate(new_current.items())) or "empty"
        pl = copy.deepcopy(base_pl)
        pl["messages"][0]["content"] = prompt.format(lst,
                                                     f"{row.tool_name} : {row.tool_description}")

        res = await call(pl)
        try:
            decs = json.loads(res["choices"][0]["message"]["content"]
                              .strip("```").strip("json"))
            if not isinstance(decs, list):
                raise ValueError
        except:
            decs = [{"decision": "ignore", "existing_tool": row.tool_name}]

        # apply per-decision logic
        new_local_skills = []
        for out in decs:
            decision = out.get("decision", "").strip().lower()
            if decision == "ignore":
                existing = out.get("existing_tool")
                if existing:
                    new_local_skills.append(existing)
                    if existing in new_current:
                        new_current[existing][1].append(idx)
                    else:
                        new_current[existing] = ("", [idx])
            elif decision == "create_new":
                name = out.get("new_name")
                desc = out.get("new_desc", "")
                if name:
                    new_local_skills.append(name)
                    new_current[name] = (desc, [idx])
            elif decision == "update_existing":
                existing = out.get("existing_tool")
                name = out.get("new_name")
                desc = out.get("new_desc", "")
                if existing in new_current and name:
                    old_idxs = new_current[existing][1]
                    # rename in historical mappings
                    for i in old_idxs:
                        if i == len(new_mapped):
                            new_local_skills = [name if t == existing else t for t in new_local_skills]
                        else:
                            new_mapped[i] = [name if t == existing else t for t in new_mapped[i]]
                    merged = list(set(old_idxs + [idx]))
                    new_current[name] = (desc, merged)
                    if name != existing:
                        del new_current[existing]
                    new_local_skills.append(name)
            # else: unrecognized decision, ignore it

        new_mapped.append(list(set(new_local_skills)))

    # build final tools_dict
    # df_tools = pd.DataFrame(all_tools)
    tools_dict = {}
    for k, (desc, comps) in new_current.items():
        tools_df_comp = list(set([y for _, r in df2.iloc[comps].iterrows() for y in r['composition']]))
        expanded_comp = [dict(r) for _, r in df_tools.iloc[tools_df_comp].iterrows()]
        tools_dict[k] = {
            "description": desc,
            "composition": expanded_comp
        }
    return tools_dict


# =============================================================================
# Step 4: Generate JSON‐schema functions for each tool
#         (now using simulate instead of LLM.chat_batch)
# =============================================================================
async def recluster_tools(brand_name, tools_dict):
    ckpt_dir = f'./checkpoints/I_generate_tools/{brand_name}'
    # in_path = os.path.join(ckpt_dir, 'tools_dict_o4.pkl')
    # tools_dict = pkl.load(open(in_path, 'rb'))

    # keep only top N_TOOLS by composition length
    tools_dict = dict(
        sorted(tools_dict.items(), key=lambda x: len(x[1]['composition']), reverse=True)[:N_TOOLS]
    )

    # build prompts
    all_tool_prompts = []
    for tool_name, info in tools_dict.items():
        curr_use_cases = random.sample(
            info['composition'],
            k=min(USE_CASES_THRESHOLD, len(info['composition']))
        )
        use_cases = [f"{i+1}. {uc['tool_description']}" for i, uc in enumerate(curr_use_cases)]
        prompt = CREATE_TOOLS_PROMPT.format(
            use_cases_str="\n".join(use_cases),
            tool_name=tool_name,
            tool_description=info['description'],
        )
        all_tool_prompts.append((tool_name, prompt))

    # build payloads for simulate()
    base_pl = {
        "client_identifier": "ml-ca-dev",
        "model": MODEL_TOOL_CREATION,
        "provider": "GOOGLE_BARD" if 'gemini' in MODEL_TOOL_CREATION else 'OPEN_AI',
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": ""}],
        "temperature": TEMPERATURE,
        "tracking_params": {"release": "ca_research"},
    }

    payloads = []
    for _, prompt in all_tool_prompts:
        pl = copy.deepcopy(base_pl)
        pl["messages"][0]["content"] = prompt
        payloads.append(pl)

    # call out in one go (simulate handles the RPS throttle)
    responses = await simulate(RPS, payloads)

    # parse JSON responses
    tool_fns = []
    for res in responses:
        try:
            txt = res["choices"][0]["message"]["content"]
            parsed = parse_json(txt)
        except:
            parsed = None
        tool_fns.append(parsed)

    print(f"Failed to parse {tool_fns.count(None)} / {len(tool_fns)} schemas")

    # attach into tools_dict
    for (tool_name, _), fn in zip(all_tool_prompts, tool_fns):
        tools_dict[tool_name]['tool'] = fn

    # enforce strict + required
    for info in tools_dict.values():
        fn = info['tool']['function']
        fn['strict'] = True
        fn['parameters']['required'] = list(fn['parameters']['properties'].keys())

    return tools_dict

# =============================================================================
# Orchestration
# =============================================================================
async def process(brand_name: str):
    issues_df = pd.read_excel(f'./checkpoints/G_generate_issue_kbs/{brand_name}/issue_kbs.xlsx')
    all_tools = await identify_skills(issues_df)

    tools_df = pd.DataFrame(all_tools)

    # 2. dedupe & merge
    currently_identified, kb_mapped = await dedupe_and_merge(tools_df)

    # 3. disambiguate
    tools_dict = await disambiguate_tools(currently_identified, tools_df)

    # save
    save_dir = f"./checkpoints/I_generate_tools/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump(tools_dict, open(f"{save_dir}/tools_dict_o4.pkl", "wb"))
    print(f"Preliminary tools_dict saved to {save_dir}/tools_dict_o4.pkl")

    # 4. generate final function schemas and save
    final_tools_dict = await recluster_tools(brand_name, tools_dict)

    # dump final pickle
    ckpt_dir = f'./checkpoints/I_generate_tools/{brand_name}'
    out_path = os.path.join(ckpt_dir, 'tools_dict_o4_final.pkl')
    pkl.dump(tools_dict, open(out_path, 'wb'))
    print(f"Re-clustering complete, final saved to {out_path}")

    # Dump the LLM call logs
    logs_path = os.path.join(ckpt_dir, 'llm_logs.json')
    with open(logs_path, 'w') as f:
        json.dump(LLM_LOGS, f, indent=2)
    print(f"LLM logs saved to {logs_path}")

    print(f"Completed pipeline for brand: {brand_name}")
    return final_tools_dict
