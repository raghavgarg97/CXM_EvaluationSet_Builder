import os
import json
import asyncio
import pickle as pkl
from typing import Any, List

from utils.llm_helper import LLM

CURR_PIPELINE_STEP = "L_verify_qm_paramters"
MODEL_NAME = "gemini-flash"
BATCH_SIZE = 5
TEMPERATURE = 1
MAX_COMPLETION_TOKENS = 4096

PROMPT_DIR = os.path.join("pipeline_steps", CURR_PIPELINE_STEP, "prompts")
PROMPT_FILE_GET = "get_answers"
PROMPT_FILE_EVAL = "evaluate_answers"
PROMPT_FILE_FIX = "fix_answers"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompt(file_name: str) -> str:
    path = os.path.join(PROMPT_DIR, file_name)
    with open(path, "r") as f:
        return f.read()

TEMPLATE_GET = load_prompt(PROMPT_FILE_GET)
TEMPLATE_EVAL = load_prompt(PROMPT_FILE_EVAL)
TEMPLATE_FIX = load_prompt(PROMPT_FILE_FIX)

def safe_eval(data: str, default: Any = None) -> Any:
    try:
        return eval(data)
    except:
        return default

def extract_json_list(text: str) -> str:
    if not text:
        return ""
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        return text[start:end+1]
    return text


import pandas as pd


def extract_convs(conv_objects):
    ret = []
    for conv_object in conv_objects:
        extracted_conv = []
        for i, (role, msg, *others) in enumerate(conv_object):
            role = 'AGENT' if role == 'bot_agent' else 'CUSTOMER'
            extracted_conv.append(f"{role}[{i + 1}]:{msg}")
        ret.append('\n'.join(extracted_conv))
    return ret


def extract_questions(agent_personas):
    return ['\n'.join([f"{i + 1}. {x}" for i, x in enumerate(x['qm_dict'].keys())]) for x in agent_personas]

def loadp(file_path):
    return pkl.load(open(file_path, "rb"))

def get_qm_df(brand_name):
    convs = loadp(f'./checkpoints/K_simulate_conversations/{brand_name}/conv_objects.pkl')
    agent_personas, _ = loadp(f'./checkpoints/K_simulate_conversations/{brand_name}/personas.pkl')
    assert len(convs) == len(agent_personas)
    final_convs = extract_convs(convs)
    final_ques = extract_questions(agent_personas)

    df = pd.DataFrame({'Conversation': final_convs, 'Questions': final_ques})
    return df

async def process(brand_name):
    # Load data
    df = get_qm_df(brand_name)
    print(f"[{CURR_PIPELINE_STEP}] Loaded {len(df)} rows for brand {brand_name}")

    # Step 1: Generate GPT answers
    df['ans_prompt'] = df.apply(
        lambda r: TEMPLATE_GET.format(conv=r['Conversation'], questions=r['Questions']),
        axis=1
    )
    ans_responses = await llm.chat_batch(
        df['ans_prompt'].tolist(),
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        task_name="qm_generation",
        add_prompter=False,
        use_tqdm=False
    )
    df['gpt-answers'] = [
        safe_eval(extract_json_list(resp)) for resp in ans_responses
    ]
    bad_idx = [i for i, ans in enumerate(df['gpt-answers']) if ans is None]
    df.drop(bad_idx, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 2: Evaluate GPT answers
    df['eval_prompt'] = df.apply(
        lambda r: TEMPLATE_EVAL.format(
            conv=r['Conversation'],
            questions=r['Questions'],
            answers=json.dumps(r['gpt-answers'], indent=4)
        ),
        axis=1
    )
    eval_responses = await llm.chat_batch(
        df['eval_prompt'].tolist(),
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        task_name="qm_evaluation",
        add_prompter=False,
        use_tqdm=False
    )
    df['evaluated_answers'] = [
        safe_eval(extract_json_list(resp)) for resp in eval_responses
    ]
    bad_eval_idx = [i for i, ans in enumerate(df['evaluated_answers']) if ans is None]
    df.drop(bad_eval_idx, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Step 3: Fix incorrect answers
    needs_fix = df['evaluated_answers'].apply(
        lambda ans: any(q.get('Validation', '').lower() != 'correct' for q in ans)
    )
    to_fix = df[needs_fix].copy()
    if not to_fix.empty:
        to_fix['fix_prompt'] = to_fix.apply(
            lambda r: TEMPLATE_FIX.format(
                conv=r['Conversation'],
                questions=r['Questions'],
                insights=json.dumps(r['evaluated_answers'], indent=4)
            ),
            axis=1
        )
        fix_responses = await llm.chat_batch(
            to_fix['fix_prompt'].tolist(),
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            batch_size=BATCH_SIZE,
            task_name="qm_fixing",
            add_prompter=False,
            use_tqdm=False
        )
        to_fix['fixed_answers'] = [
            safe_eval(extract_json_list(resp), []) for resp in fix_responses
        ]
        for idx, row in to_fix.iterrows():
            for i, fixed in enumerate(row['fixed_answers']):
                if i < len(row['evaluated_answers']) and row['evaluated_answers'][i].get('Validation','').lower() != 'correct':
                    if i < len(df.at[idx, 'gpt-answers']):
                        df.at[idx, 'gpt-answers'][i] = fixed

    # Step 4: Cleanup proof_message_ids
    for _, row in df.iterrows():
        for qa in row['gpt-answers']:
            if qa.get('Answer') in ['No', 'NA']:
                qa['proof_message_ids'] = []

    # Step 5: Serialize answer columns and save
    for col in df.columns:
        if 'answer' in col.lower():
            df[col] = df[col].apply(lambda x: json.dumps(x, indent=4))

    out_path = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}/qm_questions.xlsx"
    df.to_excel(out_path, index=False)
    print(f"[{CURR_PIPELINE_STEP}] Saved results to {out_path}, {len(df)} rows.")

if __name__ == '__main__':
    asyncio.run(process('TerraBloom'))