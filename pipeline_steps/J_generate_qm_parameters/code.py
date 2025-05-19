import asyncio
import os
import sys
import json
import pickle
import random
import datetime
import pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm.auto import tqdm

from constants import *
from utils.llm_helper import LLM
from utils.misc import parse_json

CURR_PIPELINE_STEP = "J_generate_qm_parameters"
MODEL_TO_USE = "gemini-flash"
llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def dict_to_key_value_lines(data: dict) -> str:
    if not data:
        return "None yet."
    lines = []
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

def dict_to_key_lines_numbered(data: dict) -> str:
    """
    Converts dictionary keys into numbered lines.
    """
    if not data:
        return "None yet."
    lines = []
    for i, k in enumerate(data.keys()):
        lines.append(f"{i + 1}. {k}")
    return "\n".join(lines)

def load_prompt(prompt_name: str) -> str:
    """
    Loads a prompt from the prompts directory for a given pipeline step.
    """
    prompts_dir = f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts'
    prompt_path = os.path.join(prompts_dir, prompt_name)
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

async def process_new_question(
    new_question: str,
    row_indices: List[int],
    existing_questions: Dict[str, str],
    existing_questions_composition: Dict[str, List[int]],
    prompts: Dict[str, str],
):
    """
    Process a single new evaluation question through scenarios (IGNORE, UPDATE_EXISTING, CREATE_NEW)
    and update question collections accordingly.
    """
    existing_questions_str = dict_to_key_lines_numbered(existing_questions)

    # Prepare scenario prompts
    ignore_scenario_prompt = prompts['ignore'].format(
        current_questions=existing_questions_str,
        new_question=new_question
    )
    update_scenario_prompt = prompts['update_existing'].format(
        current_questions=existing_questions_str,
        new_question=new_question
    )
    create_scenario_prompt = prompts['create_new'].format(
        current_questions=existing_questions_str,
        new_question=new_question
    )

    # Run scenarios concurrently (3 prompts) – same logic as reference
    scenario_results = await llm.chat_batch(
        [
            ignore_scenario_prompt,
            update_scenario_prompt,
            create_scenario_prompt
        ],
        task_name='evaluate_question_scenarios',
        model_name=MODEL_TO_USE,
        batch_size=3,
        temperature=0.1,
        use_tqdm=False
    )

    # print('\n\n'.join([
    #         ignore_scenario_prompt,
    #         update_scenario_prompt,
    #         create_scenario_prompt
    # ]))

    ignore_output = parse_json(scenario_results[0])
    update_output = parse_json(scenario_results[1])
    create_output = parse_json(scenario_results[2])

    # Convert scenario results for verification
    ignore_output_str = dict_to_key_value_lines(ignore_output)
    update_output_str = dict_to_key_value_lines(update_output)
    create_output_str = dict_to_key_value_lines(create_output)

    # Prepare verifier prompt
    verifier_prompt = prompts['verifier'].format(
        ignore_output=ignore_output_str,
        update_output=update_output_str,
        create_new_output=create_output_str,
        new_question=new_question,
        current_questions=existing_questions_str
    )

    # print("verifier prompt: ", verifier_prompt)
    verifier_response = await llm.chat(
        verifier_prompt,
        task_name='verify_question_scenarios',
        model_name='gpt-mini',
        add_prompter=True,
        temperature=0.1
    )
    verifier_decision = parse_json(verifier_response)
    decision = verifier_decision.get("decision", "")

    # If no existing questions yet, force creation if not decided
    if len(existing_questions) == 0 and decision != "CREATE_NEW":
        decision = "CREATE_NEW"

    print(f"Decision: {decision}")

    # Update based on verifier decision
    if decision == "IGNORE":
        matched_question = ignore_output.get("existing_question", "")
        if matched_question:
            existing_questions_composition.setdefault(matched_question, []).extend(row_indices)

    elif decision == "UPDATE_EXISTING":
        old_question = update_output.get("old_question", "")
        merged_question = update_output.get("merged_question", "")

        if old_question in existing_questions:
            del existing_questions[old_question]

        existing_questions[merged_question] = "Auto-merged evaluation question."
        indices = existing_questions_composition.pop(old_question, [])
        existing_questions_composition.setdefault(merged_question, []).extend(indices)
        existing_questions_composition[merged_question].extend(row_indices)

    elif decision == "CREATE_NEW":
        new_general_question = create_output.get("new_question", "")
        existing_questions[new_general_question] = "Generalized evaluation question."
        existing_questions_composition.setdefault(new_general_question, []).extend(row_indices)

async def process_questions_stream(
    list_of_questions: List[Dict[str, Any]],
    prompts: Dict[str, str],
) -> Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Process a stream of new questions, updating a global dictionary of
    questions and their compositions. Periodically saves intermediate checkpoints.
    """
    existing_questions: Dict[str, str] = {}
    existing_questions_composition: Dict[str, List[int]] = {}
    existing_questions_issue_composition: Dict[str, List[int]] = {}

    # Scenario checks for each question
    for idx, question_object in tqdm(enumerate(list_of_questions), total=len(list_of_questions)):
        new_question = question_object['question']
        await process_new_question(
            new_question,
            [idx],
            existing_questions,
            existing_questions_composition,
            prompts,
        )

    # Final pass to build the issue composition
    for qtext, indices in existing_questions_composition.items():
        existing_questions_issue_composition[qtext] = list(
            set(list_of_questions[i]['issue_index'] for i in indices)
        )

    return existing_questions, existing_questions_composition, existing_questions_issue_composition

async def process_multiple_question_streams(
    questions_dict: Dict[str, List[Dict[str, Any]]],
    n_semaphore: int,
    prompts: Dict[str, str],
) -> Dict[str, Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, List[int]]]]:
    """
    Process multiple sets (keys) of questions concurrently,
    each set labeled by the dictionary key. Results are stored in a time-stamped folder.
    """
    time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_folder = f"./results/{time_stamp}"
    os.makedirs(results_folder, exist_ok=True)

    semaphore = asyncio.Semaphore(n_semaphore)

    async def _process_with_semaphore(key: str, list_of_questions: List[Dict[str, Any]]):
        async with semaphore:
            return key, await process_questions_stream(
                list_of_questions,
                prompts=prompts,
            )

    tasks = []
    for key, list_of_q in questions_dict.items():
        tasks.append(asyncio.create_task(_process_with_semaphore(key, list_of_q)))

    results = await asyncio.gather(*tasks)
    return {k: v for k, v in results}

################################################################################
# Building / Handling the Extracted Questions
################################################################################

async def get_qm_ques(
    l3_intents: Dict[str, Any],
    issues_df
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each key in l3_intents, fetch the composition,
    generate prompts for each row, call LLM in batch,
    parse responses, and aggregate them back by key.
    """
    # Load the question extraction prompt template
    qm_question_extraction_template = load_prompt('qm_simple_question_extraction')

    all_prompts = []
    qm_prompt_issue_indices = []
    prompt_key_map = []

    for k, v in l3_intents.items():
        intent_def = f"{k} - {v['description']}"
        df_temp = issues_df[issues_df.l3_intent == k].reset_index(drop=True)

        # Create a prompt for each row
        qm_prompts = [
            qm_question_extraction_template.format(
                intent_def=intent_def,
                KB=row[KB_ARTICLE]
            )
            for _, row in df_temp.iterrows()
        ]
        qm_prompt_issue_indices.extend(df_temp['uID'].tolist())
        all_prompts.extend(qm_prompts)
        prompt_key_map.extend([k] * len(qm_prompts))

    qm_results = await llm.chat_batch(
        all_prompts,
        temperature=0.6,
        batch_size=10,
        model_name='gpt',  # Unchanged from original logic for extraction
        task_name='get_qm_questions'
    )

    # Parse & group
    qm_questions_dict = {k: [] for k in l3_intents.keys()}
    for key, response, issue_idx in zip(prompt_key_map, qm_results, qm_prompt_issue_indices):
        lines = [x for x in response.splitlines() if x.strip()]
        parsed_questions = []
        # Collect lines with "number." prefix
        for ln in lines:
            parts = ln.split('.', 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                parsed_questions.append(parts[1].strip())

        for question in parsed_questions:
            qm_questions_dict[key].append({"question": question, "issue_index": issue_idx})

    return qm_questions_dict

def build_top_questions_dict(
    results_detailed: Dict[str, Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, List[int]]]]
) -> Dict[str, Dict[str, List[int]]]:
    """
    Sort each set of final questions in descending order
    based on the number of issues referencing them.
    """
    qm_top_questions: Dict[str, Dict[str, List[int]]] = {}

    for intent, (dict1, dict2, dict3) in results_detailed.items():
        # Sort by composition length
        sorted_keys = sorted(
            dict3.keys(),
            key=lambda k: len(dict3[k]),
            reverse=True
        )
        sorted_dict = {k: dict3[k] for k in sorted_keys}
        qm_top_questions[intent] = sorted_dict

    return qm_top_questions

################################################################################
# Data Loading Helpers
################################################################################

def load_l3_intents(brand_name: str) -> Dict[str, Any]:
    """
    Load the L3 intents from the checkpoint path for the given brand.
    """
    path = f'./checkpoints/E_cluster_intents/{brand_name}/clustered_issues.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_issues_df(brand_name: str) -> pd.DataFrame:
    """
    Load issues DataFrame from checkpoints for the given brand.
    """
    df_path = f'./checkpoints/G_generate_issue_kbs/{brand_name}/issue_kbs.xlsx'
    return pd.read_excel(df_path)

################################################################################
# Saving Final Generated Data
################################################################################

def save_generated_qm_data(
    questions_dict: Dict[str, List[Dict[str, Any]]],
    results: Dict[str, Tuple[Dict[str, str], Dict[str, List[int]], Dict[str, List[int]]]],
    qm_top_questions: Dict[str, Dict[str, List[int]]],
    brand_name: str,
):
    """
    Saves the generated QM data (questions, scenario details, top questions)
    to a checkpoint folder for the given brand and pipeline step.
    """
    save_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "qm_questions_dict.pkl"), "wb") as f:
        pickle.dump(questions_dict, f)

    with open(os.path.join(save_dir, "qm_results_detailed.pkl"), "wb") as f:
        pickle.dump(results, f)

    with open(os.path.join(save_dir, "qm_top_questions.pkl"), "wb") as f:
        pickle.dump(qm_top_questions, f)

################################################################################
# High-Level Process
################################################################################

async def process(brand_name: str, n_semaphore) -> Dict[str, Dict[str, List[int]]]:
    """
    High-level pipeline for generating QM parameters:
      1) Load l3_intents and issues_df
      2) Extract initial questions from KB
      3) Organize & unify them with scenario logic
      4) Build sorted question reference
      5) Save final artifacts & return the structured data
    """
    print(f"[{CURR_PIPELINE_STEP}] Loading data...")

    l3_intents = load_l3_intents(brand_name)
    issues_df = load_issues_df(brand_name)
    
    questions_dict = await get_qm_ques(l3_intents, issues_df)

    # Randomize order of questions
    for k, v in questions_dict.items():
        questions_dict[k] = random.Random(1).sample(v, k=len(v))

    # Load scenario prompts
    prompts = {
        'ignore': load_prompt('ignore'),
        'update_existing': load_prompt('update_old'),
        'create_new': load_prompt('create_new'),
        'verifier': load_prompt('verifier')
    }

    results = await process_multiple_question_streams(
        questions_dict,
        n_semaphore=n_semaphore,
        prompts=prompts,
    )

    # Build final top-questions dictionary
    qm_top_questions = build_top_questions_dict(results)

    # Save final artifacts
    save_generated_qm_data(questions_dict, results, qm_top_questions, brand_name)

    print(f"[{CURR_PIPELINE_STEP}] Finished generating QM parameters.")
    return qm_top_questions
