import os
import sys
import json
import random
import pickle as pkl
import asyncio
from typing import Dict, Any, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from utils.llm_helper import LLM
from utils.misc import parse_json

CURR_PIPELINE_STEP = "F_cluster_issue_kbs"
MODEL_TO_USE = "gemini-flash"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompt(prompt_name: str) -> str:
    """
    Loads a prompt text file from the disk for this pipeline step.
    """
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}', 'r') as file:
        return file.read()

def get_prompts() -> Dict[str, str]:
    """
    Returns the collection of scenario and verifier prompts.
    """
    prompts = {
        "ignore": load_prompt("enforce_ignore"),
        "create": load_prompt("enforce_create_new"),
        "update": load_prompt("enforce_update_old"),
        "verifier": load_prompt("verify_scenarios")
    }
    return prompts

def dict_to_key_value_lines(data: dict) -> str:
    """
    Converts a dict to simple key:value lines.
    """
    if not data:
        return "None yet."
    lines = []
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

def dict_to_key_value_lines_numbered(data: dict) -> str:
    """
    Converts a dict to numbered key:value lines.
    """
    if not data:
        return "None yet."
    lines = []
    for i, (k, v) in enumerate(data.items()):
        lines.append(f"{i+1}. {k}: {v}")
    return "\n".join(lines)

async def process_new_issue(
    new_issue_title: str,
    new_issue_description: str,
    row_indices: List[int],
    existing_issues: Dict[str, str],
    existing_issues_composition: Dict[str, List[int]],
    prompts: Dict[str, str]
) -> None:
    """
    For a single new issue, this function:
    1. Runs scenario 1 (ignore), scenario 2 (update), scenario 3 (create) in parallel.
    2. Parses scenario outputs.
    3. Runs the verifier scenario to decide final action.
    4. Updates existing_issues and existing_issues_composition accordingly.
    """
    existing_issues_str = dict_to_key_value_lines_numbered(existing_issues)
    new_issue_str = f"title: {new_issue_title}\ndescription: {new_issue_description}"

    ignore_scenario_prompt = prompts["ignore"].format(
        current_issues=existing_issues_str,
        new_issue=new_issue_str
    )
    update_scenario_prompt = prompts["update"].format(
        current_issues=existing_issues_str,
        new_issue=new_issue_str
    )
    create_scenario_prompt = prompts["create"].format(
        current_issues=existing_issues_str,
        new_issue=new_issue_str
    )

    scenario_results = await llm.chat_batch(
        [ignore_scenario_prompt, update_scenario_prompt, create_scenario_prompt],
        task_name='check_all_scenarios',
        model_name=MODEL_TO_USE,
        batch_size=3,
        temperature=0.1,
        use_tqdm=False
    )

    ignore_response = scenario_results[0]
    update_response = scenario_results[1]
    create_response = scenario_results[2]

    ignore_output = parse_json(ignore_response)
    update_output = parse_json(update_response)
    create_output = parse_json(create_response)

    ignore_output_str = dict_to_key_value_lines(ignore_output)
    update_output_str = dict_to_key_value_lines(update_output)
    create_output_str = dict_to_key_value_lines(create_output)

    verifier_prompt = prompts["verifier"].format(
        ignore_output=ignore_output_str,
        update_output=update_output_str,
        create_new_output=create_output_str,
        new_issue=new_issue_str
    )

    verifier_response = await llm.chat(
        verifier_prompt,
        task_name='verify_scenarios',
        model_name='gpt-4.1-mini',
        add_prompter=True,
        temperature=0.1
    )
    verifier_decision = parse_json(verifier_response)
    decision = verifier_decision.get("decision", "") if verifier_decision else 'IGNORE'

    if len(existing_issues) == 0:
        decision = "CREATE_NEW"

    if decision == "IGNORE":
        old_issue_title = ignore_output.get("existing_issue_title", "")
        if old_issue_title in existing_issues_composition:
            existing_issues_composition[old_issue_title].extend(row_indices)

    elif decision == "UPDATE_EXISTING":
        old_issue_title = update_output.get("old_issue_title", "")
        new_title = update_output.get("new_issue_title", "")
        new_desc = update_output.get("new_issue_description", "")

        if old_issue_title in existing_issues:
            del existing_issues[old_issue_title]

        existing_issues[new_title] = new_desc
        old_indices = existing_issues_composition.pop(old_issue_title, [])
        existing_issues_composition.setdefault(new_title, []).extend(old_indices)
        existing_issues_composition[new_title].extend(row_indices)

    elif decision == "CREATE_NEW":
        new_title = create_output.get("new_issue_title", "")
        new_desc = create_output.get("new_issue_description", "")
        existing_issues[new_title] = new_desc
        existing_issues_composition.setdefault(new_title, []).extend(row_indices)

async def process_issues_stream(
    issues_stream: pd.DataFrame,
    existing_issues: Dict[str, str],
    existing_issues_composition: Dict[str, List[int]],
    prompts: Dict[str, str],
    desc: str = "Processing"
) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
    """
    Given a DataFrame of new issues (with 'title' and 'description'),
    shuffle them, then process them one by one to update 'existing_issues'
    and 'existing_issues_composition'.
    """
    issues_stream = issues_stream.sample(frac=1, random_state=4).reset_index(drop=True)

    for _, issue in tqdm(issues_stream.iterrows(), total=len(issues_stream), desc=desc):
        new_issue_title = issue["title"]
        new_issue_description = issue["description"]
        row_indices = [issue["og_idx"]]
        await process_new_issue(
            new_issue_title,
            new_issue_description,
            row_indices,
            existing_issues,
            existing_issues_composition,
            prompts
        )
    return existing_issues, existing_issues_composition

def build_composition_details(
    existing_issues: Dict[str, str],
    existing_issues_composition: Dict[str, List[int]],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Builds a dictionary carrying final processed issues along with their compositions.
    """
    composition_w_details = {}
    for k, v in existing_issues_composition.items():
        composition_w_details[k] = {
            "name": k,
            "description": existing_issues[k],
            "composition": [dict(df.loc[df.og_idx == x].reset_index(drop=True).iloc[0].items()) for x in v]
        }
    return composition_w_details

async def process_single_intent(
    key: str,
    intent_data: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    prompts: Dict[str, str]
) -> Tuple[str, Dict[str, Any]]:
    """
    Processes a single intent (issues grouped under a certain key) concurrency-safe.
    1) Converts intent_data['composition'] into a DataFrame.
    2) Runs process_issues_stream to get final existing_issues & composition.
    3) Builds a composition details dict to return.
    """
    async with semaphore:
        df_temp = pd.DataFrame(intent_data["composition"]).reset_index(drop=True)

        existing_issues = {}
        existing_issues_composition = {}

        existing_issues, existing_issues_composition = await process_issues_stream(
            df_temp, existing_issues, existing_issues_composition, prompts, desc=f"Processing key={key}"
        )

        existing_issues_composition_w_details = build_composition_details(
            existing_issues, existing_issues_composition, df_temp
        )
        return key, existing_issues_composition_w_details

async def process_all_intents(
    l3_intents: Dict[str, Any],
    concurrency: int,
    prompts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Processes all items in l3_intents in parallel (up to 'concurrency').
    Returns a dictionary: { key -> composition_details }
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for key, intent_data in l3_intents.items():
        task = asyncio.create_task(process_single_intent(key, intent_data, semaphore, prompts))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    result_dict = {k: v for k, v in results}
    return result_dict


def load_l3_intents(brand_name: str) -> dict:
    load_path = f"./checkpoints/E_cluster_intents/{brand_name}/clustered_issues.pkl"
    with open(load_path, "rb") as f:
        l3_intents = pkl.load(f)
    return l3_intents


def save_clustered_issues(brand_name: str, composition_w_details: dict) -> None:
    """
    Saves the final clustered issues composition to a pickle file
    in this pipeline step's checkpoints directory.
    """
    save_dir = f"./checkpoints/F_cluster_issue_kbs/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "intent_wise_clustered_issues.pkl")
    with open(output_path, "wb") as f:
        pkl.dump(composition_w_details, f)


async def process(brand_name: str, concurrency: int = 5) -> dict:
    """
    High-level process function for F_cluster_issue_kbs step.
    1) Loads the L3 intents data for the given brand.
    2) Processes all the intents in parallel with the specified concurrency.
    3) Saves the final results to a pickle file.
    4) Returns a dictionary of the final clustered issues composition.
    """
    l3_intents = load_l3_intents(brand_name)
    # l3_intents = dict(sorted(l3_intents.items(), key=lambda item: len(item[1]['composition']), reverse=True)[:2])
    result_dict = await process_all_intents(l3_intents, concurrency, get_prompts())
    save_clustered_issues(brand_name, result_dict)

    # 4) Return final composition
    return result_dict