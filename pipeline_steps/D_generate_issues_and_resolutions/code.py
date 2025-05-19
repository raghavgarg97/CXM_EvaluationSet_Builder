import os
import json
import pickle as pkl
import random
from fuzzywuzzy import process as fuzzywuzzy_process
from typing import Dict, Any, List
from tqdm.auto import tqdm

from constants import (
    SUB_CATEGORY_NAMES,
    DESCRIPTION,
    PROPERTIES,
)
from utils.llm_helper import LLM
from utils.misc import parse_json
import asyncio

CURR_PIPELINE_STEP = "D_generate_issues_and_resolutions"
MODEL_TO_USE = "gemini-flash"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_previous_step_data(brand_name: str) -> Dict[str, Dict[str, Any]]:
    """Loads the updated index from the previous step"""
    pruned_updated_index_path = f"./checkpoints/C_update_index_metadata/{brand_name}/pruned_updated_index.pkl"
    with open(pruned_updated_index_path, "rb") as f:
        return pkl.load(f)

def save_issues_and_resolutions(brand_name: str, issues: List[Dict[str, Any]]) -> None:
    """Saves the generated issues as a pickle"""
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)
    issues_path = os.path.join(save_dir, "issues_and_resolutions.pkl")
    with open(issues_path, "wb") as f:
        pkl.dump(issues, f)

def get_ancestor_properties(updated_index: Dict[str, Dict[str, Any]], path: str) -> Dict[str, Any]:
    """Returns the parent node's PROPERTIES if any"""
    if '->' not in path:
        return {}
    parent_path = '->'.join(path.split('->')[:-1])
    return updated_index.get(parent_path, {}).get(PROPERTIES, {})

def numbered_expanded_dict(x: Dict[str, Any]) -> str:
    """Converts a dict to a numbered list string"""
    return '\n'.join([f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(x.items())])

def get_ancestor_properties_str(updated_index: Dict[str, Dict[str, Any]], path: str) -> str:
    """Shows ancestor (parent) properties if they exist"""
    props = {k: v for k, v in get_ancestor_properties(updated_index, path).items() if v != 'NA'}
    if not props:
        return 'None Available'
    return numbered_expanded_dict(props)

def get_curr_properties_str(updated_index: Dict[str, Dict[str, Any]], path: str) -> str:
    """Shows current node's unique properties (not in the ancestor)"""
    ancestor_props = get_ancestor_properties(updated_index, path)
    full_props = updated_index.get(path, {}).get(PROPERTIES, {})
    unique_props = {k: v for k, v in full_props.items() if k not in ancestor_props and v != 'NA'}
    if not unique_props:
        return 'None Available'
    return numbered_expanded_dict(unique_props)

def get_kb_info_single(updated_index: Dict[str, Dict[str, Any]], path: str) -> str:
    """Builds a knowledge base info snippet for a single path"""
    description = updated_index.get(path, {}).get(DESCRIPTION, "")
    ancestor_props = get_ancestor_properties_str(updated_index, path)
    curr_props = get_curr_properties_str(updated_index, path)
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/single_kb_info_template') as f:
        kb_info_template = f.read()
    return kb_info_template.format(
        path=path,
        description=description,
        ancestor_props=ancestor_props,
        curr_props=curr_props
    )

def get_kb_info_list(updated_index: Dict[str, Dict[str, Any]], paths: List[str]) -> str:
    """Joins multiple KB snippets"""
    return '\n--------------------\n'.join(
        [get_kb_info_single(updated_index, x) for x in paths]
    )

async def generate_issues(updated_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates issues from the updated_index"""
    # Identify leaf paths
    all_paths = list(updated_index.keys())
    all_paths_sorted = sorted(all_paths, key=lambda x: x.count('->'), reverse=True)

    leafs = set()
    non_leafs = set()
    for p in all_paths_sorted:
        if p not in non_leafs:
            leafs.add(p)
            parent_str = '->'.join(p.split('->')[:-1])
            if parent_str:
                non_leafs.add(parent_str)

    # Partition leafs
    core_leafs = [x for x in leafs if x.startswith('Product') or x.startswith('Services')]
    non_core_leafs = [x for x in leafs if not (x.startswith('Product') or x.startswith('Services'))]
    print(f"Number of core_leafs: {len(core_leafs)}; Number of non_core_leafs: {len(non_core_leafs)}")

    # Prepare seeds for issue generation
    issue_seeds = [[x] for x in core_leafs]
    random.Random(1).shuffle(issue_seeds)

    n_mixed_issues = int(len(core_leafs) * random.Random(1).randint(25, 35)/100)
    
    for i, x in enumerate(issue_seeds):
        if i < n_mixed_issues:
            issue_seeds[i].append(non_core_leafs[i%len(non_core_leafs)])

    random.Random(1).shuffle(issue_seeds)

    # Create prompts for LLM
    prompts = []
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/issue_generation') as f:
        issue_gen_template = f.read()
        
    for curr_seed in issue_seeds:
        kb_info_block = get_kb_info_list(updated_index, curr_seed)
        prompt_text = issue_gen_template.format(kb_info=kb_info_block)
        # print(f"Prompt is: {prompt_text}")
        prompts.append({"prompt": prompt_text, "seed": curr_seed})
        
    print(f"Number of issue gen prompts: {len(prompts)}")
    print(f"[{CURR_PIPELINE_STEP}] Calling LLM for issue generation in batches...")
    responses = await llm.chat_batch(
        [p["prompt"] for p in prompts],
        temperature=0.1,
        max_tokens=8000,
        batch_size=20,
        model_name=MODEL_TO_USE,
        task_name="issue_resolution_generation",
    )

    all_issues = []
    for seeds, response in tqdm(zip(issue_seeds, responses), desc='Parsing generated issues', total = len(responses)):
        issues = parse_json(response, default_json={'issues': []}).get('issues', [])
        
        for issue in issues:
            if 'kbs' in issue:
                corrected_kbs = []
                for kb in issue['kbs']:
                    best_match, score = fuzzywuzzy_process.extractOne(kb, seeds)
                    corrected_kbs.append(best_match)
                issue['kbs'] = corrected_kbs
            else:
                issue['kbs'] = seeds
        all_issues.extend(issues)

    print(f"{CURR_PIPELINE_STEP} - Total issues generated: {len(all_issues)}")
    return all_issues

async def process(brand_name: str) -> List[Dict[str, Any]]:
    """Main process function for generating issues/resolutions"""
    print(f"[{CURR_PIPELINE_STEP}] Loading updated index for brand: {brand_name}")
    updated_index = load_previous_step_data(brand_name)

    print(f"[{CURR_PIPELINE_STEP}] Generating issues for brand: {brand_name}")
    all_issues = await generate_issues(updated_index)

    print(f"[{CURR_PIPELINE_STEP}] Saving issues for brand: {brand_name}")
    save_issues_and_resolutions(brand_name, all_issues)

    print(f"[{CURR_PIPELINE_STEP}] Finished generating issues for brand: {brand_name}")
    return all_issues
