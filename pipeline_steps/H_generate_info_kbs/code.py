import os
import sys
import json
import pickle as pkl
import asyncio
import random
import pandas as pd
from copy import deepcopy
from collections import deque, Counter
from tqdm import tqdm

# Import constants
from constants import *
from utils.llm_helper import LLM

CURR_PIPELINE_STEP = "H_generate_info_kbs"
MODEL_TO_USE = "gemini-flash"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompt(prompt_name: str) -> str:
    """
    Loads a prompt text file from the disk for this pipeline step.
    """
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}', 'r') as file:
        return file.read()

def load_brand_index(brand_name: str) -> dict:
    """
    Loads the pruned/updated index for the given brand
    (re-labeled as 'brand_index').
    """
    file_path = f"./checkpoints/C_update_index_metadata/{brand_name}/pruned_updated_index.pkl"
    with open(file_path, "rb") as f:
        brand_index = pkl.load(f)
    return brand_index

def build_graph_structures(brand_index: dict):
    """
    Build the adjacency (child_map) and in-degree structures directly
    from brand_index. Each path's 'precursor' + 'sub_category_names'
    form the dependencies (parents).
    """
    child_map = {path: [] for path in brand_index}
    in_degree = {path: 0 for path in brand_index}

    for path, data in brand_index.items():
        # Combine precursors + sub_category_names
        deps = data.get(PRECURSOR, []) + data.get(SUB_CATEGORY_NAMES, [])
        # For each dep, build the adjacency and update in-degree
        for dep in deps:
            if dep in brand_index:
                child_map[dep].append(path)
                in_degree[path] += 1
            else:
                print(f"{dep} not in brand_index")

    return child_map, in_degree

def topo_sort_in_batches(child_map: dict, in_degree: dict) -> list:
    """
    Perform a topological sort in batches (layers). Each batch is
    a list of path IDs with in-degree 0 in that round.
    """
    queue = deque(path for path, deg in in_degree.items() if deg == 0)
    result = []

    while queue:
        current_batch = []
        level_size = len(queue)
        for _ in range(level_size):
            node_path = queue.popleft()
            current_batch.append(node_path)
            # Decrement the in-degree of children
            for child in child_map[node_path]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        result.append(current_batch)

    return result

def topological_batches(brand_index: dict) -> list:
    """
    Returns a list of "batches," where each batch is a list of paths
    whose dependencies are satisfied in that layer.
    """
    child_map, in_degree = build_graph_structures(brand_index)
    sorted_batches = topo_sort_in_batches(child_map, in_degree)
    return sorted_batches

def load_issues_data(brand_name) -> pd.DataFrame:
    """
    Loads the issues data from a fixed Excel path.
    """
    issues_df = pd.read_excel(f'./checkpoints/G_generate_issue_kbs/{brand_name}/issue_kbs.xlsx')
    issues_df[CUSTOMER_PERSONA_TITLES] = issues_df[CUSTOMER_PERSONA_TITLES].apply(eval)
    issues_df[CUSTOMER_PERSONA_DESCRIPTIONS] = issues_df[CUSTOMER_PERSONA_DESCRIPTIONS].apply(eval)
    issues_df[INFO_KBS] = issues_df[INFO_KBS].apply(
        lambda x: [eval(item) if isinstance(item, str) else item for item in eval(x)]
    )
    return issues_df

def unify_issues_in_brand_index(issues_df: pd.DataFrame, brand_index: dict) -> None:
    """
    Integrates the 'issue' data into brand_index by creating a new path
    for each issue and linking it to the relevant KB paths.
    """
    for _, row in issues_df.iterrows():
        curr_path = f"Issues->{row[L3_INTENT]}->{row[ISSUE_CLUSTER_TITLE]}"
        curr_desc = row[ISSUE_CLUSTER_DESCRIPTION]
        curr_kb = row[KB_ARTICLE]
        curr_summary = row[NEW_BRAND_DATA]
        corresponding_kbs = list(
            set(y for x in row[INFO_KBS] for y in x if y in brand_index)
        )
        if not corresponding_kbs:
            continue

        # Create a new entry in brand_index without storing "path" as a key
        brand_index[curr_path] = {
            DESCRIPTION: curr_desc,
            PRECURSOR: [],
            SUB_CATEGORIES: [],
            KB_ARTICLE: curr_kb,
            SUMMARY: curr_summary
        }

        # Link the relevant KBs to the new issues path
        for kb in corresponding_kbs:
            # brand_index[kb][PRECURSOR] = list(
            #     set(brand_index[kb][PRECURSOR] + [curr_path])
            # )
            brand_index[kb][PRECURSOR].append(curr_path)
            if len(brand_index[kb][PRECURSOR]) != len(set(brand_index[kb][PRECURSOR])):
                print(f"Len mismatch: {brand_index[kb][PRECURSOR]}")

def get_pruned_path_str(brand_index: dict, path: str) -> str:
    """
    Reconstruct a textual chain of the path segments with descriptions.
    """
    path_from_root = path.split('->')
    lines = []
    for i in range(len(path_from_root)):
        sub_path = '->'.join(path_from_root[:i + 1])
        desc = brand_index.get(sub_path, {}).get(DESCRIPTION, 'No desc found')
        lines.append(f"{i + 1}. {sub_path}: {desc}")
    return '\n'.join(lines)

def get_ancestor_properties(brand_index: dict, path: str) -> dict:
    """
    Retrieve the parent's 'properties' by removing the last segment
    of the path.
    """
    if '->' not in path:
        return {}
    parent_path = '->'.join(path.split('->')[:-1])
    return brand_index.get(parent_path, {}).get(PROPERTIES, {})

def numbered_expanded_dict(x: dict) -> str:
    """
    Utility to produce a string for the dict in enumerated form.
    """
    return '\n'.join([f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(x.items())])

def get_ancestor_properties_str(brand_index: dict, path: str) -> str:
    """
    Produces a string of the parent's properties.
    """
    props = {
        k: v
        for k, v in get_ancestor_properties(brand_index, path).items()
        if v != 'NA'
    }
    return numbered_expanded_dict(props) if props else 'None Available'

def get_curr_properties_str(brand_index: dict, path: str) -> str:
    """
    Produces a string of the current node's properties minus what
    the parent already has.
    """
    ancestor_props = get_ancestor_properties(brand_index, path)
    curr_props = brand_index.get(path, {}).get(PROPERTIES, {})
    filtered_props = {
        k: v
        for k, v in curr_props.items()
        if k not in ancestor_props and v != 'NA'
    }
    return numbered_expanded_dict(filtered_props) if filtered_props else 'None Available'

def get_sub_categories_summary(brand_index: dict, path: str) -> str:
    """
    Creates a summary from immediate subcategories' summary or description.
    """
    subcats = brand_index[path].get(SUB_CATEGORIES, [])
    if not subcats:
        return 'None Available'
    summary_map = {
        x: brand_index[x].get(SUMMARY, brand_index[x].get(DESCRIPTION))
        for x in subcats
    }
    return numbered_expanded_dict(summary_map)

def get_precursor_summary(brand_index: dict, path: str) -> str:
    """
    Creates a summary from the precursor paths' summary or description fields.
    """
    precursors = brand_index[path].get(PRECURSOR, [])
    if not precursors:
        return 'None Available'
    summary_map = {
        x: brand_index[x].get(SUMMARY, brand_index[x].get(DESCRIPTION))
        for x in precursors
    }
    return numbered_expanded_dict(summary_map)

def get_properties(brand_index: dict, path: str) -> dict:
    """
    Retrieves only the existing properties on the current node
    that are not marked as 'NA'.
    """
    node_data = brand_index.get(path, {})
    props = node_data.get(PROPERTIES, {})
    return {k: v for k, v in props.items() if v != 'NA'}

def save_info_articles(brand_name: str, info_df: pd.DataFrame, brand_index) -> None:
    """
    Saves the final Info Articles DataFrame to Excel
    under the current pipeline step.
    """
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)

    info_df['uID'] = [f"INFO_{i + 1}" for i in range(1, len(info_df) + 1)]
    info_df.to_excel(f"{save_dir}/info_articles.xlsx", index=False)
    pkl.dump(brand_index, open(f"{save_dir}/all_kb_brand_index.pkl", "wb"))

async def populate_kbs(
    brand_index: dict,
    brand_overview: str,
    sorted_batches: list,
    kb_lang: str
) -> None:
    """
    Iterates through the topologically sorted batches generating KB and Summaries.
    """
    kb_generation_template = load_prompt('kb_generation_template')
    summary_template = load_prompt('summary_template')

    for level_order in tqdm(sorted_batches, desc="Populating KBs in topological batches"):
        level_prompts = []
        for path in level_order:
            prompt_text = kb_generation_template.format(
                brand_overview=brand_overview,
                entity_name=path.split("->")[-1],
                entity_desc=brand_index[path].get(DESCRIPTION, ""),
                path=get_pruned_path_str(brand_index, path),
                ancestor_properties=get_ancestor_properties_str(brand_index, path),
                properties=get_curr_properties_str(brand_index, path),
                sub_categories_summary=get_sub_categories_summary(brand_index, path),
                precursor_summary=get_precursor_summary(brand_index, path),
                lang=kb_lang
            )
            level_prompts.append(prompt_text)

        # Primary attempt
        kbs = await llm.chat_batch(
            level_prompts,
            task_name='kb_generation',
            model_name=MODEL_TO_USE,
            batch_size=10,
            temperature=0.6,
            max_tokens=8000
        )


        # Retry failed calls on a smaller model
        failed_prompts_and_indices = [
            (level_prompts[i], i) for i in range(len(kbs)) if kbs[i] is None
        ]
        if failed_prompts_and_indices:
            failed_prompts, failed_indices = zip(*failed_prompts_and_indices)
            print(f"Retrying {len(failed_prompts)} failed prompts...")
            new_kbs = await llm.chat_batch(
                failed_prompts,
                task_name='kb_generation_retry',
                model_name='gpt-4.1-mini',
                batch_size=len(failed_prompts),
                temperature=0.6,
                max_tokens=8000
            )
            for ni, new_kb_val in zip(failed_indices, new_kbs):
                if new_kb_val is not None:
                    kbs[ni] = new_kb_val

        # Summaries
        summary_prompts = [summary_template.format(kb=x if x else '') for x in kbs]
        summaries = await llm.chat_batch(
            summary_prompts,
            task_name='kb_summary_generation',
            model_name=MODEL_TO_USE,
            batch_size=10,
            temperature=0.6
        )

        for path_, kb_content, summary_content in zip(level_order, kbs, summaries):
            brand_index[path_][KB_ARTICLE] = kb_content
            brand_index[path_][SUMMARY] = summary_content

def create_final_info_df(brand_index: dict, sorted_batches: list) -> pd.DataFrame:
    """
    Creates the final DataFrame of paths, KB content, Summaries,
    metadata, and direct dependencies from brand_index.
    """
    paths_list = []
    kbs_list = []
    summaries_list = []
    metadata_list = []
    depends_on_list = []

    # Reverse the order so the last batch is on top if desired
    for level_order in reversed(sorted_batches):
        for path in level_order:
            paths_list.append(path)
            kbs_list.append(brand_index[path].get(KB_ARTICLE, None))
            summaries_list.append(brand_index[path].get(SUMMARY, None))
            props = get_properties(brand_index, path)
            metadata_list.append(json.dumps(props, indent=2))
            try:
                deps = brand_index[path].get(PRECURSOR, []) + brand_index[path].get(SUB_CATEGORY_NAMES, [])
            except Exception as e:
                print(f"Exception {e} occurred")
                print(f"Path: {path}")
                print(f"Precursor: {brand_index[path].get(PRECURSOR, [])}")
                print(f"Subcategories: {brand_index[path].get(SUB_CATEGORY_NAMES, [])}")
                raise Exception(e)
            depends_on_list.append(deps)

    info_df = pd.DataFrame({
        'Path': paths_list,
        'KB': kbs_list,
        'Summary': summaries_list,
        'Metadata': metadata_list,
        'depends_on': depends_on_list
    })
    return info_df

async def process(brand_name: str, brand_overview: str, kb_lang='English') -> pd.DataFrame:
    """
    High-level orchestration:
    1) Load brand_index
    2) Load issues data and unify
    3) Topological sort
    4) Populate KBs
    5) Create the final info articles dataframe and save
    6) Return the final dataframe
    """
    brand_index = load_brand_index(brand_name)
    sorted_batches = topological_batches(brand_index)

    issues_df = load_issues_data(brand_name)
    unify_issues_in_brand_index(issues_df, brand_index)

    await populate_kbs(brand_index, brand_overview, sorted_batches, kb_lang)
    info_df = create_final_info_df(brand_index, sorted_batches)
    save_info_articles(brand_name, info_df, brand_index)
    return info_df
