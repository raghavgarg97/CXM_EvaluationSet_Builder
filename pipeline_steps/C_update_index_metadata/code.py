import os
import json
import pickle as pkl
from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, Any

# Third-party libraries
# (Add any third-party imports if necessary. For example, pandas if you still need it.)

# Local imports (assuming similar structure to reference code)
from constants import (
    SUB_CATEGORY_NAMES,
    DESCRIPTION,
    PROPERTIES,
    PRECURSOR,
    PATH,
)
from utils.misc import parse_json
from utils.llm_helper import LLM

CURR_PIPELINE_STEP = "C_update_index_metadata"
MODEL_TO_USE = "gpt-mini"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_previous_step_data(brand_name):
    pruned_index_path = f"./checkpoints/B_index_pruning/{brand_name}/pruned_index.pkl"
    with open(pruned_index_path, "rb") as f:
        pruned_index = pkl.load(f)
    return pruned_index


def save_updated_data(brand_name, updated_index):
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)

    updated_index_path = os.path.join(save_dir, "pruned_updated_index.pkl")
    with open(updated_index_path, "wb") as f:
        pkl.dump(updated_index, f)


def parse_property_modification_result(text):
    default_json = {
        "reasoning": "Error occurred parsing LLM output",
        "new_values": {}
    }
    return parse_json(text, default_json=default_json)


async def update_all_subcategories(
    pruned_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Update subcategories iteratively by depth level, using LLM to fill missing property values.
    """
    updated_index = deepcopy(pruned_index)
    property_modification_template = open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/property_modification').read()
    
    # Group paths by their depth level
    depth_map = defaultdict(list)
    for path in updated_index.keys():
        depth = path.count('->') + 1
        depth_map[depth].append(path)
    
    max_depth = max(depth_map.keys())

    # Process each depth level in order
    for depth in range(1, max_depth + 1):
        current_level_paths = depth_map[depth]
        
        prompts_for_depth = []
        parent_paths = []

        # Log the number of candidates at the current depth
        print(f"[Depth {depth}] Number of candidates for update: {len(current_level_paths)}")

        # Prepare prompts for each parent at this depth level
        for path in current_level_paths:
            data = updated_index[path]
            
            # Skip if missing required data
            if SUB_CATEGORY_NAMES not in data or "list_of_properties" not in data:
                continue

            # print(f"SUB category names of {path} are: {data[SUB_CATEGORY_NAMES]}")
            # sub_cat_paths = ['->'.join(data[PATH] + [sub_cat_name]) for sub_cat_name in data[SUB_CATEGORY_NAMES]]
            sub_cat_paths = data[SUB_CATEGORY_NAMES]
            if not sub_cat_paths:
                continue

            # Build data structure for children
            sub_data = _build_child_data(updated_index, data, sub_cat_paths)
            if not sub_data:
                continue
                
            # Prepare prompt
            property_defs = data["list_of_properties"]
            properties_definition = "\n".join(
                f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(property_defs.items())
            )
            
            curr_prompt = property_modification_template.format(
                entity_name=path,
                entity_desc=data.get(DESCRIPTION, ""),
                sub_cat_data=json.dumps(sub_data, indent=2),
                properties=properties_definition
            )

            prompts_for_depth.append(curr_prompt)
            parent_paths.append(path)

        # Process batch of prompts
        if prompts_for_depth:
            # Print one prompt at each depth for debugging
            # print(f"\n[Depth {depth}] Sample prompt:")
            # print(prompts_for_depth[0])

            await _process_batch_prompts(prompts_for_depth, parent_paths, updated_index, depth)
            
    return updated_index


def _build_child_data(updated_index, parent_data, sub_cat_paths):
    """Build data structure for child categories."""
    sub_data = {}
    for child_path in sub_cat_paths:
        if child_path not in updated_index:
            print(f"Warning: Child path {child_path} not found in index. Skipping.")
            continue

        child_item = updated_index[child_path]
        child_description = child_item.get(DESCRIPTION, "")
        child_props = child_item.get(PROPERTIES, {})

        # Separate properties into active vs passive
        active_properties = {}
        passive_properties = {}
        for prop_key, prop_val in child_props.items():
            if prop_key in parent_data["list_of_properties"]:
                active_properties[prop_key] = prop_val
            else:
                passive_properties[prop_key] = prop_val

        sub_data[child_path] = {
            "description": child_description,
            "active_properties": active_properties,
            "passive_properties": passive_properties
        }
    
    return sub_data


async def _process_batch_prompts(prompts, parent_paths, updated_index, depth):
    """Process a batch of prompts and update the index with results."""
    try:
        responses = await llm.chat_batch(
            prompts,
            temperature=0.1,
            batch_size=10,
            task_name="properties_refilling",
            model_name=MODEL_TO_USE,
            add_prompter=True,
            use_tqdm=False
        )
    except Exception as e:
        print(f"Error: Exception during LLM batch call at depth {depth}: {e}")
        return

    # Print one sample response for debugging
    # if responses:
        # print(f"[Depth {depth}] Sample response:")
        # print(responses[0])

    # Process responses and update index
    for parent_path, raw_response in zip(parent_paths, responses):
        try:
            # Use the existing parsing function
            response_data = parse_property_modification_result(raw_response)
            
            new_values = response_data.get("new_values", {})
            for child_path, child_updates in new_values.items():
                if not child_updates or child_path not in updated_index:
                    continue
                    
                print(f"Info: Updating properties for {child_path} (parent={parent_path}) at depth {depth}")
                updated_index[child_path].setdefault(PROPERTIES, {}).update(child_updates)
                
        except Exception as err:
            print(f"Error: Processing failed for path {parent_path}: {err}")


async def process(brand_name):
    print(f"[{CURR_PIPELINE_STEP}] Loading pruned index for brand: {brand_name}")
    pruned_index = load_previous_step_data(brand_name)

    print(f"[{CURR_PIPELINE_STEP}] Starting property enrichment for {brand_name} ...")
    updated_index = await update_all_subcategories(pruned_index)

    print(f"[{CURR_PIPELINE_STEP}] Saving updated index for {brand_name} ...")
    save_updated_data(brand_name, updated_index)

    print(f"[{CURR_PIPELINE_STEP}] Finished property enrichment for {brand_name}.")
    return updated_index
