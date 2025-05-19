import asyncio
import time
import json
import os
from constants import *
from copy import deepcopy
from utils.misc import parse_json
from collections import defaultdict, deque

from utils.llm_helper import LLM
import pickle as pkl

MODEL_TO_USE = None
CURR_RUN_START_TIME = None
CURR_PIPELINE_STEP = 'A_index_generation'
llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def get_prompt_by_name(prompt_name):
    return open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}.txt').read()


def topological_sort_by_precursor(data_dict):
    """
    Returns a layered topological order of the keys in 'data_dict'
    using 'precursor' dependencies.
    """

    graph = defaultdict(list)
    in_degree = {k: 0 for k in data_dict}

    # Build graph and compute in-degrees
    for key, info in data_dict.items():
        precursors = info.get(PRECURSOR) or []
        for p in precursors:
            if p in data_dict:
                graph[p].append(key)
                in_degree[key] += 1

    queue = deque([k for k, deg in in_degree.items() if deg == 0])
    layered_order = []
    visited_count = 0

    while queue:
        level_size = len(queue)
        current_layer = []

        for _ in range(level_size):
            node = queue.popleft()
            current_layer.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        visited_count += len(current_layer)
        layered_order.append(current_layer)

    if visited_count != len(data_dict):
        print("Could not produce a valid layered topological ordering. "
              "A cycle or invalid dependency might exist.")
        # fallback
        return [[x] for x in data_dict.keys()]

    return layered_order

def get_indexed_taxonomy(entity, keys, max_level, curr_level=1, prefix=''):
    """
    Returns a list of indexed textual descriptions for entity deeper down the hierarchy
    up to 'max_level'.
    """
    if curr_level == max_level + 1 or not keys:
        return []

    result = []
    for i, key in enumerate(keys):
        spacing = '\t' * (curr_level - 1)
        curr_line = f"{spacing}{prefix}{i+1}. {key}"
        try:
            if (curr_level == max_level
                or len(entity[key].get(SUB_CATEGORIES, {})) == 0):
                curr_line += f": {entity[key][DESCRIPTION]}"
            result.append(curr_line)
            sub_cat = entity[key].get(SUB_CATEGORIES, {})
            result.extend(
                get_indexed_taxonomy(
                    sub_cat,
                    sub_cat.keys(),
                    max_level,
                    curr_level + 1,
                    prefix+f'{i+1}.'
                )
            )
        except Exception as e:
            print(f"Exception in get_indexed_taxonomy: {e}")
    return result

def get_expanded_path_str(all_definitions_dict, entity, key):
    path_str = '\n'.join(
        f"{i + 1}. {'->'.join(entity[key][PATH][:i + 1])}: "
        f"{all_definitions_dict['->'.join(entity[key][PATH][:i + 1])]}"
        for i in range(len(entity[key][PATH]))
    )
    return path_str

async def populate_precursors(entity, keys):
    """
    Calls LLM to generate or update precursor info for the subcategories
    of the given 'keys' in 'entity'.
    """
    try:
        entity_desc_batch = [entity[key][DESCRIPTION] for key in keys]
        sub_entity_batch = [entity[key][SUB_CATEGORIES] for key in keys]
        prompt_batch = []

        for entity_desc, sub_entity in zip(entity_desc_batch, sub_entity_batch):
            prompt = get_prompt_by_name('precursor_population').format(
                key_info=entity_desc,
                sub_entities=json.dumps(sub_entity, indent=2)
            )
            prompt_batch.append(prompt)

        res_batch = await llm.chat_batch(
            prompt_batch,
            task_name='precursor_population',
            batch_size=10,
            temperature=0.1,
            model_name=MODEL_TO_USE
        )

        for key, sub_cat in zip(keys, [parse_json(r) for r in res_batch]):
            entity[key][SUB_CATEGORIES] = sub_cat
        return entity

    except Exception as e:
        print("Error in populate_precursors:", e)
        for key in keys:
            sub_cat = deepcopy(entity[key][SUB_CATEGORIES])
            for k in sub_cat.keys():
                sub_cat[k][PRECURSOR] = []
            entity[key][SUB_CATEGORIES] = sub_cat
        return entity

async def update_description(all_definitions_dict, entity, keys):
    """
    Calls LLM to update the descriptions of the entities specified by 'keys'
    based on their sub-entities.
    """
    sub_entity_batch = []
    for key in keys:
        sub_entity_copy = deepcopy(entity[key][SUB_CATEGORIES])
        for k in sub_entity_copy.keys():
            sub_entity_copy[k] = {
                DESCRIPTION: sub_entity_copy[k][DESCRIPTION],
                PROPERTIES: sub_entity_copy[k].get(PROPERTIES, {})
            }
        sub_entity_batch.append(sub_entity_copy)

    prompt_batch = [
        get_prompt_by_name('update_desc').format(
            sub_entities=json.dumps(se, indent=2)
        )
        for se in sub_entity_batch
    ]
    res_batch = await llm.chat_batch(
        prompt_batch,
        task_name='description_update',
        batch_size=10,
        temperature=0.1,
        model_name=MODEL_TO_USE
    )

    for key, new_desc in zip(keys, res_batch):
        entity[key][OLD_DESCRIPTION] = entity[key][DESCRIPTION]
        entity[key][DESCRIPTION] = new_desc.strip("```").strip()
        all_definitions_dict['->'.join(entity[key][PATH])] = (
            new_desc.strip("```").strip()
        )
    return entity

def get_appropriate_level_prompt(brand_description, brand_properties_keys, all_definitions_dict, entity, key, curr_level, precursor_info):
    """
    Determines which prompt template to use based on the current level (L1, L2, or Ln).
    """
    max_level_val = entity[key][MAX_LEVEL]

    # If no further level expansion is needed or category is purely descriptive:
    if (max_level_val + 1 == curr_level
        or entity[key].get(CATEGORY, 'NA') == CATEGORY_DESCRIPTION
        or entity[key].get(N_SUBCATEGORIES, 1) == 0):
        return None

    properties_info = entity[key].get(PROPERTIES, {})
    properties_str = '\n'.join(
        [f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(properties_info.items())]
    )
    properties_str = (f'\nSome more attributes/properties of this entity are:\n{properties_str}\n'
                      if properties_str else '')

    precursor_str = (f"\nYou are also given some data about other sibling entities "
                     f"which might help in generating subcategories. Use if relevant:\n"
                     f"{precursor_info}\n") if precursor_info else ''

    if curr_level == 1:
        sibling_entities = [
            f"- {x}: {entity[x][DESCRIPTION]}" for x in entity.keys() if x != key
        ]
        prompt_text = get_prompt_by_name('l1_prompt').format(
            brand_description=brand_description,
            entity_name=key,
            entity_desc=entity[key][DESCRIPTION],
            sibling_entities='\n'.join(sibling_entities),
            precursor_info=precursor_str,
            properties_info=properties_str
        )
        return prompt_text

    if curr_level == 2 and entity[key][PATH][0] in ['Product Range', 'Services Range']:
        parent_name = entity[key][PATH][-2]
        parent_desc = all_definitions_dict['->'.join(entity[key][PATH][:-1])]
        sibling_entities = [
            f"- {x}: {entity[x][DESCRIPTION]}" for x in entity.keys() if x != key
        ]
        # Also consider brand-level properties
        sibling_entities += [
            f"- {x}: {all_definitions_dict[x]}"
            for x in brand_properties_keys
            if x not in entity[key][PATH]
        ]
        prompt_text = get_prompt_by_name('l2_prompt').format(
            brand_description=brand_description,
            entity_name=key,
            entity_desc=entity[key][DESCRIPTION],
            sibling_entities='\n'.join(sibling_entities),
            parent_name=parent_name,
            parent_desc=parent_desc,
            precursor_info=precursor_str,
            properties_info=properties_str,
            n_subcategories=entity[key][N_SUBCATEGORIES]
        )
        return prompt_text

    # Ln case
    path_str = get_expanded_path_str(all_definitions_dict, entity, key)
    prompt_text = get_prompt_by_name('ln_prompt').format(
        brand_description=brand_description,
        entity_name=key,
        entity_desc=entity[key][DESCRIPTION],
        path=path_str,
        precursor_info=precursor_str,
        properties_info=properties_str,
        n_subcategories=entity[key][N_SUBCATEGORIES]
    )
    return prompt_text

async def update_properties(brand_description, entity, key_batch):
    """
    Calls LLM to update properties for each key in 'key_batch'
    and merges them with the existing subcategories.
    """
    try:
        sub_entity_batch = [entity[k][SUB_CATEGORIES] for k in key_batch]
        prompt_batch = []
        for k, sub_entity in zip(key_batch, sub_entity_batch):
            curr_props = entity[k].get(PROPERTIES, {})
            props_str =  (
                [f"{i+1}. {pk}: {pv}" for i, (pk, pv) in enumerate(curr_props.items())]
                if curr_props else 'None'
            )
            sub_entities_str = (
                [f"{i+1}. {mk}: {mv[DESCRIPTION]}" for i, (mk, mv) in enumerate(sub_entity.items())]
                if sub_entity else 'None'
            )
            prompt = get_prompt_by_name('update_properties').format(
                brand_description=brand_description,
                entity_name=k,
                entity_desc=entity[k][DESCRIPTION],
                sub_entities=sub_entities_str,
                curr_properties=props_str
            )
            prompt_batch.append(prompt)

        results = await llm.chat_batch(
            prompt_batch,
            task_name='update_properties',
            batch_size=10,
            temperature=0.1,
            add_prompter=True,
            model_name=MODEL_TO_USE
        )
        results = [parse_json(x) for x in results]

        for k, result in zip(key_batch, results):
            entity[k][LIST_OF_PROPERTIES] = result[LIST_OF_PROPERTIES]
            # Update sub_entity properties
            sub_entity_map = entity[k][SUB_CATEGORIES]
            for sub_c_key in sub_entity_map.keys():
                try:
                    sub_entity_map[sub_c_key][PROPERTIES].update(result[sub_c_key])
                except Exception:
                    # Fallback: update with the global list of properties
                    sub_entity_map[sub_c_key][PROPERTIES].update(result[LIST_OF_PROPERTIES])
            entity[k][SUB_CATEGORIES] = sub_entity_map

        return entity

    except Exception as e:
        print(f"Exception in update_properties: {e}")
        return entity

async def update_subcategories_count(brand_description, all_definitions_dict, entity, key_batch):
    """
    Allows the LLM to decide how many subcategories each entity's sub-entity will have.
    """
    try:
        sub_entity_batch = [entity[k][SUB_CATEGORIES] for k in key_batch]
        prompt_batch = []
        for key, sub_entity in zip(key_batch, sub_entity_batch):
            path_str = get_expanded_path_str(all_definitions_dict, entity, key)
            sub_entities_str = (
                [f"{i+1}. {mk}: {mv[DESCRIPTION]}" for i, (mk, mv) in enumerate(sub_entity.items())]
                if sub_entity else 'None'
            )
            prompt = get_prompt_by_name('subcategory_count').format(
                brand_description=brand_description,
                entity_name=key,
                entity_desc=entity[key][DESCRIPTION],
                sub_entities=sub_entities_str,
                path=path_str
            )
            prompt_batch.append(prompt)

        results = await llm.chat_batch(
            prompt_batch,
            task_name='update_subcategories_count',
            batch_size=10,
            temperature=0.1,
            add_prompter=True,
            model_name=MODEL_TO_USE
        )
        results = [parse_json(x) for x in results]

        for k, result in zip(key_batch, results):
            sub_entity_map = entity[k][SUB_CATEGORIES]
            for sub_c_key in sub_entity_map.keys():
                sub_entity_map[sub_c_key][N_SUBCATEGORIES] = (
                    result['allotment'][sub_c_key]
                )
            entity[k][SUB_CATEGORIES] = sub_entity_map

        return entity

    except Exception as e:
        print(f"Exception in update_subcategories_count: {e}")
        # fallback: set all to 5
        for k in key_batch:
            sub_entity_map = entity[k][SUB_CATEGORIES]
            for sub_c_key in sub_entity_map.keys():
                sub_entity_map[sub_c_key][N_SUBCATEGORIES] = 5
            entity[k][SUB_CATEGORIES] = sub_entity_map
        return entity

async def sub_entity_generation(brand_description, brand_properties_keys, all_definitions_dict, entity, key_batch, curr_level):
    """
    For each entity in 'key_batch', it calls the LLM to propose
    new sub-entity expansions if needed.
    """
    precursors = [entity[k][PRECURSOR] for k in key_batch]
    precursor_data = [
        '\n'.join(get_indexed_taxonomy(entity, x, 2)) for x in precursors
    ]
    prompts = [
        get_appropriate_level_prompt(brand_description, brand_properties_keys, all_definitions_dict, entity, key,
                                     curr_level, precursor_info)
        for key, precursor_info in zip(key_batch, precursor_data)
    ]

    # Filter out None prompts
    chosen_indices = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in chosen_indices]

    results = await llm.chat_batch(
        valid_prompts,
        task_name='sub_entity_generation',
        batch_size=10,
        temperature=0.1,
        add_prompter=True,
        model_name=MODEL_TO_USE
    )
    parsed_results = [parse_json(x, default_json={'answer': CATEGORY_DESCRIPTION, SUB_CATEGORIES: {}}) for x in results]
    final_results = [
        {
            CATEGORY: r.get('answer', CATEGORY_DESCRIPTION),
            SUB_CATEGORIES: r.get(SUB_CATEGORIES, {})
        }
        for r in parsed_results
    ]

    # Default fallback for each key in key_batch
    ret = [{
        CATEGORY: CATEGORY_DESCRIPTION,
        SUB_CATEGORIES: {}
    } for _ in key_batch]

    for idx, chosen_idx in enumerate(chosen_indices):
        ret[chosen_idx] = final_results[idx]

    return ret

def log_time_and_path(brand_name, sub_categories, key):
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/logs/{brand_name}.txt', 'a') as f:
        elapsed = time.time() - CURR_RUN_START_TIME
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        time_str = ""
        if hours > 0:
            time_str += f"{hours}h "
        if minutes > 0:
            time_str += f"{minutes}m "
        if seconds > 0 or not time_str:
            time_str += f"{seconds:.2f}s"
        f.write(f"Time elapsed: {time_str} for path: "
                + '->'.join(sub_categories[key][PATH]) + '\n')

async def expand_brand_index(brand_name, brand_description, brand_properties_keys, all_definitions_dict, entity, curr_level=1):
    """
    Recursively expands the brand index for an entity dictionary.
    Each pass will:
      - generate sub-entities if needed,
      - populate precursors,
      - update properties,
      - decide subcategory counts,
      - drill down recursively,
      - and finally update descriptions.
    """
    sorted_key_batches = topological_sort_by_precursor(entity)

    for key_batch in sorted_key_batches:
        try:
        # if True:
            sub_entities_data = await sub_entity_generation(brand_description, brand_properties_keys, all_definitions_dict, entity, key_batch, curr_level)
            for key, res in zip(key_batch, sub_entities_data):
                entity[key][CATEGORY] = res[CATEGORY]
                sub_cats = res[SUB_CATEGORIES]
                for k in sub_cats.keys():
                    sub_cats[k] = {
                        DESCRIPTION: sub_cats[k],
                        PATH: entity[key][PATH] + [k],
                        MAX_LEVEL: entity[key][MAX_LEVEL],
                        PROPERTIES: entity[key].get(PROPERTIES, {})
                    }
                    # Update our global dictionary
                    all_definitions_dict['->'.join(sub_cats[k][PATH])] = sub_cats[k][DESCRIPTION]
                    # Log time and path
                    log_time_and_path(brand_name, sub_cats, k)

                entity[key][SUB_CATEGORIES] = sub_cats
                # print(f"Setting subcategories of {key} to {json.dumps(sub_cats, indent=2)}")
                # print('='*40)

            valid_subcategories_index = [
                i for i, ky in enumerate(key_batch)
                if entity[ky].get(SUB_CATEGORIES, {})
            ]
            valid_keys = [key_batch[i] for i in valid_subcategories_index]

            entity = await populate_precursors(entity, valid_keys)
            entity = await update_properties(brand_description, entity, valid_keys)
            entity = await update_subcategories_count(brand_description, all_definitions_dict, entity, valid_keys)

            # Recursively expand subcategories
            for idx in valid_subcategories_index:
                sub_cat_entity = entity[key_batch[idx]][SUB_CATEGORIES]
                entity[key_batch[idx]][SUB_CATEGORIES] = await expand_brand_index(brand_name, brand_description, brand_properties_keys, all_definitions_dict, sub_cat_entity, curr_level+1)

            entity = await update_description(all_definitions_dict, entity, valid_keys)

        except Exception as e:
            print(f"Exception occurred during processing key_batch {key_batch}: {e}")

    return entity


def get_brand_properties(brand_name):
    specific_brand_path = f'./configs/specific_brand_data/{brand_name}.json'

    if os.path.exists(specific_brand_path):
        with open(specific_brand_path, 'r') as specific_file:
            brand_properties = json.load(specific_file)
        print("Using specific brand properties")
    else:
        with open('./configs/brand_data.json', 'r') as default_file:
            brand_properties = json.load(default_file)
        print("Using default brand properties")

    return brand_properties

def restructure_expanded_index(expanded_brand):
    """
    Restructures the expanded index to create a flattened dictionary where:
    - Keys are the paths of categories/subcategories
    - Values contain all existing info except subcategories details
    
    This makes it easier to access data without recursion.
    """
    result = {}
    expanded_brand_copy = deepcopy(expanded_brand)
    
    def process_entity(entity, parent_path=None):
        for key, data in entity.items():
            # Get the path for this entity
            if parent_path:
                path = parent_path + '->' + key
            else:
                path = key
                
            # Create a copy of the data without SUB_CATEGORIES
            entity_data = deepcopy(data)
            if SUB_CATEGORIES in entity_data:
                # Store only the names of subcategories
                if entity_data[SUB_CATEGORIES]:
                    entity_data[SUB_CATEGORY_NAMES] = list(entity_data[SUB_CATEGORIES].keys())
                    # Process subcategories recursively
                    process_entity(entity_data[SUB_CATEGORIES], path)
                    # Remove the full subcategories data
                    del entity_data[SUB_CATEGORIES]
                else:
                    entity_data[SUB_CATEGORY_NAMES] = []
                
            # Store the processed entity data
            result[path] = entity_data
    
    process_entity(expanded_brand_copy)
    return result


def convert_subcategories_and_precursors_to_full_paths(restructured_index):
    """
    Converts subcategory names and precursor names from basenames to their
    full path equivalents, if available in the restructured_index.
    """
    def get_full_precursor_path(curr_path, precursor_name):
        parent_path = '->'.join(curr_path.split('->')[:-1])
        return f"{parent_path}->{precursor_name}" if curr_path.count('->') else precursor_name

    def get_full_subcategory_path(curr_path, subcategory_name):
        return f"{curr_path}->{subcategory_name}"

    for full_path, data in restructured_index.items():
        data[SUB_CATEGORY_NAMES] = [get_full_subcategory_path(full_path, x) for x in data.get(SUB_CATEGORY_NAMES, [])]
        data[PRECURSOR] = [get_full_precursor_path(full_path, x) for x in data.get(PRECURSOR, [])]

        data[SUB_CATEGORY_NAMES] = [x for x in data[SUB_CATEGORY_NAMES] if x in restructured_index]
        data[PRECURSOR] = [x for x in data[PRECURSOR] if x in restructured_index]

async def update_initial_definitions(brand_name, brand_description, all_definitions):
    prompt_template = open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/update_initial_definition', 'r').read()
    prompt = prompt_template.format(
        brand_name = brand_name,
        brand_description = brand_description,
        curr_definitions = '\n'.join(f"- {k}: {v}" for k, v in all_definitions.items())
    )
    res = await llm.chat(prompt, model_name='gpt-4.1', task_name='update_initial_definitions', add_prompter=True)
    # print("Res is", res)
    res = parse_json(res, default_json=all_definitions)
    # print("res parsed is", res)
    for k, v in all_definitions.items():
        all_definitions[k] = res.get(k, v)
    return all_definitions

async def process(brand_name, brand_description):
    global CURR_RUN_START_TIME, MODEL_TO_USE
    CURR_RUN_START_TIME = time.time()
    MODEL_TO_USE = 'gemini-flash'

    log_dir = f'./pipeline_steps/{CURR_PIPELINE_STEP}/logs'
    os.makedirs(log_dir, exist_ok=True)
    open(f'{log_dir}/{brand_name}.txt', 'w').close()

    brand_properties = get_brand_properties(brand_name)
    brand_properties_keys = list(brand_properties.keys())
    all_definitions = {k: v[DESCRIPTION] for k, v in brand_properties.items()}

    all_definitions = await update_initial_definitions(brand_name, brand_description, all_definitions)
    for k, v in all_definitions.items():
        brand_properties[k][DESCRIPTION] = v

    expanded_brand = await expand_brand_index(brand_name, brand_description, brand_properties_keys, all_definitions, brand_properties)

    result_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}'
    os.makedirs(result_dir, exist_ok=True)

    # Restructure the expanded index for easier access
    restructured_index = restructure_expanded_index(expanded_brand)
    convert_subcategories_and_precursors_to_full_paths(restructured_index)

    pkl.dump(expanded_brand, open(f'{result_dir}/expanded_index.pkl', 'wb'))
    pkl.dump(all_definitions, open(f'{result_dir}/all_definitions.pkl', 'wb'))
    pkl.dump(restructured_index, open(f'{result_dir}/restructured_index.pkl', 'wb'))
    
    return expanded_brand, all_definitions, restructured_index