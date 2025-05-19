import asyncio
import time
import os
import json
from collections import defaultdict, Counter
import random

# Third-party libraries
import numpy as np
import pickle as pkl
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports (assuming similar structure as reference code)
from constants import *
from utils.misc import parse_json
from utils.llm_helper import LLM

CURR_PIPELINE_STEP = "B_index_pruning"
MODEL_TO_USE = 'gemini-flash'
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
llm = LLM(log_file_name=CURR_PIPELINE_STEP)


def load_previous_step_data(brand_name):
    restructured_index_path = f'./checkpoints/A_index_generation/{brand_name}/restructured_index.pkl'

    with open(restructured_index_path, 'rb') as f:
        restructured_index = pkl.load(f)

    return restructured_index


def save_pruned_data(brand_name, pruned_index):
    # Output file paths
    save_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}'
    os.makedirs(save_dir, exist_ok=True)

    pruned_index_path = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}/pruned_index.pkl'

    with open(pruned_index_path, 'wb') as f:
        pkl.dump(pruned_index, f)


def create_cluster_analysis_prompt(cluster, restructured_index):
    prompt_template = open(
        f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/index_pruning').read()  # Build the paths and definitions string
    paths_and_defs = ""
    for key in sorted(cluster):
        paths_and_defs += f"\n\t- {key}: {restructured_index[key][DESCRIPTION]}"

    return prompt_template.format(paths_and_defs=paths_and_defs)


def parse_cluster_analysis_result(text):
    default_json = {
        "duplicates_found": True,
        "redundant_candidates": [
            {
                "reason": "Error occurred parsing LLM output.",
                "verdict": "false",
                "key": "ERROR_OCCURRED"
            }
        ]
    }
    return parse_json(text, default_json=default_json)


import copy


def remove_redundancies(restructured_index, redundant_keys, similar_pairs):
    # Create a deepcopy of restructured_index
    index_copy = copy.deepcopy(restructured_index)

    to_remove = []
    for k in list(index_copy.keys()):
        if any(k.startswith(prefix) for prefix in redundant_keys):
            to_remove.append(k)
    to_remove = sorted(to_remove)

    for key in to_remove:
        # If this key has a parent, remove the full path from the parent
        if '->' in key:
            parent_key = '->'.join(key.split('->')[:-1])
            if parent_key in index_copy and SUB_CATEGORY_NAMES in index_copy[parent_key]:
                if key in index_copy[parent_key][SUB_CATEGORY_NAMES]:
                    index_copy[parent_key][SUB_CATEGORY_NAMES].remove(key)

        # Print the key being removed along with any similar keys
        similar_keys = [pair['key2'] for pair in similar_pairs if pair['key1'] == key] + [pair['key1'] for pair in
                                                                                          similar_pairs if
                                                                                          pair['key2'] == key]
        print(f"Removing key: {key}, similar keys: {similar_keys}")

        # Delete the key from the index
        del index_copy[key]

    return index_copy


async def compute_embeddings(keys_list, values_list):
    key_embeddings = EMBEDDING_MODEL.encode(keys_list, show_progress_bar=False)
    value_embeddings = EMBEDDING_MODEL.encode(values_list, show_progress_bar=False)
    return key_embeddings, value_embeddings


def find_similar_pairs(keys_list, key_embeddings, value_embeddings, threshold=0.75):
    """Find pairs of items with similarity above threshold"""
    key_cos_sim = cosine_similarity(key_embeddings)
    value_cos_sim = cosine_similarity(value_embeddings)

    similar_pairs = []
    for i in range(len(keys_list)):
        for j in range(i + 1, len(keys_list)):
            if key_cos_sim[i, j] > threshold and value_cos_sim[i, j] > threshold:
                similar_pairs.append({
                    'key1': keys_list[i],
                    'key2': keys_list[j],
                    'similarity': key_cos_sim[i, j] * value_cos_sim[i, j]
                })

    return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)


def filter_related_pairs(similar_pairs):
    """Filter out siblings or parent-child relationships"""
    filtered_pairs = []
    for pair in similar_pairs:
        k1_parts = pair['key1'].split('->')
        k2_parts = pair['key2'].split('->')

        are_siblings = (len(k1_parts) == len(k2_parts)
                        and k1_parts[:-1] == k2_parts[:-1])

        is_ancestor = (
                (len(k1_parts) > len(k2_parts) and k2_parts == k1_parts[:len(k2_parts)])
                or
                (len(k2_parts) > len(k1_parts) and k1_parts == k2_parts[:len(k1_parts)])
        )

        if not (are_siblings or is_ancestor):
            filtered_pairs.append(pair)

    return filtered_pairs


def cluster_similar_items(similar_pairs):
    """Group similar items into clusters"""
    graph = defaultdict(set)
    for pair in similar_pairs:
        graph[pair['key1']].add(pair['key2'])
        graph[pair['key2']].add(pair['key1'])

    def get_connected_component(node, visited, adjacency):
        component = set()
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)
                stack.extend(neigh for neigh in adjacency[current] if neigh not in visited)
        return component

    clusters = []
    visited_nodes = set()
    for node in graph:
        if node not in visited_nodes:
            comp = get_connected_component(node, visited_nodes, graph)
            if comp:
                clusters.append(sorted(list(comp)))

    return clusters


def extract_keys_to_remove(parsed_results):
    """Extract keys marked for removal from LLM results"""
    keys_to_remove = []
    for i, result in enumerate(parsed_results):
        try:
            if result['duplicates_found']:
                for redundancy in result.get('redundant_candidates', []):
                    if str(redundancy.get('verdict', '')).lower() == 'true':
                        raw_key = redundancy['key'].split(':')[0].strip()
                        keys_to_remove.append(raw_key)
        except Exception as e:
            print(f"Error reading result for cluster {i}: {e}")
    return keys_to_remove

def remove_hanging_dependencies(pruned_restructured_index):
    """
    Converts subcategory names and precursor names from basenames to their
    full path equivalents, if available in the restructured_index.
    """
    for full_path, data in pruned_restructured_index.items():
        data[SUB_CATEGORY_NAMES] = [x for x in data[SUB_CATEGORY_NAMES] if x in pruned_restructured_index]
        data[PRECURSOR] = [x for x in data[PRECURSOR] if x in pruned_restructured_index]


async def process(brand_name):
    global MODEL_TO_USE
    # Load data
    restructured_index = load_previous_step_data(brand_name)
    keys_list = list(restructured_index.keys())
    values_list = [restructured_index[key][DESCRIPTION] for key in keys_list]

    key_embeddings, value_embeddings = await compute_embeddings(keys_list, values_list)

    similar_pairs = find_similar_pairs(keys_list, key_embeddings, value_embeddings)
    print(f"Number of similar pairs found: {len(similar_pairs)}")

    similar_pairs = filter_related_pairs(similar_pairs)
    print(f"Number of filtered pairs: {len(similar_pairs)}")

    clusters = cluster_similar_items(similar_pairs)

    prompts = [create_cluster_analysis_prompt(cluster, restructured_index) for cluster in clusters]
    results = await llm.chat_batch(
        prompts,
        task_name='cluster_analysis_for_index_pruning',
        batch_size=10,
        temperature=0.1,
        model_name=MODEL_TO_USE
    )

    parsed_results = [parse_cluster_analysis_result(r) for r in results]
    keys_to_remove = extract_keys_to_remove(parsed_results)

    print(f"Size of restructured index before removing redundant keys: {len(restructured_index)}")
    pruned_restructured_index = remove_redundancies(restructured_index, keys_to_remove, similar_pairs)
    print(f"Size of restructured index after removing redundant keys: {len(pruned_restructured_index)}")

    remove_hanging_dependencies(pruned_restructured_index)
    save_pruned_data(brand_name, pruned_restructured_index)

    print(f"[{CURR_PIPELINE_STEP}] Finished pruning for {brand_name}; Removed {len(keys_to_remove)} redundant keys.")

    return pruned_restructured_index
