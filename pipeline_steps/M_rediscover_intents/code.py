import os
import json
import pickle as pkl
import re
from collections import defaultdict
from typing import Dict, Any, List, Set

import pandas as pd
from tqdm.auto import tqdm

from utils.llm_helper import LLM
from utils.misc import parse_json

CURR_PIPELINE_STEP = "M_rediscover_intents"
MODEL_TO_USE = "gpt-4.1-mini"
# BRAND_NAME = "Momentum"

llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompts() -> Dict[str, str]:
    prompts_dir = os.path.join("./pipeline_steps", CURR_PIPELINE_STEP, "prompts")
    templates = {}
    for name in ["merge_product_names", "discover_product", "single_triage"]:
        path = os.path.join(prompts_dir, name)
        with open(path, "r") as f:
            templates[name] = f.read()
    return templates

PROMPTS = load_prompts()

def load_conversation_data(brand_name: str):
    conv_path = f"./checkpoints/K_simulate_conversations/{brand_name}/conv_objects.pkl"
    personas_path = f"./checkpoints/K_simulate_conversations/{brand_name}/personas.pkl"

    with open(conv_path, "rb") as f:
        conv_objects = pkl.load(f)

    with open(personas_path, "rb") as f:
        agent_personas, customer_personas = pkl.load(f)

    return conv_objects, agent_personas, customer_personas

def build_conv_df(conv_objects, agent_personas, customer_personas) -> pd.DataFrame:
    def extract_conv(conv):
        lines = []
        for role, msg, *rest in conv:
            speaker = "Agent" if "agent" in role.lower() else "Customer"
            lines.append(f"{speaker}: {msg}")
        return "\n".join(lines)
    records = []

    for i in range(len(agent_personas)):
        records.append({
            "og_idx": i,
            "Complete Conversations": extract_conv(conv_objects[i]),
            **agent_personas[i],
            **customer_personas[i]
        })
    return pd.DataFrame(records)

STOPWORDS = {
    'the', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'with', 'a', 'an',
    'at', 'by', 'from', 'all', 'etc', 'automation'
}

def extract_keywords(name: str) -> Set[str]:
    words = re.findall(r'\b\w+\b', name.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def find_product_name_clusters(product_types: Set[str]) -> List[Set[str]]:
    keywords_map = {p: extract_keywords(p) for p in product_types}
    clusters = []
    visited = set()
    for p in product_types:
        if p in visited:
            continue
        cluster = {p}
        fringe = {p}
        while fringe:
            current = fringe.pop()
            for other, other_kw in keywords_map.items():
                if other not in cluster and keywords_map[current] & other_kw:
                    cluster.add(other)
                    fringe.add(other)
        visited.update(cluster)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters

async def get_common_product_name(cluster_names: List[str]) -> str:
    cluster_list = "\n".join(f"- {n}" for n in cluster_names)
    prompt = PROMPTS["merge_product_names"].format(cluster_list=cluster_list)
    resp = await llm.chat(
        prompt,
        task_name="merge_product_names",
        model_name=MODEL_TO_USE,
        temperature=0.2
    )
    return resp.strip().strip('"')

async def merge_products_by_name_clusters(
    product_types: Set[str],
    composition: Dict[str, List[int]],
    labels: List[str]
):
    clusters = find_product_name_clusters(product_types)
    for cluster in clusters:
        names = sorted(cluster)
        size = sum(len(composition.get(n, [])) for n in names)
        if size > 300:
            continue
        merged = await get_common_product_name(names)
        if merged.upper() == "DO_NOT_MERGE":
            continue
        all_idxs = []
        for old in names:
            all_idxs.extend(composition.pop(old, []))
            product_types.discard(old)
        product_types.add(merged)
        composition[merged] = all_idxs
        labels[:] = [merged if lbl in names else lbl for lbl in labels]
    return product_types, composition, labels

async def assign_products_with_merging(
    conv_df: pd.DataFrame,
    log_filename: str = "products_log.txt",
    log_file_frequency: int = 100
):
    product_types: Set[str] = set()
    composition: Dict[str, List[int]] = defaultdict(list)
    labels: List[str] = []

    if os.path.exists(log_filename):
        os.remove(log_filename)

    merge_upper_bound = max(20, len(conv_df)//20)
    print("Setting merge_upper_bound to {}".format(merge_upper_bound))
    pbar = tqdm(total=len(conv_df), desc="Products identified")
    for idx, row in conv_df.iterrows():
        current_products = "\n".join(f"{p} : {len(v)}" for p, v in composition.items()) or "None yet."
        conversation = "\n".join(row['Complete Conversations'].split("\n")[:20])
        prompt = PROMPTS["discover_product"].format(
            current_products=current_products,
            conversation=conversation,
            merge_upper_bound=merge_upper_bound,
        )
        resp = await llm.chat(
            prompt,
            task_name="discover_product",
            model_name=MODEL_TO_USE,
            temperature=0.1
        )
        out = parse_json(resp, default_json={})

        if not out:
            labels.append('<ERROR>')
            print(f"Invalid response for idx {idx}, skipping")
            continue

        decision = out.get("decision", "")
        existing = out.get("existing_product_title", "")
        new_title = out.get("new_product_title", "")

        if decision == "EXISTING" and existing:
            product = existing
        elif decision == "CREATE_NEW" and new_title:
            product = new_title
            product_types.add(product)
        elif decision == "UPDATE_EXISTING" and existing and new_title:
            old, new = existing, new_title
            if old in product_types:
                product_types.remove(old)
                product_types.add(new)
                composition[new] = composition.pop(old, [])
                labels[:] = [new if lbl == old else lbl for lbl in labels]
            product = new
        else:
            product = "Unspecified"
            product_types.add(product)

        composition.setdefault(product, []).append(row['og_idx'])
        labels.append(product)

        pbar.update(1)
        if (idx + 1) % log_file_frequency == 0 or (idx + 1) == len(conv_df):
            with open(log_filename, "w") as f:
                f.write(f"After {idx+1} conversations: {len(product_types)} products\n")
                for p, v in sorted(composition.items(), key=lambda x: len(x[1])):
                    f.write(f"- {p} : {len(v)}\n")
            product_types, composition, labels = await merge_products_by_name_clusters(
                product_types, composition, labels
            )
    pbar.close()
    return labels, composition, product_types

def dict_to_numbered_lines(data: Dict[str, str]) -> str:
    if not data:
        return "None yet."
    return "\n".join(f"{i+1}. {k}: {v}" for i, (k, v) in enumerate(data.items()))

async def process_issue_triage(
    product_name: str,
    product_df: pd.DataFrame
):
    existing_issues: Dict[str, str] = {}
    composition: Dict[str, List[int]] = defaultdict(list)
    steps: List[str] = []

    for _, row in tqdm(product_df.iterrows(), total=len(product_df), desc=f"Cluster issues for product {product_name}"):
        conv = row["Complete Conversations"]
        issues_str = dict_to_numbered_lines(existing_issues)
        prompt = PROMPTS["single_triage"].format(issues_str, conv)
        resp = await llm.chat(
            prompt,
            task_name="single_triage",
            model_name=MODEL_TO_USE,
            temperature=0.1,
            add_prompter=False
        )

        try:
            out = parse_json(resp, default_json={})
            action = out.get("action", "")
            steps.append(action)

            if action == "IGNORE":
                key = out.get("existing_parent_title", "")
                composition.setdefault(key, []).append(row["og_idx"])
            elif action == "UPDATE_EXISTING":
                old = out.get("old_parent_title", "")
                new = out.get("new_parent_title", "")
                desc = out.get("new_parent_description", "")
                existing_issues.pop(old, None)
                existing_issues[new] = desc
                idxs = composition.pop(old, [])
                composition[new] = idxs + [row["og_idx"]]
            elif action == "CREATE_NEW":
                new = out.get("new_parent_title", "")
                desc = out.get("new_parent_description", "")
                existing_issues[new] = desc
                composition.setdefault(new, []).append(row["og_idx"])
        except Exception as e:
            print(f"Exception while processing issues for {product_name}: {e}")
            steps.append("<ERROR>")

    return existing_issues, composition, steps

def save_conv_df(
    conv_df: pd.DataFrame,
    brand_name: str
):
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}/{MODEL_TO_USE}"
    os.makedirs(save_dir, exist_ok=True)
    conv_df.to_excel(f"{save_dir}/conversations_tagged.xlsx", index=False)
    return conv_df

def save_taxonomies(
    taxonomies: Dict[str, Any],
    brand_name: str
):
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}/{MODEL_TO_USE}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/product_level_taxonomies.pkl", "wb") as f:
        pkl.dump(taxonomies, f)
    rows = []
    for prod, info in taxonomies.items():
        for title, desc in info["issues"].items():
            rows.append((prod, title, desc))
    df = pd.DataFrame(rows, columns=["L1", "L2", "Description"])
    df.to_csv(f"{save_dir}/taxonomy_final.csv", index=False)
    return df

def populate_conv_df(conv_df, all_taxonomies):
    # Tag conv_df with issue assignments
    row_to_issue = {}
    for prod, info in all_taxonomies.items():
        for issue_title, og_indices in info["composition"].items():
            for og_idx in og_indices:
                row_to_issue[og_idx] = (prod, issue_title)

    conv_df["assigned_issue_cluster"] = conv_df["og_idx"].map(lambda x: row_to_issue.get(x, (None, None))[1])
    conv_df["assigned_product"] = conv_df["og_idx"].map(lambda x: row_to_issue.get(x, (None, None))[0])

    def get_cluster_desc(x):
        prod, title = row_to_issue.get(x, (None, None))
        return all_taxonomies.get(prod, {}).get("issues", {}).get(title)

    conv_df["assigned_issue_description"] = conv_df["og_idx"].map(get_cluster_desc)
    conv_df["Taxonomy_2"] = conv_df.apply(
        lambda row: f"{row['assigned_product']}:{row['assigned_issue_cluster']}", axis=1
    )
    return conv_df

async def process(brand_name: str):
    conv_objects, agent_personas, customer_personas = load_conversation_data(brand_name)
    conv_df = build_conv_df(conv_objects, agent_personas, customer_personas)

    labels, prod_comp, prod_types = await assign_products_with_merging(conv_df)
    conv_df["product_label"] = labels
    conv_df = conv_df[conv_df['product_label'] != '<ERROR>'].reset_index(drop=True)

    ALL_TAXONOMIES = {}
    for prod in tqdm(sorted(prod_types), desc="Products"):
        subset = conv_df[conv_df["product_label"] == prod].reset_index(drop=True)
        issues, comp, steps = await process_issue_triage(prod, subset)
        ALL_TAXONOMIES[prod] = {
            "issues": issues,
            "composition": comp,
            "steps_taken": steps
        }

    taxonomies_df = save_taxonomies(ALL_TAXONOMIES, brand_name)
    conv_df = populate_conv_df(conv_df, ALL_TAXONOMIES)
    save_conv_df(conv_df, brand_name)

    return conv_df, taxonomies_df

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(process(BRAND_NAME))