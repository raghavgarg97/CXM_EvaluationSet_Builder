import os
import asyncio
import pickle as pkl
import pandas as pd
from constants import *
from utils.llm_helper import LLM

CURR_PIPELINE_STEP = "G_generate_issue_kbs"
MODEL_TO_USE = "gemini-flash"


llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt text file from the disk for this pipeline step."""
    with open(f'./pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}', 'r') as file:
        return file.read()

def load_clusters_within_intents(brand_name: str) -> dict:
    load_path = f"./checkpoints/F_cluster_issue_kbs/{brand_name}/intent_wise_clustered_issues.pkl"
    with open(load_path, "rb") as f:
        l3_intent_clusters = pkl.load(f)
    return l3_intent_clusters

def save_issue_kbs(brand_name: str, issue_kb_data: list) -> None:
    """Saves the final clustered issues composition to an Excel file."""
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)
    issue_df = pd.DataFrame(issue_kb_data)
    issue_df['uID'] = [f"ISSUE_{i}" for i in range(1, len(issue_kb_data) + 1)]
    issue_df.to_excel(f'{save_dir}/issue_kbs.xlsx', index=False)

def get_kb_data(l3_intent, issue_clusters):
    """Generates knowledge base data from intent clusters."""
    kb_data = []
    for issues_cluster_title, issues_cluster_details in issue_clusters.items():
        cluster_details_string = []
        cluster_info_kbs = []
        customer_persona_titles = []
        customer_persona_descriptions = []
        for i, issue_details in enumerate(issues_cluster_details['composition']):
            cluster_details_string.append(
                f"ISSUE {i + 1} TITLE: {issue_details['title']}\n"
                f"ISSUE {i + 1} RESOLUTION: {issue_details['resolution']}"
            )
            cluster_info_kbs.append(issue_details['kbs'])
            customer_persona_titles.append(issue_details['title'])
            customer_persona_descriptions.append(issue_details['description'])
        line_br = '-' * 40 + '\n'
        cluster_details_string = f'\n{line_br}\n'.join(cluster_details_string)
        kb_data.append({
            L3_INTENT: l3_intent,
            ISSUE_CLUSTER_TITLE: issues_cluster_title,
            ISSUE_CLUSTER_DESCRIPTION: issues_cluster_details['description'],
            CUSTOMER_PERSONA_TITLES: customer_persona_titles,
            CUSTOMER_PERSONA_DESCRIPTIONS: customer_persona_descriptions,
            RESOLUTION_DETAILS: cluster_details_string,
            INFO_KBS: cluster_info_kbs,
        })
    return kb_data

async def populate_kb_articles(kb_data, kb_lang):
    """Populates knowledge base articles based on generated prompts."""
    kb_generation_template = load_prompt('kb_generation')
    prompts = [kb_generation_template.format(resolution_details=v[RESOLUTION_DETAILS], lang=kb_lang) for v in kb_data]
    generated_kbs = await llm.chat_batch(
        prompts,
        task_name='kb_generation',
        model_name=MODEL_TO_USE,
        batch_size=10,
        temperature=0.6,
    )

    for kb_article, kb_dict in zip(generated_kbs, kb_data):
        if kb_article:
            kb_dict[KB_ARTICLE] = kb_article

async def generate_kb_articles(l3_intent_clusters, kb_lang):
    """Generates knowledge base articles for all intent clusters."""
    all_kb_data = []
    for l3_intent, issue_clusters in l3_intent_clusters.items():
        all_kb_data.extend(get_kb_data(l3_intent, issue_clusters))
    await populate_kb_articles(all_kb_data, kb_lang)
    return all_kb_data

async def populate_new_brand_data(brand_overview, issue_kb_data):
    info_kb_data_extraction_template = load_prompt('brand_data_extraction')
    prompts = [
        info_kb_data_extraction_template.format(
            BRAND_OVERVIEW=brand_overview,
            kb=x[KB_ARTICLE]
        ) for x in issue_kb_data
    ]
    results = await llm.chat_batch(
        prompts,
        task_name='brand_data_extraction',
        model_name=MODEL_TO_USE,
        batch_size=10,
        temperature=0.6,
    )

    for issue_kb_dict, res in zip(issue_kb_data, results):
        issue_kb_dict[NEW_BRAND_DATA] = res if res else 'NA'


async def process(brand_name: str, brand_overview: str, kb_lang: str = 'English') -> list:
    """Processes the L3 intents data for the given brand and saves results."""
    l3_intent_clusters = load_clusters_within_intents(brand_name)
    issue_kb_data = await generate_kb_articles(l3_intent_clusters, kb_lang)
    await populate_new_brand_data(brand_overview, issue_kb_data)
    save_issue_kbs(brand_name, issue_kb_data)
    return issue_kb_data
