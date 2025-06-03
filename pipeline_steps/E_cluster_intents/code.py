import asyncio
import os
import sys
import json
import random
import pickle as pkl
from collections import Counter
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from utils.llm_helper import LLM
from utils.misc import parse_json

CURR_PIPELINE_STEP = "E_cluster_intents"
MODEL_TO_USE = "gemini-flash"
llm = LLM(log_file_name=CURR_PIPELINE_STEP)

def load_prompt(prompt_name: str) -> str:
    """Loads a prompt text file from disk."""
    with open(f'pipeline_steps/{CURR_PIPELINE_STEP}/prompts/{prompt_name}', 'r') as file:
        return file.read()

def dict_to_key_value_lines(data: dict) -> str:
    """Converts a dict to simple key:value lines."""
    if not data:
        return "None yet."
    lines = []
    for k, v in data.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def dict_to_key_value_lines_numbered(data: dict) -> str:
    """Converts a dict to numbered key:value lines."""
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
    steps_taken: List[str],
    prompts: Dict[str, str]
):
    """
    For a single new issue, this function will:
    1. Run scenario 1 (ignore), scenario 2 (update), scenario 3 (create) in parallel.
    2. Parse scenario outputs.
    3. Run the verifier scenario to decide final action.
    4. Update existing_issues and existing_issues_composition accordingly.
    """
    try:
        existing_issues_str = dict_to_key_value_lines_numbered(existing_issues)
        new_issue_str = f"title: {new_issue_title}\ndescription: {new_issue_description}"

        # Prepare scenario prompts
        ignore_scenario_prompt = prompts['ignore'].format(
            current_issues=existing_issues_str,
            new_issue=new_issue_str
        )
        update_scenario_prompt = prompts['update'].format(
            current_issues=existing_issues_str,
            new_issue=new_issue_str
        )
        create_scenario_prompt = prompts['create'].format(
            current_issues=existing_issues_str,
            new_issue=new_issue_str
        )

        # Run all scenarios
        scenario_results = await llm.chat_batch(
            [ignore_scenario_prompt, update_scenario_prompt, create_scenario_prompt],
            task_name='check_all_scenarios',
            model_name=MODEL_TO_USE,
            batch_size=3,
            temperature=0.1,
            use_tqdm=False
        )

        # Extract scenario outputs
        ignore_output = parse_json(scenario_results[0])
        update_output = parse_json(scenario_results[1])
        create_output = parse_json(scenario_results[2])

        ignore_output_str = dict_to_key_value_lines(ignore_output)
        update_output_str = dict_to_key_value_lines(update_output)
        create_output_str = dict_to_key_value_lines(create_output)

        # Prepare verifier prompt
        verifier_prompt = prompts['verifier'].format(
            ignore_output=ignore_output_str,
            update_output=update_output_str,
            create_new_output=create_output_str,
            new_issue=new_issue_str
        )

        # Run the verifier scenario
        verifier_response = await llm.chat(
            verifier_prompt,
            task_name='verify_scenarios',
            model_name='gpt-4.1-mini',
            add_prompter=True,
            temperature=0.1
        )
        verifier_decision = parse_json(verifier_response)

        # Extract the final decision
        decision = verifier_decision.get("decision", "") if verifier_decision else 'IGNORE'
        if len(existing_issues) == 0:
            decision = 'CREATE_NEW'

        # print("Decision is:", decision)

        steps_taken.append(decision)

        # Update existing issues based on the decision
        if decision == "IGNORE":
            old_issue_title = ignore_output.get("existing_parent_title", "")
            if old_issue_title in existing_issues_composition and row_indices:
                existing_issues_composition[old_issue_title].extend(row_indices)

        elif decision == "UPDATE_EXISTING":
            old_issue_title = update_output.get("old_parent_title", "")
            new_title = update_output.get("new_parent_title", "")
            new_desc = update_output.get("new_parent_description", "")

            if old_issue_title in existing_issues:
                del existing_issues[old_issue_title]

            existing_issues[new_title] = new_desc

            old_indices = existing_issues_composition.pop(old_issue_title, [])
            existing_issues_composition.setdefault(new_title, []).extend(old_indices)
            if row_indices:
                existing_issues_composition[new_title].extend(row_indices)

        elif decision == "CREATE_NEW":
            new_title = create_output.get("new_parent_title", "")
            new_desc = create_output.get("new_parent_description", "")

            existing_issues[new_title] = new_desc
            if row_indices:
                existing_issues_composition.setdefault(new_title, []).extend(row_indices)
    except Exception as e:
        print("Exception occurred:", e, " while processing new issue.")


async def process_full_stream(
    issues_stream: pd.DataFrame,
    desc: str = 'Processing',
    prompts: Dict[str, str] = None
):
    """
    Given a DataFrame of new issues (with 'title' and 'description'),
    process them one by one to update 'existing_issues' and 'existing_issues_composition'.
    """
    existing_issues = {}
    existing_issues_composition = {}
    steps_taken = []

    for idx, issue in tqdm(issues_stream.iterrows(), total=len(issues_stream), desc=desc):
        new_issue_title = issue["title"]
        new_issue_description = issue["description"]
        await process_new_issue(
            new_issue_title,
            new_issue_description,
            [issue['og_idx']],
            existing_issues,
            existing_issues_composition,
            steps_taken,
            prompts
        )
    return existing_issues, existing_issues_composition, steps_taken


async def compress_existing_issues(
    existing_issues: Dict[str, str],
    existing_issues_composition: Dict[str, List[int]],
    scarcity_threshold: int,
    df: pd.DataFrame,
    prompts: Dict[str, str] = None
):
    """
    Merges 'scarce' clusters into refined clusters by reprocessing them
    until they become large enough.
    """
    scarce_indices = []
    for k, v in existing_issues_composition.items():
        if len(v) <= scarcity_threshold:
            scarce_indices.extend(v)

    print('='*50)
    print("Total clusters:", len(existing_issues_composition))
    print()
    print("Scarcity Threshold:", scarcity_threshold)
    print("Non-Scarce clusters:", len([1 for x in existing_issues_composition.values() if len(x) > scarcity_threshold]))
    print("Scarce clusters:", len([1 for x in existing_issues_composition.values() if len(x) <= scarcity_threshold]))
    print("Scarce indices:", len(scarce_indices))
    print('='*50)

    # Reprocess scarce items
    scarce_df = df[df.og_idx.isin(scarce_indices)].sample(frac=1, random_state=69).reset_index(drop=True)
    e_issues, e_comp, e_steps = await process_full_stream(scarce_df, desc='Processing scarce issues', prompts=prompts)

    finalised_issues = {k: v for k, v in existing_issues.items() if len(existing_issues_composition[k]) > scarcity_threshold}
    finalised_issues_composition = {k: v for k, v in existing_issues_composition.items() if len(v) > scarcity_threshold}

    # Merge scarce again into finalised
    for k, v in tqdm(e_issues.items(), total=len(e_issues), desc='Compressing existing issues'):
        new_issue_title = k
        new_issue_description = v
        await process_new_issue(
            new_issue_title,
            new_issue_description,
            e_comp[k],
            finalised_issues,
            finalised_issues_composition,
            [],
            prompts
        )
    return finalised_issues, finalised_issues_composition


async def process_and_compress_stream(
    df: pd.DataFrame,
    scarcity_threshold: int,
    compression_frequency: int
):
    """
    Repeatedly processes the issues and periodically
    merges sparse clusters to create robust groupings.
    """
    # Load prompts once
    prompts = {
        'ignore': load_prompt('enforce_ignore'),
        'update': load_prompt('enforce_update_old'),
        'create': load_prompt('enforce_create_new'),
        'verifier': load_prompt('verify_scenarios')
    }
    
    existing_issues = {}
    existing_issues_composition = {}
    steps_taken = []
    number_of_clusters_tracker = []

    progress_bar = tqdm(total=len(df), desc="Identified 0 clusters", dynamic_ncols=True)

    for idx, issue in df.iterrows():
        new_issue_title = issue["title"]
        new_issue_description = issue["description"]
        await process_new_issue(
            new_issue_title,
            new_issue_description,
            [issue['og_idx']],
            existing_issues,
            existing_issues_composition,
            steps_taken,
            prompts
        )

        if not (idx + 1) % compression_frequency:
            existing_issues, existing_issues_composition = await compress_existing_issues(
                existing_issues,
                existing_issues_composition,
                scarcity_threshold + (idx // (2 * compression_frequency)),
                df,
                prompts
            )

        progress_bar.update(1)
        progress_bar.set_description(f"Identified {len(existing_issues)} clusters")
        number_of_clusters_tracker.append(len(existing_issues))

    # Final compression pass
    existing_issues, existing_issues_composition = await compress_existing_issues(
        existing_issues,
        existing_issues_composition,
        scarcity_threshold + (len(df) // (2 * compression_frequency)),
        df,
        prompts
    )
    number_of_clusters_tracker.append(len(existing_issues))
    progress_bar.close()

    return existing_issues, existing_issues_composition, steps_taken, number_of_clusters_tracker


async def build_composition_details(
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
            'name': k,
            'description': existing_issues[k],
            'composition': [dict(df.loc[df.og_idx == x].reset_index(drop=True).iloc[0].items()) for x in v]
        }
    return composition_w_details


def export_plots(
    number_of_clusters_tracker: List[int],
    steps_taken: List[str],
    existing_issues_composition: Dict[str, List[int]],
    df: pd.DataFrame,
    brand_name: str
) -> None:
    """
    Exports all plots generated by the pipeline to a checkpoints directory.
    """
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 1) Plot the cluster tracker line chart
    plt.figure()
    plt.plot(number_of_clusters_tracker, label='Number of Clusters')
    # Mark points where the cluster count significantly decreases
    decrease_points = [
        i for i in range(1, len(number_of_clusters_tracker))
        if (number_of_clusters_tracker[i - 1] > number_of_clusters_tracker[i] + 5) or (i == 1)
    ]
    plt.plot(
        decrease_points,
        [number_of_clusters_tracker[i] for i in decrease_points],
        color='red',
        label='Decrease Points'
    )
    plt.xlabel('Index')
    plt.ylabel('Number of Clusters')
    plt.title('Number of Clusters Tracker with Decrease Points')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "number_of_clusters_tracker.png"))
    plt.close()

    # 2) Plot cumulative step counts
    df_steps = pd.DataFrame({"Steps": steps_taken})
    df_dummies = pd.get_dummies(df_steps["Steps"])
    cumulative_counts = df_dummies.cumsum()

    plt.figure()
    cumulative_counts.plot()
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Count of Steps")
    plt.title("Cumulative Step Counts Over Time")
    plt.legend(title="Step Type")
    plt.savefig(os.path.join(save_dir, "cumulative_step_counts.png"))
    plt.close()

    # 3) Plot distribution of cluster sizes
    value_lengths = [len(value_list) for value_list in existing_issues_composition.values()]
    length_counts = Counter(value_lengths)
    max_len = max(length_counts) if length_counts else 0
    if max_len > 0:
        x = list(range(1, max_len + 1))
        y = [length_counts.get(length, 0) for length in x]

        fig_width = max(6, 0.5 * len(x))
        plt.figure(figsize=(fig_width, 4))
        plt.bar(x, y)
        plt.xlabel("Length of Values")
        plt.ylabel("Number of Keys with that Length")
        plt.title("Distribution of List Lengths in existing_issues_composition")
        plt.xticks(x, x)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distribution_of_list_lengths.png"))
        plt.close()


def save_clustered_issues(brand_name: str, composition_w_details: Dict[str, Any]) -> None:
    """
    Saves the final clustered issues composition to a pickle file in the checkpoints directory.
    """
    save_dir = f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, "clustered_issues.pkl")
    with open(output_path, "wb") as f:
        pkl.dump(composition_w_details, f)


def load_issues_data(brand_name: str) -> pd.DataFrame:
    """
    Loads the issues from a json file and returns a DataFrame.
    For demonstration, brand_name is not used here.
    """
    issues = pkl.load(open(f'./checkpoints/D_generate_issues_and_resolutions/{brand_name}/issues_and_resolutions.pkl', 'rb'))
    random.Random(1).shuffle(issues)
    return pd.DataFrame(issues)

async def process(brand_name: str, scarcity_threshold=3, compression_frequency=500) -> Dict[str, Any]:
    """
    High-level process function for E_cluster_intents step. It:
    1) Loads the data for the given brand.
    2) Processes and compresses the stream of issues.
    3) Builds composition details for final clusters.
    4) Exports all relevant plots.
    5) Saves final clustered issues to a pickle file.
    """
    print(f"[{CURR_PIPELINE_STEP}] Loading issues for brand: {brand_name}")
    df = load_issues_data(brand_name)
    df['og_idx'] = range(len(df))

    print(f"[{CURR_PIPELINE_STEP}] Processing and compressing stream for brand: {brand_name}")
    existing_issues, existing_issues_composition, steps_taken, number_of_clusters_tracker = await process_and_compress_stream(
        df=df,
        scarcity_threshold=scarcity_threshold,
        compression_frequency=compression_frequency
    )

    print(f"[{CURR_PIPELINE_STEP}] Building composition details for brand: {brand_name}")
    composition_w_details = await build_composition_details(existing_issues, existing_issues_composition, df)

    print(f"[{CURR_PIPELINE_STEP}] Exporting plots for brand: {brand_name}")
    export_plots(number_of_clusters_tracker, steps_taken, existing_issues_composition, df, brand_name)

    print(f"[{CURR_PIPELINE_STEP}] Saving final results for brand: {brand_name}")
    save_clustered_issues(brand_name, composition_w_details)

    print(f"[{CURR_PIPELINE_STEP}] Finished clustering intents for brand: {brand_name}")
    return composition_w_details