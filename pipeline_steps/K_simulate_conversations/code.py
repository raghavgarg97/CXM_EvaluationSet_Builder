import ast
import asyncio
import os
import json
import random
import pickle as pkl
from collections import defaultdict
from fuzzywuzzy import process as fuzzywuzzy_process
from typing import Dict, Any, List

from collections import Counter
from sklearn.model_selection import train_test_split

import pandas as pd
from tqdm.auto import tqdm

from constants import *
from utils.constants import KB_ARTICLE
from utils.llm_helper import LLM
from utils.misc import parse_json

CURR_PIPELINE_STEP = "K_simulate_conversations"
MODEL_TO_USE = "gemini-flash"
llm = LLM(log_file_name=CURR_PIPELINE_STEP)


def load_pickle(x):
    return pkl.load(open(x, 'rb'))


import pandas as pd
import ast
from typing import List


def load_prompt(prompt_name: str) -> str:
    prompts_dir = f'pipeline_steps/{CURR_PIPELINE_STEP}/prompts'
    with open(os.path.join(prompts_dir, prompt_name), 'r', encoding='utf-8') as f:
        return f.read()


# Load all prompts once
PROMPTS = {
    'customer_system': load_prompt('customer_system_prompt_template'),
    'agent_system': load_prompt('agent_system_prompt_template'),
    'kb_clipping': load_prompt('kb_clipping_prompt_template'),
    'blueprint': load_prompt('blueprint_template'),
    'kb_ranking': load_prompt('kb_ranking_template'),
    'convert_qm_to_rules': load_prompt('convert_qm_to_rules'),
    'filter_qm_rules': load_prompt('filter_qm_rules'),
    'mock_tool_call': load_prompt('mock_tool_call'),
    'agent_response_from_tool_results': load_prompt('agent_response_from_tool_results')
}


def process_and_explode(df: pd.DataFrame, columns_to_process: List[str]) -> pd.DataFrame:
    df_processed = df.copy()

    # 1. Convert string representations to lists safely
    for col in columns_to_process:
        if col not in df_processed.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        try:
            df_processed[col] = df_processed[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        except (ValueError, SyntaxError) as e:
            print(f"Error converting column '{col}': {e}. Ensure strings are valid literals.")
            raise  # Halt execution on conversion error

    # 2. Check list length consistency per row
    def _check_row_lengths(row):
        if not columns_to_process:  # Handle empty list case
            return
        try:
            iter_lengths = iter(len(row[col]) for col in columns_to_process)
            first_len = next(iter_lengths)
            if not all(l == first_len for l in iter_lengths):
                lengths_dict = {col: len(row[col]) for col in columns_to_process}
                raise AssertionError(
                    f"Inconsistent list lengths at index {row.name}: {lengths_dict}"
                )
        except TypeError as e:
            # Handle cases where len() fails (e.g., item wasn't a list after conversion)
            raise TypeError(
                f"Error getting length at index {row.name}. Check data types after conversion. Original error: {e}")

    try:
        df_processed.apply(_check_row_lengths, axis=1)
    except AssertionError as e:
        print(f"Length consistency check failed: {e}")
        raise  # Halt execution on assertion error

    # 3. Explode the DataFrame on specified columns
    df_exploded = df_processed.explode(columns_to_process)

    return df_exploded


def preprocess(brand_name: str):
    tools_dict_path = f'./checkpoints/I_generate_tools/{brand_name}/tools_dict_o4_final.pkl'
    tools_dict = load_pickle(tools_dict_path)

    issue_df_path = f'./checkpoints/G_generate_issue_kbs/{brand_name}/issue_kbs.xlsx'
    issue_df = pd.read_excel(issue_df_path)

    brand_index_path = f'./checkpoints/H_generate_info_kbs/{brand_name}/all_kb_brand_index.pkl'
    brand_index = load_pickle(brand_index_path)

    qm_questions_paths = f'./checkpoints/J_generate_qm_parameters/{brand_name}/qm_top_questions.pkl'
    qm_top_questions_dict = load_pickle(qm_questions_paths)

    issueid2tools = defaultdict(set)
    for tool_name, tool_details in tools_dict.items():
        composition = tool_details['composition']
        for x in composition:
            issueid2tools[x['kb_uID']].add(tool_name)

    # Add tools to issue_df and brand_index
    issue_df['tools'] = issue_df.uID.apply(lambda x: list(issueid2tools.get(x, []))).tolist()
    for _, row in issue_df.iterrows():
        brand_index[f"Issues->{row.l3_intent}->{row.issue_cluster_title}"]['tools'] = row.tools

    issue_df['info_kbs_consolidated'] = issue_df.info_kbs.apply(
        lambda curr_info_kbs: [y for x in ast.literal_eval(curr_info_kbs) for y in x]
    )
    issue_df['info_kbs_consolidated'] = issue_df['info_kbs_consolidated'].apply(lambda x: list(set(x)))

    issue_df_exploded = process_and_explode(issue_df, columns_to_process=['customer_persona_titles',
                                                                          'customer_persona_descriptions', 'info_kbs'])
    print("Len of issue_df: ", len(issue_df))
    print("Len of issue_df_exploded: ", len(issue_df_exploded))

    return issue_df_exploded, brand_index, tools_dict, qm_top_questions_dict


def stratified_sample(df, n_convs, stratify_col='l3_intent', min_occurrences=2, random_state=42):
    if n_convs >= len(df):
        return df.copy()

    # 1. count and filter rare intents
    intent_counts = df[stratify_col].value_counts()
    valid_intents = intent_counts[intent_counts >= min_occurrences].index

    # 2. if you have more intents than samples, pick the top-n_convs intents
    if len(valid_intents) > n_convs:
        # keep only the n_convs most frequent intents
        valid_intents = intent_counts.loc[valid_intents].nlargest(n_convs).index
        print("Too less convs, picking top intents: ", valid_intents)

    # 3. filter the df
    valid_df = df[df[stratify_col].isin(valid_intents)]

    # 4. if you ended up with too few rows, fall back to original stratified split
    if len(valid_df) < n_convs:
        print(f"Warning: Not enough rows with {min_occurrences}+ occurrences. Using full dataset.")
        _, sampled_df = train_test_split(
            df,
            test_size=n_convs,
            stratify=df[stratify_col],
            random_state=random_state
        )
    else:
        # stratify over the (possibly pruned) valid_df
        _, sampled_df = train_test_split(
            valid_df,
            test_size=min(n_convs, len(valid_df)),
            stratify=valid_df[stratify_col],
            random_state=random_state
        )

    return sampled_df


def get_ancestors(kb_path, n = 2):
    path_list = kb_path.split('->')
    if path_list[0].strip() not in ['Product Range', 'Services Range']:
        return []

    return ['->'.join(path_list[:i + 1]) for i in range(len(path_list) - 1)][-n:]

def get_customer_personality():
    curr_persona = {}
    persona_traits = json.load(open('./configs/customer_persona_details.json'))
    for v in persona_traits.values():
        for x in random.sample(list(v['subcategories'].keys()), k=random.randint(1, 2)):
            val = random.choice(v['subcategories'][x]['all possible values'])
            curr_persona[x] = val

    persona_keys = random.sample(list(curr_persona.keys()), k=10)
    return '\n'.join([f"\t- {k}: {v}" for k, v in {x: curr_persona[x] for x in persona_keys}.items()])


async def populate_customer_personas(customer_personas, brand_description, mask_percentage):
    prompts = []
    prompt_template = PROMPTS['blueprint']
    for customer_persona in customer_personas:
        curr_aspects = '\n'.join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(customer_persona['entities'].items()))
        prompts.append(prompt_template.format(
            brand_overview=brand_description,
            customer_persona=customer_persona['personality'],
            aspect=curr_aspects,
            issue=customer_persona['issue']
        ))
    results = await llm.chat_batch(prompts, batch_size=20, model_name='gemini-flash',
                                   task_name='populate_customer_personas')
    results = [parse_json(x) for x in results]

    populated_customer_personas = []
    for res, customer_persona in zip(results, customer_personas):
        try:
            context = res['context']
            contextual_issue = res['issue_description']
            customer_behaviour = res['customer_behaviour']

            curr_mask_percentage = random.randint(mask_percentage - 10, mask_percentage + 10)
            masked_context = {k: ("NA" if random.randint(1, 100) < curr_mask_percentage else v) for k, v in
                              context.items()}

            customer_persona.update({
                'context': masked_context,
                'contextual_issue': contextual_issue,
                'customer_behaviour': customer_behaviour,
            })
            populated_customer_personas.append(customer_persona)
        except Exception as e:
            print(f"Exception: {e} occured during populating customer_personas")
            populated_customer_personas.append(None)

    return populated_customer_personas


def get_all_secondary_kbs(brand_index, info_kbs, curr_issue):
    all_precursors = [y for x in info_kbs for y in brand_index[x].get(PRECURSOR, []) if not y.startswith('Issues->')]
    all_ancestors = [y for x in info_kbs for y in get_ancestors(x, n=2)]
    return list(set(all_precursors + all_ancestors))


def get_qm_rules_subset(qm_top_questions_dict, l3_intent, issue_uID):
    qm_rule_candidates = qm_top_questions_dict[l3_intent]
    n_qm_rules = random.randint(8, 15)
    qm_rules = []
    for k, v in qm_rule_candidates.items():
        if issue_uID in v or random.choice([1, 2]) == 1:
            qm_rules.append(k)

        if len(qm_rules) == n_qm_rules:
            break
    return qm_rules


async def modify_agent_persona_qm(agent_personas, no_percentage):
    all_qm_questions = [x['qm'] for x in agent_personas]
    filter_qm_prompts = []
    for persona_qm_questions in all_qm_questions:
        qm_questions_str = '\n'.join([f"{i + 1}. {x}" for i, x in enumerate(persona_qm_questions)])
        qm_questions_prompt = PROMPTS['filter_qm_rules'].format(qm_rules=qm_questions_str)
        filter_qm_prompts.append(qm_questions_prompt)

    results = await llm.chat_batch(
        filter_qm_prompts,
        task_name='filer_qm_rules',
        model_name='gpt-4.1',
        add_prompter=True,
        batch_size=10
    )

    # print(random.choice(filter_qm_prompts))

    new_questions = [parse_json(res, default_json={'new_list': old_questions})['new_list'] for res, old_questions in
                     zip(results, all_qm_questions)]
    qm_sentences_prompts = []
    for agent_persona, curr_questions in zip(agent_personas, new_questions):
        yes_no_dict = {x: 'No' if random.randint(1, 100) <= no_percentage else 'Yes' for x in curr_questions}
        # print(Counter(list(yes_no_dict.values())))
        agent_persona['qm_dict'] = yes_no_dict
        qm_sentences_prompts.append(
            PROMPTS['convert_qm_to_rules'].format(qm=json.dumps(yes_no_dict, indent=2))
        )

    all_qm_rules_strings = await llm.chat_batch(qm_sentences_prompts, model_name='gemini-flash',
                                                task_name='convert_qm_to_rules', batch_size=20)

    # print(random.choice(qm_sentences_prompts))
    for i, qm_rule_string in enumerate(all_qm_rules_strings):
        all_qm_rules_strings[i] = '\n'.join([f"{i + 1}: {x}" for i, x in enumerate(qm_rule_string.splitlines()) if x])

    for agent_persona, qm_rules_string in zip(agent_personas, all_qm_rules_strings):
        agent_persona['qm'] = qm_rules_string

    return agent_personas


async def get_agent_and_customer_personas(issue_df_exploded, qm_top_questions_dict, brand_index, brand_description,
                                          n_convs):
    issue_df_exploded_filtered = issue_df_exploded[issue_df_exploded.l3_intent.isin(qm_top_questions_dict)].reset_index(drop=True)
    issue_df_sampled = stratified_sample(issue_df_exploded_filtered, n_convs)
    agent_personas, customer_personas = [], []
    for _, row in issue_df_sampled.iterrows():
        if row.l3_intent not in qm_top_questions_dict:
            print("Skipping")
            continue

        qm_rules = get_qm_rules_subset(qm_top_questions_dict, row.l3_intent, row.uID)

        curr_issue = f"Issues->{row.l3_intent}->{row.issue_cluster_title}"
        primary_related_info_kbs = row.info_kbs

        primary_kbs = primary_related_info_kbs + [curr_issue]
        secondary_kbs = get_all_secondary_kbs(brand_index, primary_related_info_kbs, curr_issue)
        all_kbs = primary_kbs + secondary_kbs

        all_tools = [y for x in all_kbs for y in brand_index[x].get('tools', [])]
        all_tools = list(set(all_tools))

        agent_personas.append({
            'primary_kbs': primary_kbs,
            'secondary_kbs': secondary_kbs,
            'tools': all_tools,
            'qm': qm_rules,
            'intent': row.l3_intent
        })

        info_kb_dict = {x: brand_index[x][SUMMARY] for x in row.info_kbs}
        customer_personality = get_customer_personality()
        customer_personas.append({
            'personality': customer_personality,
            'entities': info_kb_dict,
            'issue': row.customer_persona_descriptions
        })

    customer_personas = await populate_customer_personas(customer_personas, brand_description, mask_percentage=30)
    agent_personas = await modify_agent_persona_qm(agent_personas, no_percentage=30)

    no_error_mask = pd.notna(customer_personas)
    customer_personas = pd.Series(customer_personas)[no_error_mask].tolist()
    agent_personas = pd.Series(agent_personas)[no_error_mask].tolist()
    return agent_personas, customer_personas


def build_conversation_history_string(conversation_history):
    lines = []
    for role, text in conversation_history:
        role_label = "AGENT" if role == "bot_agent" else "CUSTOMER"
        lines.append(f"{role_label}:{text}")
    return "\n".join(lines)


async def get_relevant_agent_kbs_and_tools(brand_index, agent_persona, tools_dict, conversation_string):
    all_kbs = agent_persona.get('primary_kbs', [])  + agent_persona.get('secondary_kbs', [])
    all_kbs = list(set(all_kbs))

    all_tools = agent_persona.get('tools', [])
    all_tools_string = '\n'.join([
        f"\t- {x}: {tools_dict[x]['tool']['function']['description']}" for x in all_tools
    ])

    prompts = [PROMPTS['kb_clipping'].format(kb=brand_index[kb]['kb_article'], conv=conversation_string,
                                             skills=all_tools_string) for kb in all_kbs]
    results = await llm.chat_batch(prompts, model_name='gemini-flash', task_name='kb_clipping', batch_size=10,
                                   use_tqdm=False)
    results_parsed = [parse_json(res, default_json={'excerpt': res, 'actions': []}) for res in results]

    relevant_kbs = {}
    all_tools = []

    for kb_key, response in zip(all_kbs, results_parsed):
        if response['excerpt'].strip().lower() != '<none>':
            relevant_kbs[kb_key] = response['excerpt'].replace('`', '').strip()
            all_tools.extend(response['actions'])

    all_tools = [tools_dict[x]['tool'] for x in set(all_tools) if x in tools_dict]

    line_br = '-' * 50
    formatted_kbs_string = f'\n{line_br}\n'.join([
        f'''```    
KB Name: {k}
KB content:
{v}
```''' for i, (k, v) in enumerate(relevant_kbs.items())
    ])

    return formatted_kbs_string, all_tools, list(relevant_kbs.keys())


def parse_output_agent_str(response_text):
    response_obj = parse_json(response_text)
    if not response_obj:
        return response_text, "MODEL_FAILURE", "MODEL_FAILURE"

    return response_obj['message'], response_obj['category'], response_obj['source']


async def mock_tool_calls_batch(tool_specs_and_args, conversation_history_string):
    print("TOOL CALL TRIGGERED")
    prompts = [
        PROMPTS['mock_tool_call'].format(tool_spec=tool_spec, tool_args=tool_args,
                                         conversation_history_string=conversation_history_string)
        for tool_spec, tool_args in tool_specs_and_args
    ]
    # print("Mock tool call prompts are:", '\n\n'.join(prompts))
    results = await llm.chat_batch(prompts, task_name='mock_tool_call', batch_size=10, model_name='gemini-flash',
                                   use_tqdm=False)
    # print("Mock tool call results are:", '\n\n'.join(results))
    results_parsed = [parse_json(x, default_json={'mock_tool_result': 'TOOL CALL FAILED', 'sufficient_data': "NO"}) for
                      x in results]
    return results_parsed


async def generate_agent_response_from_tool_results(tool_names_and_descriptions, mock_results,
                                                    conversation_history_string, conv_lang):
    valid_mock_result_indices = [i for i, x in enumerate(mock_results) if x['sufficient_data'].lower().strip() == "yes"]
    invalid_mock_result_indices = [i for i, x in enumerate(mock_results) if
                                   x['sufficient_data'].lower().strip() != "yes"]

    final_result_indices = valid_mock_result_indices if valid_mock_result_indices else invalid_mock_result_indices
    result_type = 'TOOL_CALL' if valid_mock_result_indices else 'TOOL_INFO_GATHERING'

    index_mask = pd.Series(range(len(mock_results))).isin(final_result_indices)
    final_tools = dict(pd.Series(tool_names_and_descriptions.items())[index_mask].tolist())
    final_results = pd.Series(mock_results)[index_mask].tolist()

    tool_results_formatted = []
    for (tool_name, tool_description), result in zip(final_tools.items(), final_results):
        tool_results_formatted.append(
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Result: {result}\n"
        )
    tool_results_formatted = '\n'.join(tool_results_formatted)

    prompt = PROMPTS['agent_response_from_tool_results'].format(tool_results_formatted=tool_results_formatted,
                                                                conversation_history=conversation_history_string,
                                                                lang=conv_lang)

    response = await llm.chat(prompt, task_name='generate_agent_response_from_tool_results', temperature=0.7,
                              model_name='gemini-flash')
    return result_type, response.strip()


async def parse_output_agent_list(response_text, tools_agent, conversation_history_string, conv_lang):
    tool_specs_and_args = []
    tool_names_and_descriptions = {}
    for x in response_text:
        tool_name = x['function']['name']
        try:
            corresponding_tool = [y for y in tools_agent if y['function']['name'] == tool_name][0]
        except:
            print(f"Could not find any tool with name {tool_name}")
            tool_names = [tool['function']['name'] for tool in tools_agent]
            best_match_name, best_match_score = fuzzywuzzy_process.extractOne(tool_name, tool_names)
            corresponding_tool = next(tool for tool in tools_agent if tool['function']['name'] == best_match_name)
            print(f"Best match found using fuzzy match is {best_match_name}")

        tool_description = corresponding_tool['function']['description']
        tool_names_and_descriptions[tool_name] = tool_description

        curr_tool = corresponding_tool['function']
        curr_res = x['function']

        tool_specs_and_args.append((curr_tool, curr_res['arguments']))

    # Get mock responses for all tool calls at once
    mock_results = await mock_tool_calls_batch(tool_specs_and_args, conversation_history_string)
    # Get agent's response based on tool results
    result_type, agent_response = await generate_agent_response_from_tool_results(tool_names_and_descriptions,
                                                                                  mock_results,
                                                                                  conversation_history_string, conv_lang)
    # print("Result type:", result_type)
    # print("Agent response:", agent_response)
    return agent_response, result_type, {"tool_calls": response_text, "tool_results": mock_results}


async def parse_output_agent(response_text, tools_agent, conversation_history_string, conv_lang):
    if type(response_text) == str:
        result = parse_output_agent_str(response_text)
    else:
        result = await parse_output_agent_list(response_text, tools_agent, conversation_history_string, conv_lang)

    return result


async def handle_agent_turn(brand_index, conversation_history, conversation_history_simplified, agent_persona,
                            tools_dict, brand_description, agent_llm, conv_lang):
    conv_history_string = build_conversation_history_string(conversation_history_simplified)

    formatted_kbs_string, tools_agent, kbs_agent = await get_relevant_agent_kbs_and_tools(brand_index, agent_persona,
                                                                                          tools_dict,
                                                                                          conv_history_string)
    # print("All tools for this turn are:", tools_agent)

    qm_rules_string = agent_persona['qm']

    system_msg_agent = PROMPTS['agent_system'].format(
        brand_overview=brand_description,
        kbs=formatted_kbs_string,
        qm_rules=qm_rules_string,
        lang=conv_lang,
    )

    # if random.randint(1, 10) == 1:
    #     print(system_msg_agent)

    # print("System message of agent is", system_msg_agent)

    response_text = await llm.chat(
        conversation_history_simplified,
        model_name=agent_llm,
        perspective="agent",
        system_message=system_msg_agent,
        add_prompter=True,
        tools=tools_agent,
        task_name='conv_simulation_agent_turn',
    )
    # print("Response text is", response_text)

    # Check if no response was generated
    if not response_text:
        print("bot_agent did not respond. Terminating.")
        return False, False

    # Parse the agent response
    parsed_text, msg_class, msg_src = await parse_output_agent(
        response_text, tools_agent, conv_history_string, conv_lang
    )

    conversation_history.append(
        ("bot_agent", str(parsed_text), msg_class, msg_src, kbs_agent)
    )
    conversation_history_simplified.append(("bot_agent", str(parsed_text)))
    return True, False


def parse_func_customer(response_text):
    response_text = response_text.strip().splitlines()[0]
    should_terminate = "<EOC>" in response_text
    return response_text, should_terminate


async def handle_customer_turn(conversation_history, conversation_history_simplified, customer_system_prompt,
                               customer_llm):
    response_text = await llm.chat(
        roles_and_messages=conversation_history_simplified,
        perspective="customer",
        system_message=customer_system_prompt,
        model_name=customer_llm,
        task_name='conv_simulation_customer_turn'
    )

    if not response_text:
        # print("Not reasponse text hence terminating conversation")
        print("bot_customer did not respond. Terminating.")
        return False, False

    parsed_text, should_terminate = parse_func_customer(response_text)

    conversation_history.append(("bot_customer", parsed_text))
    conversation_history_simplified.append(("bot_customer", parsed_text))

    if should_terminate:
        # print("Bot wanted to terminate conversation:::", parsed_text)
        # print("bot_customer decided to end the conversation.")
        return True, True

    return True, False


async def simulate_conversation(
        brand_description,
        brand_index,
        agent_persona,
        customer_persona,
        agent_llm,
        customer_llm,
        default_first_message,
        tools_dict,
        max_turns,
        conv_lang
):
    context_string = '\n'.join([f"\t- {k}: {v}" for k, v in customer_persona['context'].items()])
    customer_system_prompt = PROMPTS['customer_system'].format(
        brand_overview=brand_description,
        issue_description=customer_persona['contextual_issue'],
        context=context_string,
        behaviour=customer_persona['customer_behaviour'],
        lang=conv_lang
    )
    # print("Customer system prompt is:", customer_system_prompt)
    assert isinstance(default_first_message, tuple) and default_first_message[0] in ['bot_customer', 'bot_agent']

    conversation_history = []
    conversation_history_simplified = [default_first_message]
    last_speaker = default_first_message[0]
    turn_count = 0

    break_convo_from_customer = False
    while turn_count < max_turns:
        try:
            # if True:
            turn_count += 1
            if last_speaker == "bot_agent":
                if break_convo_from_customer:
                    break

                did_respond, should_terminate = await handle_customer_turn(conversation_history,
                                                                           conversation_history_simplified,
                                                                           customer_system_prompt, customer_llm)

                if not did_respond or should_terminate:
                    break_convo_from_customer = True

                last_speaker = "bot_customer"
            else:
                did_respond, should_terminate = await handle_agent_turn(brand_index, conversation_history,
                                                                        conversation_history_simplified, agent_persona,
                                                                        tools_dict, brand_description, agent_llm, conv_lang)

                if not did_respond or should_terminate:
                    break

                last_speaker = "bot_agent"

            # print(*conversation_history_simplified[-1])

        except Exception as e:
            print(f"Exception {e} occured during processing a conversation")
    return conversation_history


import asyncio
import pickle as pkl
from tqdm import tqdm


async def run_simulations(
        brand_description,
        brand_index,
        tools_dict,
        agent_personas,
        customer_personas,
        n_semaphores: int,
        max_turns: int,
        conv_lang:str,
        agent_llm: str = 'gpt-4.1',
        customer_llm: str = 'gemini-flash'
):
    assert len(agent_personas) == len(customer_personas)

    pbar = tqdm(total=len(agent_personas), desc="Simulations Completed")
    semaphore = asyncio.Semaphore(n_semaphores)

    async def simulate_with_semaphore(agent_persona, customer_persona):
        async with semaphore:
            return await simulate_conversation(
                brand_description=brand_description,
                brand_index=brand_index,
                agent_persona=agent_persona,
                customer_persona=customer_persona,
                agent_llm=agent_llm,
                customer_llm=customer_llm,
                tools_dict=tools_dict,
                max_turns=max_turns,
                default_first_message=('bot_agent', "Hey! How can I help you?"),
                conv_lang=conv_lang
            )

    tasks = [
        asyncio.create_task(simulate_with_semaphore(a, c))
        for a, c in zip(agent_personas, customer_personas)
    ]

    for t in tasks:
        t.add_done_callback(lambda _t: pbar.update(1))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    results = []
    for idx, r in enumerate(raw_results):
        if isinstance(r, Exception):
            print(f"Simulation {idx} crashed:", repr(r))
            print("EXCEPTION")
        elif not r or not r[-1]:
            print(f"Simulation {idx} parsing failed")
            print("ERROR")
        else:
            results.append(r)

    # 7) Dump once at the end
    with open('./temp/greenbuild_convs.pkl', 'wb') as f:
        pkl.dump(results, f)

    return results


def save_personas(brand_name, agent_personas, customer_personas):
    save_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "personas.pkl"), "wb") as f:
        pkl.dump((agent_personas, customer_personas), f)


def save_conversations(brand_name, conversation_objects):
    save_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "conv_objects.pkl"), "wb") as f:
        pkl.dump(conversation_objects, f)


def load_personas(brand_name):
    save_dir = f'./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}/personas.pkl'
    if os.path.exists(save_dir):
        return load_pickle(save_dir)

    return None, None


async def process(brand_name, brand_description, n_convs, n_semaphores, max_turns, get_new_personas, conv_lang='English'):
    issue_df_exploded, brand_index, tools_dict, qm_top_questions_dict = preprocess(brand_name)

    agent_personas, customer_personas = load_personas(brand_name)
    if not agent_personas or get_new_personas:
        agent_personas, customer_personas = await get_agent_and_customer_personas(issue_df_exploded,
                                                                                  qm_top_questions_dict,
                                                                                  brand_index, brand_description,
                                                                                  n_convs)
        save_personas(brand_name, agent_personas, customer_personas)
    else:
        assert len(agent_personas) == n_convs, f"Previous run had different n_convs, callibrate or run afresh"
        print("Using previous personas!!")

    conversation_objects = await run_simulations(
        brand_description=brand_description,
        brand_index=brand_index,
        agent_personas=agent_personas,
        customer_personas=customer_personas,
        agent_llm='gpt-4.1',
        customer_llm='gemini-flash',
        max_turns=max_turns,
        tools_dict=tools_dict,
        n_semaphores=n_semaphores,
        conv_lang=conv_lang
    )
    save_conversations(brand_name, conversation_objects)
    return agent_personas, customer_personas, conversation_objects