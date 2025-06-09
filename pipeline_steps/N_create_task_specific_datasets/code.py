import pandas as pd
import pickle as pkl
import random
import json
from tqdm import tqdm
from copy import deepcopy
import os
from collections import defaultdict, Counter

CURR_PIPELINE_STEP = "N_create_task_specific_datasets"

def loadp(x):
    return pkl.load(open(x, 'rb'))

# ## AQM
##Already in format

# ## KB_REFINEMENT
## Ask karan to add to the script

# ## ARTICLE_SEARCH

def get_articles_df(brand_name):
    all_kbs_path = f'./checkpoints/H_generate_info_kbs/{brand_name}/all_kb_brand_index.pkl'
    all_kbs_index = loadp(all_kbs_path)
    all_kbs = {k: v['kb_article'].strip().strip('```markdown').strip('```').strip() for k, v in all_kbs_index.items()}
    
    articles_df = pd.DataFrame(all_kbs.items(), columns = ['document_key', 'document_content']).sample(frac=1).reset_index(drop=True)
    kb2id = {x: f"KB_{i+1}" for i, x in enumerate(articles_df.document_key.tolist())}
    articles_df['document_id'] = articles_df.document_key.apply(lambda x: kb2id[x])
    articles_df['document_path'] = articles_df.document_key.apply(lambda x: f"./articles/{kb2id[x]}.txt")
    articles_df = articles_df[['document_id', 'document_path', 'document_content']]
    return articles_df, kb2id

# ## MULTI_TURN_RAG
def formatted_conv_string(conv_obj):
    ret = []
    for role, msg, *others in conv_obj:
        role = 'Customer' if role == 'bot_customer' else 'Agent'
        ret.append(f"{role}: {msg}")
    return '\n'.join(ret)

def get_conv(conversation_objects, i, j):
    conv_obj = conversation_objects[i]
    clipped_conv_obj = conv_obj[:j]
    conv = formatted_conv_string(clipped_conv_obj)
    msg = formatted_conv_string(conv_obj[j:j+1])
    return conv, msg

def get_multi_turn_rag_df(brand_name, kb2id):
    conversation_objects = loadp(f'./checkpoints/K_simulate_conversations/{brand_name}/conv_objects.pkl')
    agent_personas,customer_personas = loadp(f'./checkpoints/K_simulate_conversations/{brand_name}/personas.pkl')
    
    kb_subset_metadata = []
    for i, conv_obj in enumerate(conversation_objects):
        for j, (role, msg, *others) in enumerate(conv_obj):
            if len(others) and others[0] == 'KNOWLEDGE_BASE':
                kb_subset_metadata.append((i, j))
    
    print("Message with KB tagged:", len(kb_subset_metadata))
    
    kb_recall_data = []
    for i, j in kb_subset_metadata:
        conv, msg = get_conv(conversation_objects, i, j)
        msg_data = conversation_objects[i][j]
    
        primary_kbs = agent_personas[i]['primary_kbs']
        relevant_kbs = msg_data[4]
        source_kb = msg_data[3]
    
        curr_sources = list(set(primary_kbs + [source_kb]).intersection(relevant_kbs))
        curr_ids = [kb2id[x] for x in curr_sources]
        kb_recall_data.append(
            (i, agent_personas[i], customer_personas[i], conv, msg, curr_sources,curr_ids)
        )
    median_len = [len(x[-1]) for x in kb_recall_data][len(kb_recall_data)//2]
    print("Median length of relevant KBs:", median_len)
    
    df = pd.DataFrame(kb_recall_data, columns = ['conversation_id', 'agent_persona', 'customer_persona', 'conversation_context', 'correct_agent_response', 'ordering', 'kb_ids'])
    df = df[['conversation_id', 'conversation_context', 'correct_agent_response','kb_ids']]
    return df

# ## TOOL_CALLING

def get_tools_data(brand_name):
    TOOLS_TO_PROVIDE = [16, 32, 64, 96, 128]
    convs = loadp(f"./checkpoints/K_simulate_conversations/{brand_name}/conv_objects.pkl")
    agent_personas, _ = loadp(f"./checkpoints/K_simulate_conversations/{brand_name}/personas.pkl")
    tools_dict = loadp(f"./checkpoints/I_generate_tools/{brand_name}/tools_dict_o4_final.pkl")

    def simplify_conversation(msg_objs):
        return [(r, m) for r, m, *_ in msg_objs]

    def extract_tool_calls(msg_objs):
        calls = []
        for idx, turn in enumerate(msg_objs):
            if turn[0] == "bot_agent" and len(turn) >= 4 and turn[2] == "TOOL_CALL":
                calls.append((idx, turn[3]))
        return calls

    # Extract all executed tool calls
    all_calls = []
    for ci in tqdm(range(len(convs)), desc="Scanning convs"):
        try:
            infos = extract_tool_calls(convs[ci])
            for idx, info in infos:
                all_calls.append((ci, idx, info))
        except Exception:
            continue

    # Normalize calls: (conv_idx, msg_idx, fn_names, fn_args)
    normalized = []
    for conv_idx, msg_idx, info in all_calls:
        names = [c["function"]["name"] for c in info["tool_calls"]]
        args = [eval(c["function"]["arguments"]) for c in info["tool_calls"]]
        normalized.append((conv_idx, msg_idx, names, args))

    # Identify top tools
    freq = Counter(t for _, _, fns, _ in normalized for t in fns)
    top_tools = [t for t, _ in freq.most_common()]
    remaining = [
        td["tool"]["function"]["name"]
        for td in tools_dict.values()
        if 'tool' in td and td["tool"]["function"]["name"] not in top_tools
    ]
    top_tools.extend(remaining)

    # Build balanced dataset
    random.shuffle(normalized)
    counter = defaultdict(int)
    dataset = []
    LIMIT = float('inf')
    for entry in normalized:
        names = entry[2]
        if len(names) == 1:
            if counter[names[0]] < LIMIT:
                dataset.append(entry)
                counter[names[0]] += 1
        else:
            dataset.append(entry)

    # Build conv_histories, system_messages, dataset_tools
    conv_histories = []
    dataset_tools = {k: [] for k in TOOLS_TO_PROVIDE}

    for conv_idx, msg_idx, names, _ in tqdm(dataset, desc="Building eval sets"):
        conv_histories.append(simplify_conversation(convs[conv_idx])[:msg_idx])
        for k in TOOLS_TO_PROVIDE:
            chosen = set(names)
            pool = deepcopy(top_tools[:k])
            noise = [t for t in pool if t not in chosen]
            combined = list(chosen) + noise
            selected = combined[:k]
            # print(selected)
            random.shuffle(selected)
            openai_tools = [
                td["tool"]
                for td in tools_dict.values()
                if 'tool' in td and td["tool"]["function"]["name"] in selected
            ]
            dataset_tools[k].append(openai_tools)

    # Save pickles
    # save_pickle(dataset, os.path.join(data_dir, "dataset.pkl"))
    # save_pickle(conv_histories, os.path.join(data_dir, "conv_histories.pkl"))
    # save_pickle(dataset_tools, os.path.join(data_dir, "dataset_tools.pkl"))

    # print("Preprocessing complete.")
    return dataset, conv_histories, dataset_tools


def get_tool_calling_dataset(brand_name, kb2id):
    dataset, conv_histories, dataset_tools = get_tools_data(brand_name)
    agent_personas, _ = loadp(f"./checkpoints/K_simulate_conversations/{brand_name}/personas.pkl")
    
    ds = defaultdict(list)
    
    for i in range(len(conv_histories)):
        ds['conversation_context'].append(get_conv_string(conv_histories[i]))
    
        persona_idx, _, correct_tools, _ = dataset[i]
        ap =  agent_personas[persona_idx]
    
        pkbs =  ap.get('primary_kbs', [])
        skbs = ap.get('secondary_kbs', []) 
        kbs = pkbs + skbs
        correct_tools = list(set(correct_tools))
        
        ds['kbs'].append([kb2id[x] for x in kbs])        
        ds['true_tools'].append(correct_tools)
    
        for k, v in dataset_tools.items():
            ds[f"tool_candidates{k}"].append([x['function']['name'] for x in v[i]])
    
    df = pd.DataFrame(ds)
    return df

def get_conv_string(conv_object):
    ret = []
    for role, msg in conv_object:
        role = 'Customer' if role == 'bot_customer' else 'Agent'
        ret.append(f"{role}: {msg}")
    return '\n'.join(ret)

def dump_file(brand_name, obj, file_name):
    file_dir =  f"./checkpoints/{CURR_PIPELINE_STEP}/{brand_name}"
    os.makedirs(file_dir, exist_ok=True)

    file_path = f"{file_dir}/{file_name}"

    if file_name.endswith('.pkl'):
        pkl.dump(obj, open(file_path, 'wb'))
    elif file_name.endswith('.xlsx'):
        obj.to_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_name}")

    print(f"File saved to {file_path}")

def export_datasets(brand_name, intent_model):
    articles_df, kb2id = get_articles_df(brand_name)
    dump_file(brand_name, articles_df, "articles.xlsx")
    dump_file(brand_name, kb2id, "kb2ids.pkl")

    aqm_df = pd.read_excel(f"./checkpoints/L_verify_qm_parameters/{brand_name}/qm_questions.xlsx")
    dump_file(brand_name, aqm_df, "aqm_df.xlsx")

    intent_conv_df = pd.read_excel(f'./checkpoints/M_rediscover_intents/{brand_name}/{intent_model}/conversations_tagged.xlsx')
    intent_conv_df = intent_conv_df[['Complete Conversations', 'Taxonomy_2', 'assigned_product']]
    intent_conv_df.columns = ['Complete Conversations', 'Taxonomy_2', 'Taxonomy_2_product']
    taxonomy_2 = pd.read_csv(f'./checkpoints/M_rediscover_intents/{brand_name}/{intent_model}/taxonomy_final.csv')
    dump_file(brand_name, intent_conv_df, "intent_conv_df.xlsx")
    dump_file(brand_name, taxonomy_2, "taxonomy_2.xlsx")

    multi_turn_rag_df = get_multi_turn_rag_df(brand_name, kb2id)
    dump_file(brand_name, multi_turn_rag_df, "multi_turn_rag_df.xlsx")

    tool_calling_df = get_tool_calling_dataset(brand_name, kb2id)
    dump_file(brand_name, tool_calling_df, "tool_calls.xlsx")

def process(brand_name, intent_model='gpt-4.1-mini'):
    export_datasets(brand_name, intent_model)

if __name__ == "__main__":
    brand_name = 'TerraBloom'
    process(brand_name)