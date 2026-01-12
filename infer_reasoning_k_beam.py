import transformers
import torch
import random
from datasets import load_dataset
import requests
import os
from yurain_utils import *
import json 
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from kg_function import *
import re
from tqdm import tqdm
from collections import defaultdict
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.config:
        import json
        with open(args.config, "r") as f:
            config = json.load(f)
        return config
    return None


# KG Query Service Config
KG_QUERY_SERVICE_URL = os.getenv("KG_QUERY_SERVICE_URL", "http://localhost:5501")

def test_kg_service_connection():
    """Test KG query service connection"""
    try:
        response = requests.get(f"{KG_QUERY_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("KG success ✅")
            return True
        else:
            print("KG error ❌")
            return False
    except Exception as e:
        print(f"KG error ❌: {e}")
        return False

# Call test function
test_kg_service_connection()

def test_sparql_execution(query):
    query_url = f"{KG_QUERY_SERVICE_URL}/kg_query"
    headers = {"Content-Type": "application/json"}
    response = requests.post(query_url, json=query, headers=headers)
    if response.status_code != 200:
        return {
            'status': 'error',
            'message': f"Request failed, status code: {response.status_code}"
        }
    else:
        return response.json()

def kg_query_request(query_data, timeout=60):
    """Unified KG query request function"""
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(
            f"{KG_QUERY_SERVICE_URL}/kg_query",
            json={"query": query_data},
            timeout=timeout,
            headers=headers
        )
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                return result.get('results', {})
        return {}
    except Exception as e:
        print(f"KG time out: {e}")
        return {}

def query_full_sparql(sparql):
    """Query full SPARQL"""
    query_data = {
        "type": "sparql",
        "content": sparql,
        "parameters": {"idx": 0},
    }
    result = kg_query_request(query_data, timeout=30)
    return result.get("results", [])

def query_node_relation(mid=None, sparql="", question=""):
    """Query relations for a node"""
    if mid is None:
        return []
    
    query_data = {
        "type": "node",
        "content": mid,
        "entity_query": True,
        "parameters": {
            "idx": 0,
            "previous_sparql": sparql,
            "question": question,
            "filter_k": 10,
            "filter_threshold": 0.0
        }
    }
    result = kg_query_request(query_data, timeout=90)
    relations = result.get("results", [])
    if relations:
        return relations
    return []

def extract_outer_braces_content(s: str) -> str:
    stack = []
    start = end = -1
    
    for i, ch in enumerate(s):
        if ch == '{':
            if not stack:
                start = i
            stack.append(ch)
        elif ch == '}':
            stack.pop()
            if not stack:
                end = i
                break
    
    if start != -1 and end != -1:
        return s[start+1:end].strip()
    return ""

# prompt
reasoning_assistant_prompt = """You are an intelligent Q&A assistant capable of interacting with a knowledge graph (KG). By continuously interacting with the KG, you obtain the necessary information to answer questions and provide users with accurate and effective responses. The KG service is running normally; if no valid information is returned, it means that your interaction request contains an error."""

reasoning_prompt = """For every question asked by the user, you must include your reasoning inside <think> and </think> tags, and summarize your reasoning process and provide the final answer inside <answer> and </answer> tags.

When interacting with the KG, you may use <node> and <SPARQL> to communicate with the KG. The KG system will execute your queries and return corresponding information within <relation> and <information> tags.

You can output a key node inside <node> </node> tags to request exploration of that node. The KG will return all relations associated with that node. A key node may be either the ID of a concrete entity, or an intermediate variable from your SPARQL query. Returned relations are wrapped in <relation> tags. Within these relations, <node> indicates the position of the node you queried, serving as either the subject or object of the relation. ?a or ?b represent connected entities. You may only explore information through relations returned inside <relation> tags.

You may write a SPARQL query wrapped in <sparql> </sparql> tags to query information from the knowledge graph. The KG system will execute it and return results within <information> </information> tags. If your SPARQL query returns no results, you may attempt a different SPARQL query.

If you believe you have obtained sufficient information to support your answer, summarize your reasoning and provide the final answer within <answer> </answer> tags, and enclose the answer in \\boxed{{}}.

Question: {}

Additional Information:
The standard name of the entity involved in the question in the knowledge graph is:
{}"""

def process_input(input_text, tokenizer):
    messages = [
                {"role": "system", "content": reasoning_assistant_prompt},
                {"role": "user", "content": input_text}
            ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text

# Initialize the stopping criteria
target_sequences = [
    "</sparql>", "</SPARQL>", "</node>"
]

def extract_boxed_content(text):
    match = re.search(r'boxed\{(.*?)\}', text)
    if match:
        return match.group(1)
    else:
        return None

from openai import OpenAI
import logging
logging.getLogger().setLevel(logging.WARNING)

def main():
    # 1. Define default params
    config_params = {
        "model_path": os.getenv("MODEL_PATH", "./models/Qwen2.5-3B"),
        "k_samples": 16,
        "max_interactions": 6,
        "start_epoch": 1,
        "end_epoch": 2,
        "batch_size": 128,
        "save_path": "default_save_path",
        "dataset_path": os.getenv("DATASET_PATH", "./datasets/cwq/train.json"),
        "results_dir": "./results"
    }

    # 2. Load external config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.config:
        import json
        with open(args.config, "r") as f:
            config_params = json.load(f)

    # 3. Unpack params
    model_id = config_params["model_path"]
    K_SAMPLES = config_params["k_samples"]
    MAX_INTERACTIONS = config_params["max_interactions"]
    START_EPOCH = config_params["start_epoch"]
    END_EPOCH = config_params["end_epoch"]
    BATCH_SIZE = config_params["batch_size"]
    RESULTS_DIR = config_params["results_dir"]
    SAVE_PATH = os.path.join(RESULTS_DIR, config_params["save_path"])
    DATASET_PATH = config_params["dataset_path"]

    print(f"Config Loaded: Model={model_id}, Range={START_EPOCH}-{END_EPOCH}, SavePath={SAVE_PATH}")

    # load models
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, tensor_parallel_size=1, dtype="bfloat16", max_model_len=6000, gpu_memory_utilization=0.85)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    # load dataset
    full_dataset = read_json(DATASET_PATH)
    start_idx = START_EPOCH * BATCH_SIZE
    end_idx = END_EPOCH * BATCH_SIZE
    
    # Avoid out of bounds
    if start_idx >= len(full_dataset):
        print("Start index out of bounds, exiting.")
        return
    end_idx = min(end_idx, len(full_dataset))
    
    dataset = full_dataset[start_idx : end_idx]
    print(dataset[0]["question"])
    print(f"Processing dataset slice: {start_idx} to {end_idx} (Total: {len(dataset)})")
    # vllm accelerated inference evaluation
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=40, min_p=0.0, max_tokens=1024, stop=target_sequences)
    relation_format = "{prompt}{stop_reason}\n\n<relation>\n{search_results}</relation>\n\n"
    information_format = "{prompt}{stop_reason}\n\n<information>\n{search_results}</information>\n\n"

    # ========== Improved Data Structure ==========
    # Store complete tree structure
    trajectory_tree = {}  # trajectory_id -> node info

    # Store detailed info per step (including all candidates)
    step_details = []  # detailed record of each interaction round

    # Initialize data state
    active_trajectories = []
    completed_trajectories = []

    for idx, data in enumerate(dataset):
        global_sample_idx = idx
        ori_input = process_input(reasoning_prompt.format(data["question"], data["mention_entity"]), tokenizer)
        
        trajectory_id = f"sample_{global_sample_idx}_step_0"
        initial_state = {
            "trajectory_id": trajectory_id,
            "parent_id": None,
            "sample_index": global_sample_idx, # Use global index
            "prompt": ori_input,
            "previous_sparql": "",
            "cnt": 0,
            "question": data["question"],
            "answer": data["answer"],
            "complete": False,
            "relation_list": [],
            "full_trajectory": ori_input,
            "step": 0,
        }
        active_trajectories.append(initial_state)
        
        trajectory_tree[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "parent_id": None,
            "sample_index": global_sample_idx,
            "step": 0,
            "prompt": ori_input,
            "output": "",
            "children": [],
            "is_complete": False,
        }

    # Multi-turn interaction
    for interaction_round in range(MAX_INTERACTIONS):
        print(f"Interaction Round {interaction_round + 1}")
        
        if not active_trajectories:
            print("All trajectories are complete.")
            break

        # Prepare input list - copy k times for each active trajectory
        input_text_list = []
        state_mapping = []  # Record original trajectory index for each input
        
        for trajectory_idx, state in enumerate(active_trajectories):
            for _ in range(K_SAMPLES):
                input_text_list.append(state["prompt"])
                state_mapping.append(trajectory_idx)
        
        # Batch generate
        response_list = llm.generate(input_text_list, sampling_params=sampling_params)
        
        # Collect candidates
        candidates_map = defaultdict(list)  # parent_trajectory_id -> list of candidates
        
        # Step 1: Parse generation results
        for response_idx, response in enumerate(response_list):
            original_trajectory_idx = state_mapping[response_idx]
            parent_state = active_trajectories[original_trajectory_idx]
            
            output = response.outputs[0].text
            finish_reason = response.outputs[0].finish_reason
            stop_reason = response.outputs[0].stop_reason
            
            candidate = {
                "parent_trajectory_id": parent_state["trajectory_id"],
                "candidate_index": response_idx % K_SAMPLES,  # Intra-parent index
                "output": output,
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "query_type": None,
                "query_info": None,
                "is_complete": False,
                "predict": ""
            }
            
            if finish_reason == "stop" and stop_reason:
                if isinstance(stop_reason, str):
                    if stop_reason == "</node>":
                        identify_node = output.split("<node>")[-1].strip()
                        query_content = parent_state["question"]
                        candidate["query_type"] = "node"
                        candidate["query_info"] = {
                            "identify_node": identify_node,
                            "query_content": query_content,
                            "sparql": parent_state["previous_sparql"]
                        }
                    elif stop_reason.lower() == "</sparql>":
                        parts = re.split(r"<sparql>", output, flags=re.IGNORECASE)
                        step_sparql = parts[-1].strip()
                        candidate["query_type"] = "sparql"
                        candidate["query_info"] = {
                            "step_sparql": step_sparql
                        }
                    else:
                        candidate["is_complete"] = True
                else:
                    candidate["is_complete"] = True
            else:
                boxed_answer = extract_boxed_content(output)
                candidate["predict"] = boxed_answer if boxed_answer else ""
                candidate["is_complete"] = True
            
            candidates_map[parent_state["trajectory_id"]].append(candidate)

        # Step 2: Execute queries (Optimized: deduplicate identical queries)
        queries_to_run = []
        query_map = {}  # map for deduplication: query_key -> candidate list
        query_dedup_list = []  # deduplicated query list

        for parent_id, candidates in candidates_map.items():
            for cand in candidates:
                if cand["query_type"]:
                    queries_to_run.append(cand)
                    
                    # Generate unique ID for query
                    if cand["query_type"] == "node":
                        info = cand["query_info"]
                        # Note: query_key includes all parameters affecting result
                        query_key = (
                            "node",
                            info["identify_node"],
                            info["sparql"],
                            info["query_content"]
                        )
                    elif cand["query_type"] == "sparql":
                        info = cand["query_info"]
                        query_key = (
                            "sparql",
                            info["step_sparql"]
                        )
                    else:
                        continue
                    
                    # If new query, add to dedup list
                    if query_key not in query_map:
                        query_map[query_key] = []
                        query_dedup_list.append((query_key, cand))
                    
                    # Add candidate to corresponding query map
                    query_map[query_key].append(cand)

        if query_dedup_list:
            print(f"Total queries: {len(queries_to_run)}, Unique queries: {len(query_dedup_list)} (reduced {len(queries_to_run) - len(query_dedup_list)} duplicates)")
            print(f"Executing {len(query_dedup_list)} unique KG queries...")
            
            for query_key, cand in tqdm(query_dedup_list):
                try:
                    query_result = None
                    query_result_str = None
                    
                    if cand["query_type"] == "node":
                        info = cand["query_info"]
                        identify_node = info["identify_node"]
                        
                        node_relation_list = []
                        if identify_node.startswith("m.") or identify_node.startswith("g."):
                            node_relation_list = query_node_relation(mid=identify_node, sparql="", question=info["query_content"])
                        elif identify_node.startswith("?"):
                            if identify_node in info["sparql"]:
                                node_relation_list = query_node_relation(mid=identify_node, sparql=info["sparql"], question=info["query_content"])
                        elif isinstance(identify_node, str) and identify_node:
                            node_relation_list = query_node_relation(mid=identify_node, sparql="", question=info["query_content"])
                        
                        query_result = node_relation_list
                        query_result_str = json.dumps(node_relation_list, sort_keys=True)
                        
                    elif cand["query_type"] == "sparql":
                        info = cand["query_info"]
                        research_information = query_full_sparql(info["step_sparql"])
                        query_result = research_information
                        query_result_str = json.dumps(research_information, sort_keys=True)
                    
                    # Distribute query results to all identical candidates
                    for target_cand in query_map[query_key]:
                        target_cand["query_result"] = query_result
                        target_cand["query_result_str"] = query_result_str
                        
                except Exception as e:
                    print(f"Query execution error: {e}")
                    # Mark all identical candidates as error
                    for target_cand in query_map[query_key]:
                        target_cand["query_result"] = []
                        target_cand["query_result_str"] = "[]"
                        target_cand["error"] = str(e)
        
        # ========== Improved: Record detailed info per step ==========
        step_record = {
            "round": interaction_round + 1,
            "parent_nodes": []  # Record each parent node and all its candidates
        }
        
        for parent_id, candidates in candidates_map.items():
            parent_state = None
            for state in active_trajectories:
                if state["trajectory_id"] == parent_id:
                    parent_state = state
                    break
            
            parent_record = {
                "parent_trajectory_id": parent_id,
                "parent_prompt": parent_state["prompt"] if parent_state else "",
                "sample_index": parent_state["sample_index"] if parent_state else -1,
                "question": parent_state["question"] if parent_state else "",
                "all_candidates": [],  # All generated candidates
                "kept_candidates": [],  # Candidates kept after pruning
            }
            
            # Record all candidates
            for i, cand in enumerate(candidates):
                candidate_record = {
                    "candidate_index": i,
                    "output": cand["output"],
                    "stop_reason": cand["stop_reason"],
                    "query_type": cand["query_type"],
                    "query_info": cand["query_info"],
                    "query_result": cand.get("query_result_str", ""),
                    "is_complete": cand["is_complete"],
                    "predict": cand["predict"],
                }
                parent_record["all_candidates"].append(candidate_record)
            
            step_record["parent_nodes"].append(parent_record)
        
        # Step 3: Filter and update state
        next_step_trajectories = []
        child_counter = defaultdict(int)  # Count children for each parent
        
        for parent_id, candidates in candidates_map.items():
            parent_state = None
            for state in active_trajectories:
                if state["trajectory_id"] == parent_id:
                    parent_state = state
                    break
            
            if not parent_state:
                continue
            
            seen_results = set()
            kept_count = 0
            
            # Find corresponding parent_record
            parent_record = None
            for pr in step_record["parent_nodes"]:
                if pr["parent_trajectory_id"] == parent_id:
                    parent_record = pr
                    break
            
            for i, cand in enumerate(candidates):
                # Determine uniqueness key
                is_failed_query = cand["query_type"] and cand.get("query_result_str", "[]") == "[]"

                if is_failed_query:
                    # For failed queries, we consider different outputs unique failures
                    key = ("failed_query", cand["output"])
                elif cand["query_type"]:
                    # For successful queries, dedup by result
                    key = ("successful_query", cand["query_type"], cand.get("query_result_str"))
                elif cand["is_complete"]:
                    # For complete trajectories, dedup by output
                    key = ("complete", cand["output"])
                else:
                    key = ("other", cand["output"])
                
                # Check if kept
                is_kept = key not in seen_results
                
                if not is_kept:
                    continue
                
                seen_results.add(key)
                
                # Create new trajectory_id
                child_id = f"{parent_id}_child_{child_counter[parent_id]}"
                child_counter[parent_id] += 1
                
                # Create new state
                new_state = {
                    "trajectory_id": child_id,
                    "parent_id": parent_id,
                    "sample_index": parent_state["sample_index"],
                    "prompt": parent_state["prompt"],
                    "previous_sparql": parent_state["previous_sparql"],
                    "cnt": parent_state["cnt"],
                    "question": parent_state["question"],
                    "answer": parent_state["answer"],
                    "complete": False,
                    "relation_list": parent_state["relation_list"].copy(),
                    "full_trajectory": parent_state["full_trajectory"],
                    "step": interaction_round + 1,
                }
                
                output = cand["output"]
                stop_reason = cand["stop_reason"]
                new_state["full_trajectory"] += output
                
                # Record in tree structure
                tree_node = {
                    "trajectory_id": child_id,
                    "parent_id": parent_id,
                    "sample_index": parent_state["sample_index"],
                    "step": interaction_round + 1,
                    "prompt": new_state["prompt"],
                    "output": output,
                    "stop_reason": stop_reason,
                    "query_type": cand["query_type"],
                    "query_info": cand["query_info"],
                    "query_result": cand.get("query_result_str", ""),
                    "children": [],
                    "is_complete": cand["is_complete"],
                    "predict": cand.get("predict", ""),
                }
                
                if cand["is_complete"]:
                    new_state["predict"] = cand["predict"]
                    new_state["complete"] = True
                    new_state["prompt"] += output
                    tree_node["is_complete"] = True
                elif cand["query_type"] == "node":
                    node_relation_list = cand.get("query_result", [])
                    if node_relation_list:
                        new_state["relation_list"].append(node_relation_list)
                        new_state["cnt"] += 1
                        updated_prompt = relation_format.format(
                            prompt=new_state["prompt"] + output, 
                            stop_reason=stop_reason, 
                            search_results="\n".join(node_relation_list)
                        )
                        new_state["prompt"] = updated_prompt
                        new_state["full_trajectory"] = updated_prompt
                        tree_node["updated_prompt"] = updated_prompt
                    else:
                        updated_prompt = relation_format.format(
                            prompt=new_state["prompt"] + output, 
                            stop_reason=stop_reason, 
                            search_results="No relevant relations found for the given node query."
                            )
                        new_state["complete"] = True
                        new_state["prompt"] = updated_prompt
                        new_state["full_trajectory"] = updated_prompt
                        tree_node["is_complete"] = True
                elif cand["query_type"] == "sparql":
                    research_information = cand.get("query_result", [])
                    new_state["cnt"] += 1
                    if research_information:
                        new_state["previous_sparql"] = extract_outer_braces_content(cand["query_info"]["step_sparql"])
                        updated_prompt = information_format.format(
                            prompt=new_state["prompt"] + output, 
                            stop_reason=stop_reason, 
                            search_results=research_information
                        )
                        new_state["prompt"] = updated_prompt
                        new_state["full_trajectory"] = updated_prompt
                        tree_node["updated_prompt"] = updated_prompt
                    else:
                        updated_prompt = information_format.format(
                            prompt=new_state["prompt"] + output, 
                            stop_reason=stop_reason, 
                            search_results="No results found for the given SPARQL query.\nPlease try generating a different SPARQL query."
                        )
                        new_state["prompt"] = updated_prompt
                        new_state["full_trajectory"] = updated_prompt
                        new_state["complete"] = True
                        tree_node["updated_prompt"] = updated_prompt
                        tree_node["is_complete"] = True
                
                # Check length
                tokenize_prompt = tokenizer(new_state["prompt"], add_special_tokens=False)
                if len(tokenize_prompt["input_ids"]) >= 4096:
                    new_state["complete"] = True
                    tree_node["is_complete"] = True
                
                # Add to tree
                trajectory_tree[child_id] = tree_node
                # Update parent's children list
                if parent_id in trajectory_tree:
                    trajectory_tree[parent_id]["children"].append(child_id)
                
                next_step_trajectories.append(new_state)
                
                # Record kept candidate
                if parent_record:
                    parent_record["kept_candidates"].append({
                        "candidate_index": i,
                        "new_trajectory_id": child_id,
                    })
        
        step_details.append(step_record)
        
        # Update active and completed trajectories
        active_trajectories = []
        for state in next_step_trajectories:
            if state["complete"]:
                completed_trajectories.append(state)
            else:
                active_trajectories.append(state)
        
        total_trajectories = len(active_trajectories) + len(completed_trajectories)
        print(f"Round {interaction_round + 1}: Total trajectories: {total_trajectories}, Active: {len(active_trajectories)}, Completed: {len(completed_trajectories)}")

    # Move remaining active trajectories to completed
    completed_trajectories.extend(active_trajectories)

    import os
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print(f"\nSaving trajectory tree...")
    save_json(f"{SAVE_PATH}/k_sampling_trajectory_tree.json", trajectory_tree)
    
    print(f"Saving step-level details...")
    save_json(f"{SAVE_PATH}/k_sampling_step_details.json", step_details)

    final_results = []
    processed_traj_ids = set()

    for state in completed_trajectories:
        result = {
            "question": state["question"],
            "answer": state["answer"],
            "trajectory": state["full_trajectory"],
            "predict": state.get("predict", ""),
            "sample_index": state["sample_index"],
            "trajectory_id": state["trajectory_id"],
            "parent_id": state.get("parent_id"),
            "is_complete": True
        }
        final_results.append(result)
        processed_traj_ids.add(state["trajectory_id"])

    for traj_id, node in trajectory_tree.items():
        # 如果是叶子节点且未被处理过
        if not node["children"] and traj_id not in processed_traj_ids:
            sample_idx = node["sample_index"]
            
            full_text = node.get("updated_prompt", node["prompt"] + node["output"])
            
            result = {
                "question": dataset[sample_idx]["question"],
                "answer": dataset[sample_idx]["answer"],
                "trajectory": full_text,
                "predict": node.get("predict", ""),
                "sample_index": sample_idx,
                "trajectory_id": traj_id,
                "parent_id": node["parent_id"],
                "is_complete": False  # 标记为未完成/断头路
            }
            final_results.append(result)

    print(f"Saving final trajectories: {len(final_results)}")
    save_json(f"{SAVE_PATH}/k_sampling_final_trajectories.json", final_results)

    print("\nGenerating path summaries for each sample...")
    sample_summaries = defaultdict(lambda: {"question": "", "answer": "", "all_trajectories": [], "trajectory_count": 0})

    completed_traj_map = {ct["trajectory_id"]: ct for ct in completed_trajectories}

    for traj_id, node in trajectory_tree.items():
        sample_idx = node["sample_index"]
        
        if not node["children"]:
            path = []
            current_id = traj_id
            while current_id:
                if current_id in trajectory_tree:
                    path.insert(0, current_id)
                    current_id = trajectory_tree[current_id]["parent_id"]
                else:
                    break
            
            full_traj = completed_traj_map.get(traj_id)
            
            ground_truth = dataset[sample_idx]["answer"]
            question = dataset[sample_idx]["question"]
            
            sample_summaries[sample_idx]["question"] = question
            sample_summaries[sample_idx]["answer"] = ground_truth
            if full_traj:
                predict = full_traj.get("predict", "")
            else:
                predict = node.get("predict", "")
            is_correct = False
            if predict and ground_truth:
                is_correct = (predict == ground_truth)
                
            sample_summaries[sample_idx]["all_trajectories"].append({
                "trajectory_id": traj_id,
                "path": path,
                "predict": predict,
                "is_correct": is_correct,
                "is_complete": full_traj is not None,
                "length": len(path)
            })
            sample_summaries[sample_idx]["trajectory_count"] += 1

    sample_summaries_list = [{"sample_index": k, **v} for k, v in sample_summaries.items()]
    print(f"Saving sample summaries for {len(sample_summaries_list)} samples...")
    save_json(f"{SAVE_PATH}/k_sampling_sample_summaries.json", sample_summaries_list)

    print("\n=== Summary ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Total tree nodes: {len(trajectory_tree)}")
    print(f"Total completed trajectories: {len(completed_trajectories)}")
    print(f"Interaction rounds: {len(step_details)}")

if __name__ == "__main__":
    main()