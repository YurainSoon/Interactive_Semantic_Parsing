import json
import re
import sys
import requests
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import ast
import argparse
import os
from yurain_utils import *

try:
    from kg_rl import (
        F1_score, 
        validate_xml_tags, 
        check_xml_enter,
        extract_boxed_answer,
        check_sparql_syntax,
        is_failure,
    )
except ImportError:
    print("Warning: Could not import kg_rl. Please ensure 'verl' project is in PYTHONPATH.")
    pass 


def extract_node_content(output_text: str) -> Optional[str]:
    """Extract content from <node> tag in output"""
    node_pattern = r'<node>\n\s*(.*?)\s*</node>'
    match = re.search(node_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def extract_sparql_from_text(output_text: str) -> Optional[str]:
    """Extract <sparql> content from output"""
    sparql_pattern = r'<sparql>\n\s*(.*?)\s*</sparql>'
    sparql_match = re.search(sparql_pattern, output_text, re.DOTALL | re.IGNORECASE)
    if sparql_match:
        return sparql_match.group(1).strip()
    return None


def extract_mention_entity(prompt: str) -> Optional[str]:
    """Extract mention entity from prompt"""
    try:
        mention_entity = prompt.split("The standard name of the entity involved in the question in the knowledge graph is:\n")[-1].split("<|im_end|>")[0].strip()
        if mention_entity:
            return ast.literal_eval(mention_entity)
    except Exception as e:
        # print("Error extracting mention entity:", e)
        return []


def extract_select_variable(sparql: str) -> Optional[str]:
    """Extract variable after SELECT DISTINCT in SPARQL"""
    pattern = r"SELECT\s+DISTINCT\s+(\?\w+)"
    match = re.search(pattern, sparql, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def flatten_sparql_result(query_result: Any) -> List[str]:
    """Flatten SPARQL query results into a list of strings"""
    if not query_result:
        return []
    if isinstance(query_result, list):
        if not query_result:
            return []
        # If it is a list of strings
        if isinstance(query_result[0], str):
            return query_result
        # If it is a list of dicts (bindings)
        if isinstance(query_result[0], dict):
            # Extract all values
            values = []
            for item in query_result:
                for v in item.values():
                    values.append(str(v))
            return values
    return []


def compute_distance_to_answer(
    entities: List[str], 
    ground_truth: List[str], 
    distance_api_url: str, 
    max_distance: int = 3
) -> float:
    """
    Compute distance score from entity list to ground truth answers
    Returns normalized distance score [0, 1], larger is better (smaller distance -> higher score)
    """
    if not entities or not ground_truth:
        return 0.0
    
    # First try F1 matching
    try:
        f1 = F1_score(entities, ground_truth)
        if f1 >= 0.9:
            return 1.0  # Perfect hit
        elif f1 > 0:
            return 0.5 + f1 * 0.5  # Partial hit, score in [0.5, 1.0]
    except:
        pass
    
    # Call distance API
    try:
        payload = {
            "set_a": entities,
            "set_b": ground_truth,
            "max_distance": max_distance,
            "early_stop_global_min": 1
        }
        response = requests.post(distance_api_url, json=payload, timeout=300)
        if response.status_code == 200:
            distance = response.json().get("distance", max_distance)
            distance = max(0, min(max_distance, float(distance)))
            # Normalize: smaller distance, higher score
            normalized_score = (max_distance - distance) / max_distance
            return normalized_score
    except Exception as e:
        print(f"Warning: distance API call failed: {e}")
    
    return 0.0


def score_trajectory_path(
    path_ids: List[str],
    trajectory_tree: Dict[str, Dict[str, Any]],
    ground_truth: List[str],
    trajectory_final_f1: float,
    distance_api_url: str,
    max_distance: int = 3
) -> List[Dict[str, Any]]:
    """
    Score a complete trajectory path (New Mechanism)
    """
    if not path_ids:
        return []

    # 1. Extract mention entity and compute its base score
    root_node = trajectory_tree[path_ids[0]]
    prompt = root_node.get("prompt", "")
    mention_entity = extract_mention_entity(prompt)
    
    mention_entity_score = 0.0
    if mention_entity:
        mention_entity_score = compute_distance_to_answer([mention_entity], ground_truth, distance_api_url, max_distance)
    
    # 2. Initialize state
    variable_scores = {}  # variable name -> max score (min distance)
    executed_sparqls = set()
    
    # Record type and score of the previous valid node
    prev_node_state = {
        "type": "START",
        "score": 0.0
    }
    
    step_scores = []
    
    for node_id in path_ids:
        if node_id not in trajectory_tree:
            continue
        
        tree_node = trajectory_tree[node_id]
        output_text = tree_node.get("output", "")
        query_type = tree_node.get("query_type")
        
        # ========== 1. Format Score ==========
        # -1: Invalid format, 1: Valid format
        format_score = 1.0
        has_think = '<think>' in output_text.lower() and '</think>' in output_text.lower()
        if query_type == "sparql":
            output_text += "</SPARQL>"
        elif query_type == "node":
            output_text += "</node>"
        if not has_think:
            format_score = -1.0
        elif not validate_xml_tags(output_text):
            format_score = -1.0
        elif not check_xml_enter(output_text)[0]:
            format_score = -1.0
        else:
            format_score = 1.0
            
        # ========== 2. Progress Score ==========
        # -1: Error, 0: No progress/Base step, 1: Progress made/Perfect
        progress_score = 0.0
        current_node_type = "OTHER"
        current_node_score = 0.0
        
        # Identify node type and score
        if query_type == "node":
            current_node_type = "NODE"
            node_content = extract_node_content(output_text)
            
            is_mention = False
            if mention_entity and node_content and node_content in mention_entity:
                is_mention = True
            
            is_variable = False
            if node_content and node_content in variable_scores:
                is_variable = True
                
            if is_mention:
                progress_score = 0.0
                current_node_score = mention_entity_score
            elif is_variable:
                progress_score = 0.0
                current_node_score = variable_scores[node_content]
            else:
                # Neither mention entity nor known intermediate variable -> Error
                progress_score = -1.0
                current_node_score = 0.0
                
        elif query_type == "sparql":
            current_node_type = "SPARQL"
            sparql = extract_sparql_from_text(output_text)
            
            # Syntax check
            if not sparql or not check_sparql_syntax(sparql):
                progress_score = -1.0
            # Repetition check
            elif sparql in executed_sparqls:
                progress_score = -1.0
            else:
                executed_sparqls.add(sparql)
                
                # Get query result and compute score
                query_result = tree_node.get("query_result", [])
                if is_failure(query_result):
                    continue
                if isinstance(query_result, str):
                    try:
                        query_result = ast.literal_eval(query_result)
                    except:
                        query_result = []
                flat_result = flatten_sparql_result(query_result)
                result_score = compute_distance_to_answer(flat_result, ground_truth, distance_api_url, max_distance)
                current_node_score = result_score
                
                # Update variable scores
                selected_var = extract_select_variable(sparql)
                if selected_var:
                    old_var_score = variable_scores.get(selected_var, 0.0)
                    variable_scores[selected_var] = max(old_var_score, result_score)
                
                # Calculate Progress Score
                if prev_node_state["type"] == "NODE" and progress_score != -1.0:
                    # Case A: Previous node is NODE
                    # Compare current result score with previous node score
                    # If result_score > prev_score (closer distance), then +1
                    if result_score > prev_node_state["score"]:
                        progress_score = 1.0
                    else:
                        progress_score = 0.0
                else:
                    # Case B: Previous node is not NODE (could be SPARQL or other)
                    # Check F1
                    f1 = 0.0
                    if flat_result:
                        f1 = F1_score(flat_result, ground_truth)
                    
                    if f1 >= 0.9:
                        progress_score = 1.0
                    else:
                        progress_score = 0.0
        
        elif "</answer>" in output_text:
            current_node_type = "ANSWER"
            progress_score = 0.0
        else:
            # Other cases
            progress_score = 0.0
            
        # ========== 3. Outcome Score ==========
        # If format or progress score is negative, cancel outcome score
        outcome_score = 0.0
        if format_score == -1.0 or progress_score == -1.0:
            outcome_score = 0.0
        else:
            outcome_score = trajectory_final_f1
            
        # ========== 4. Total Score Calculation ==========
        total_score = 0.1 * format_score + 0.3 * progress_score + 0.6 * outcome_score
        
        # Update prev_node_state
        # Only update state when current node is valid, serving as baseline for next node
        if current_node_type == "NODE" and progress_score != -1.0:
            prev_node_state = {"type": "NODE", "score": current_node_score}
        elif current_node_type == "SPARQL" and progress_score != -1.0:
            prev_node_state = {"type": "SPARQL", "score": current_node_score}
        # If ANSWER or other, or invalid node, do not update (or reset?)
        elif progress_score == -1.0:
             prev_node_state = {"type": "INVALID", "score": 0.0}
        
        step_scores.append({
            "trajectory_id": node_id,
            "step": tree_node.get("step", 0),
            "output": output_text,
            "query_type": query_type,
            "progress_score": progress_score,
            "format_score": format_score,
            "outcome_score": outcome_score,
            "total_score": round(total_score, 4),
            "current_distance": current_node_score,
        })
        
    return step_scores


def process_k_beam_results(
    trajectory_tree_path: str,
    sample_summaries_path: str,
    output_path: str,
    distance_api_url: str,
    max_distance: int = 3
):
    print("Loading data files...")
    with open(trajectory_tree_path, 'r', encoding='utf-8') as f:
        trajectory_tree = json.load(f)
    
    with open(sample_summaries_path, 'r', encoding='utf-8') as f:
        sample_summaries = json.load(f)
    
    # Step 1: Compute F1 scores for all trajectories and score each node in the path
    print("Computing F1 scores for all trajectories...")
    node_trajectory_scores = defaultdict(list)  # node_id -> list of scores from its descendant trajectories
    
    for sample in tqdm(sample_summaries, desc="Scoring trajectories"):
        ground_truth = sample["answer"]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        
        for trajectory in sample["all_trajectories"]:
            # Compute final F1 for this trajectory
            predict = trajectory.get("predict", "")
            if predict:
                if isinstance(predict, str):
                    predict = [predict]
                trajectory_f1 = F1_score(predict, ground_truth)
            else:
                trajectory_f1 = 0.0
            
            # Score each node in the path
            path_ids = trajectory["path"]
            step_scores = score_trajectory_path(
                path_ids=path_ids,
                trajectory_tree=trajectory_tree,
                ground_truth=ground_truth,
                trajectory_final_f1=trajectory_f1,
                distance_api_url=distance_api_url,
                max_distance=max_distance
            )
            
            # Record the scores for each node
            for step_score in step_scores:
                node_id = step_score["trajectory_id"]
                node_trajectory_scores[node_id].append({
                    "trajectory_f1": trajectory_f1,
                    "step_score": step_score["total_score"],
                    "progress_score": step_score["progress_score"],
                    "format_score": step_score["format_score"],
                })
    
    # Step 2: Build training data - Each parent node and all its children
    print("\nBuilding training data...")
    training_data = []

    # Collect all nodes with children (non-leaf nodes)
    parent_nodes = [node_id for node_id, node in trajectory_tree.items() if node.get("children")]

    for parent_id in tqdm(parent_nodes, desc="Processing parent nodes"):
        parent_node = trajectory_tree[parent_id]
        children_ids = parent_node.get("children", [])
        
        if not children_ids:
            continue
        
        # Collect outputs and scores for all children
        completions = []
        rewards = []
        children_metadata = []
        
        for child_id in children_ids:
            if child_id not in trajectory_tree:
                continue
            
            child_node = trajectory_tree[child_id]
            # Child node's output is the completion
            output = child_node.get("output", "")
            query_type = child_node.get("query_type", "")
            # Get average/max score for this child node (from all trajectories passing through it)
            if child_id in node_trajectory_scores:
                scores = node_trajectory_scores[child_id]
                # Take max value as the score for this node
                final_score = max(s["step_score"] for s in scores)
                # final_score = sum(s["step_score"] for s in scores) / len(scores)
            else:
                final_score = 0.0
            
            if query_type == "node":
                completions.append(output + "</node>")
            elif query_type == "sparql":
                completions.append(output + "</SPARQL>")
            elif '//boxed{' in output:
                completions.append(output + "<|im_end|>")
            else:
                completions.append(output)
            
            rewards.append(round(final_score, 4))
            children_metadata.append({
                "child_id": child_id,
                "query_type": child_node.get("query_type"),
                "is_complete": child_node.get("is_complete", False),
            })
        
        if not completions:
            continue
        
        # All children should share the same prompt
        first_child_id = children_ids[0]
        if first_child_id in trajectory_tree:
            input_prompt = trajectory_tree[first_child_id].get("prompt", "")
        else:
            continue
        
        # Build training sample
        training_sample = {
            "prompt": input_prompt,
            "completions": completions,
            "rewards": rewards,
            "metadata": {
                "sample_index": parent_node.get("sample_index"),
                "parent_trajectory_id": parent_id,
                "parent_step": parent_node.get("step", 0),
                "children_ids": children_ids,
                "children_metadata": children_metadata,
            }
        }
        
        training_data.append(training_sample)
    
    with open(output_path.replace("k_sampling_scored_training_data", "k_sampling_all_data"), 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"Total generate {len(training_data)} Training Samples")
    print(f"{len(set(s['metadata']['sample_index'] for s in training_data))} Unique Questions")
    
    # Print some stats
    total_completions = sum(len(s["completions"]) for s in training_data)
    avg_completions = total_completions / len(training_data) if training_data else 0
    print(f"Average Node has {avg_completions:.2f} candidates.")
    
    # Filter final training data: keep only if scores differ
    num = 0
    train_data = []
    for result in training_data:
        if len(list(set(result["rewards"]))) != 1:
            train_data.append(
                {
                    "prompt": result["prompt"],
                    "completions": result["completions"],
                    "rewards": result["rewards"]
                }
            )
            num += 1
    print(f"Number of samples with varying rewards: {num} out of {len(training_data)}. percentage: {num/len(training_data)*100:.2f}%")
    save_json(output_path, train_data)

    return training_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Manager passes specific args, not config json
    parser.add_argument("--trajectory_tree_path", type=str, required=True)
    parser.add_argument("--sample_summaries_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--distance_api", type=str, default=os.getenv("DISTANCE_API_URL", "http://localhost:5501/entity_distance"))
    
    args = parser.parse_args()

    # Process files
    process_k_beam_results(
        trajectory_tree_path=args.trajectory_tree_path,
        sample_summaries_path=args.sample_summaries_path,
        output_path=args.output_path,
        distance_api_url=args.distance_api,
        max_distance=3
    )