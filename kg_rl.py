import re
import string
from typing import List
from collections import defaultdict
import random
import requests
from enum import Enum, auto
from typing import List, Set, Tuple, Dict
import os
import ast
from rdflib.plugins.sparql.parser import parseQuery

# ====================================

def normalize_boxed_answer(text):
    # Extract content within \boxed{}
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        content = match.group(1)
        # Split by comma, strip whitespace, and convert to lowercase
        items = [item.strip().lower() for item in content.split(',')]
        return items
    else:
        return []

# Calculate F1 score for answers
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    try:
        s = s.lower()
        exclude = set(string.punctuation)
        s = "".join(char for char in s if char not in exclude)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # remove <pad> token:
        s = re.sub(r"\b(<pad>)\b", " ", s)
        s = " ".join(s.split())
        return s
    except Exception as e:
        return ""

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0 or answer is None or len(answer) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1

    # Cap matched to not exceed the length of prediction or answer
    matched = min(matched, len(prediction), len(answer))
    
    precision = matched / len(prediction)
    recall = matched / len(answer)
    
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def evaluate_single_sample(prediction, ground_truth):
    if not isinstance(prediction, list):
        prediction = prediction.split(", ")
    
    # Use existing functions for processing
    f1_score, precision, recall = eval_f1(prediction, ground_truth)
    return {
        'F1': f1_score,
        'Precision': precision,
        'Recall': recall
    }

# Calculate answer correctness score via F1
def F1_score(prediction, golden_answers):
    """Calculate F1 score - assumes golden_answers is always a list"""
    if golden_answers is None or len(golden_answers) == 0:
        return 0.0
    
    # Normalize prediction
    if isinstance(prediction, str) and "\\boxed{" in prediction:
        normalized_prediction = normalize_boxed_answer(prediction)
    elif isinstance(prediction, str):
        normalized_prediction = [item.strip().lower() for item in prediction.split(',')]
    else:
        normalized_prediction = [normalize(ans) for ans in prediction]
    
    # Normalize answers (already a list)
    normalized_answers = [normalize(ans) for ans in golden_answers]
    
    prediction_score = evaluate_single_sample(normalized_prediction, normalized_answers)
    score = prediction_score['F1']
    return score

def is_failure(info: str) -> bool:
    """Determine if query failed"""
    if isinstance(info, list):
        return False
    return info == "No results found for the given SPARQL query.\nPlease try generating a different SPARQL query."
    
def check_xml_enter(text: str):
    tags = ["think", "node", "relation", "SPARQL", "information", "answer"]
    errors = []

    for tag in tags:
        # Match each <tag> ... </tag> block
        pattern = re.compile(rf"<{tag}>\n(.*?)</{tag}>", re.DOTALL)
        for match in pattern.finditer(text):
            block = match.group(0)
            inner = match.group(1)
            if not block.startswith(f"<{tag}>\n"):
                errors.append(f"<{tag}> start tag not followed by newline")
            if block.startswith(f"<{tag}>\n\n"):
                errors.append(f"<{tag}> start tag followed by extra newlines")
            if inner.endswith("\n"):
                errors.append(f"</{tag}> preceded by extra newline")

    return len(errors) == 0, errors


def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        content = match.group(1).strip()
        if content:
            return [item.strip() for item in content.split(',')]
        else:
            return []
    return None

def validate_xml_tags(text):
    tags_to_check = ["think", "relation", "information", "SPARQL", "answer", "node"]
    
    for tag in tags_to_check:
        # 使用不区分大小写的匹配
        opening_pattern = f"<{tag}>\n"
        closing_pattern = f"</{tag}>"
        
        opening_count = len(re.findall(opening_pattern, text, re.IGNORECASE))
        closing_count = len(re.findall(closing_pattern, text, re.IGNORECASE))
        # print(f"Tag: {tag}, Opening: {opening_count}, Closing: {closing_count}")
        if opening_count != closing_count:
            return False
    return True

def check_sparql_syntax(query: str):
    try:
        parseQuery(query)
        return True
    except Exception as e:
        return False