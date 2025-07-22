import torch
import json
import math
import re
import statistics
from numpy import dtype
import numpy as np
import os

def format_reward(r):
    """
    Calculates the format reward based on the presence and correctness of tags and content.

    Args:
        r: The input string representing the model's response.

    Returns:
        A reward value (1.0 for correct format, 0.0 otherwise).
    """
    pattern = r"^<think>.*?</think>.*\\boxed\{.*?\}.*$"  # Modified pattern

    think_count = r.count("<think>") + r.count("</think>")

    if (re.match(pattern, r, re.DOTALL) and think_count == 2):
        return 1.0
    else:
        return 0.0

def accuracy_reward(response, answer):
    match = re.search(r'\\boxed{([^}]*)}', response)
    if match:
        return 1.0 if match.group(1) == answer else 0.0
    else:
        return 0.0

def reward_func(queries, prompts, labels, tokenizer, **kwargs):
    rewards = []
    acc_rewards = []
    format_rewards = []
    sum_length = 0
    min_length = 0
    count = 0
    lengths = []

    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt) - 7:]
        length = len(tokenizer.encode(response))
    
        acc = accuracy_reward(response, answer)
    
        if acc:
            sum_length += len(tokenizer.encode(response))
            count += 1
            lengths.append(length)
        else:
            min_length = length if length < min_length else min_length
    
    mean_length = sum_length / count if count > 0 else 0
    median_length = statistics.median(lengths) if lengths else 0
    
    if lengths:
        lower_q = np.percentile(lengths, 30)
        upper_q = np.percentile(lengths, 70)
    else:
        lower_q, upper_q = 0, 0

    current_prompt = prompts[0]
    log_data = {
        "prompt": current_prompt,
        "median_length": median_length
    }
    
    log_data_0 = {
        "prompt": current_prompt,
        "median_length": mean_length
    }

    log_data_1 = {
        "prompt": current_prompt,
        "median_length": min_length
    }

    log_file = "all_mapping_file"

    with open(log_file, "a+", encoding="utf-8") as f:
        f.seek(0)
        first_char = f.read(1)
        f.seek(0, os.SEEK_END)
        if not first_char:
            f.write(json.dumps(log_data, ensure_ascii=False))
        else:
            f.write('\n' + json.dumps(log_data, ensure_ascii=False))
    
    # with open(log_file_0, "a+", encoding="utf-8") as f:
    #     f.seek(0)
    #     first_char = f.read(1)
    #     f.seek(0, os.SEEK_END)
    #     if not first_char:
    #         f.write(json.dumps(log_data_0, ensure_ascii=False))
    #     else:
    #         f.write('\n' + json.dumps(log_data_0, ensure_ascii=False))

    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt) - 7:]

        acc = accuracy_reward(response, answer)
        format = format_reward(response)
        length = len(tokenizer.encode(response))
        len_reward = 0.0
        
        if acc and lower_q < upper_q:
            if lower_q <= length <= upper_q:
                len_reward = 1.0
            else:
                margin = min(abs(length - lower_q), abs(length - upper_q))
                len_reward = max(0.0, 1.0 - margin / 100.0)
        else:
            len_reward = 0.0

        rewards.append(1.0 * acc + 0.7 * len_reward)
        rewards.append(acc)

        acc_rewards.append(acc)
        format_rewards.append(format)

    acc_rewards = torch.tensor(acc_rewards, dtype=torch.float)
    format_rewards = torch.tensor(format_rewards, dtype=torch.float)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float)

    return {
        "rewards": rewards_tensor,  # Rewards for advantage calculation
        "scores": rewards_tensor,  # Scores for dynamic filtering (0-1 reward)
        "extra_logs": {"dummy_scores": rewards_tensor},  # Additional logging info for wandb
        "acc_rewards": acc_rewards,  # Accuracy rewards
        "format_rewards": format_rewards,  # Format rewards
    }
