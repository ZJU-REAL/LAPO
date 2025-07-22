from filelock import FileLock
import torch
import json
import math
import re
from typing import Dict
import statistics
import random

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
    pattern = r".*?</think>.*\\boxed\{.*?\}.*$"  # Modified pattern

    think_count = r.count("</think>")

    if (re.match(pattern, r, re.DOTALL) and think_count == 1):
        return 1.0
    else:
        return 0.0

def extract_predicted_tokens(text):
    match = re.search(r'I will answer the question with\s*(\d+)\s*tokens', text)
    if match:
        return int(match.group(1))
    else:
        return 0


def accuracy_reward(response, answer):
    match = re.search(r'\\boxed{([^}]*)}', response)
    if match:
        return 1.0 if match.group(1) == answer else 0.0
    else:
        return 0.0


def reward_func(queries, prompts, labels, tokenizer):
    # queries is prompts + responses
    # labels is answers
    rewards = []
    acc_rewards = []
    format_rewards = []
    sum_length = 0.0
    count = 0
    lengths = []
    prompt_0 = prompts[0]

    tokens = extract_predicted_tokens(prompts[0])

    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt):]
        # print(f"response :{response} NNNNNNNNNNN")
        length = len(tokenizer.encode(response))

        acc = accuracy_reward(response, answer)
        # format = format_reward(response)

        if acc:
            sum_length += len(tokenizer.encode(response))
            count += 1
            lengths.append(length)

    median_length = statistics.median(lengths) if lengths else 0
    mean_length = sum_length / count if count > 0 else 0
    min_length = min(lengths) if lengths else 0

    input_file = 'stage1_file_path'
    lock = FileLock(input_file + '.lock')

    if (tokens == 4096 and median_length != 0) or (median_length != 0 and median_length < tokens):
        prompt_key = get_substring_up_to_last_gt(prompt_0)
        prompt_key = prompt_0
        with lock:
            with open(input_file, "r", encoding="utf-8") as f:
                q2len = json.load(f)

            q2len[prompt_key] = median_length

            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(q2len, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt) - 51:]

        acc = accuracy_reward(response, answer)
        format = format_reward(response)
        length = len(tokenizer.encode(response))

        sigma = 120
        if acc > 0:
            if tokens != 4096:
                len_reward = math.exp(-((length - tokens) ** 2) / (2 * (sigma ** 2)))
            else:
                len_reward = math.exp(-((length - median_length) ** 2) / (2 * (sigma ** 2)))
        else:
            len_reward = 0.0

        rewards.append(1.0 * acc + 0.8 * len_reward)

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

def get_substring_up_to_last_gt(s):
    index = s.rfind('>')
    if index != -1:
        return s[:index + 1]
    else:
        return s
