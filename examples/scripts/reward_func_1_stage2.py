from filelock import FileLock
import torch
import json
import math
import re
from typing import Dict
import statistics  # 导入用于计算统计量的库
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
    # 使用正则表达式查找<think>标签内的token数量
    match = re.search(r'I will answer the question with\s*(\d+)\s*tokens', text)
    if match:
        # 如果找到匹配，返回第一个捕获组中的数字，即token的数量
        return int(match.group(1))
    else:
        # 如果没有找到匹配，返回None或其它你希望的值
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

    mean_length = sum_length / count if count > 0 else 0
    median_length = statistics.median(lengths) if lengths else 0  # 如果lengths为空，则中位数为0

    # input_file = '/home/xy/median_length_process.json'
    input_file = '/home/xy/median_length_process_dict.json'
    # input_file_0 = '/home/xy/mean_length_process.json'
    lock = FileLock(input_file + '.lock')  # 加一把文件锁！


    # if median_length != 0:
    #     if median_length < tokens:
    #         prompt_key = get_substring_up_to_last_gt(prompt_0)
    #         with open(input_file, "r", encoding="utf-8") as f:
    #             q2len = json.load(f)
    #
    #         q2len[prompt_key] = median_length
    #         with open(input_file, "w", encoding="utf-8") as f:
    #             json.dump(q2len, f, ensure_ascii=False, indent=2)
    #             f.flush()
    #             os.fsync(f.fileno())

    if (tokens == 2048 and median_length != 0) or (median_length != 0 and median_length < tokens):
    # if tokens == 2048 and median_length != 0:
        prompt_key = get_substring_up_to_last_gt(prompt_0)
        with lock:
            # 先安全读取
            with open(input_file, "r", encoding="utf-8") as f:
                q2len = json.load(f)

            # 更新
            q2len[prompt_key] = median_length

            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(q2len, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())


    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt) - 51:]
        # print(f"response :{response} NNNNNNNNNNN")
        # think_length = extract_think_length(response, tokenizer)

        acc = accuracy_reward(response, answer)
        format = format_reward(response)
        length = len(tokenizer.encode(response))

        # 计算长度奖励
        sigma = 120

        if acc > 0:  # 只有答案正确时才计算长度奖励
            if tokens != 2048:
                # 使用预设的目标长度
                len_reward = math.exp(-((length - tokens) ** 2) / (2 * (sigma ** 2)))

                # len_reward = max(1.0 - abs(length - tokens) * 0.002, 0.0)
            else:
                # 探索阶段
                len_reward = math.exp(-((length - median_length) ** 2) / (2 * (sigma ** 2)))

                # len_reward = max(1.0 - abs(length - median_length) * 0.002, 0.0)
        else:
            len_reward = 0.0  # 错误答案不给长度奖励

        # len_reward = math.exp(-((length - tokens) ** 2) / (2 * (sigma ** 2))) if tokens != 2048 else 0.0

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
    # 找到最后一个'>'字符的位置
    index = s.rfind('>')
    if index != -1:
        # 截取到最后一个'>'字符（包含该字符）
        return s[:index + 1]
    else:
        # 如果没有找到'>'，返回原始字符串或者空字符串
        return s
