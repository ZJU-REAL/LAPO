import torch
import json
import math
import re
import statistics  # 导入用于计算统计量的库
from numpy import dtype
import numpy as np
import os


def extract_think_length(response, tokenizer):
    """从response中提取 <think> 标签内的文本长度（token数）。"""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        # 使用tokenizer对文本进行编码并计算token数量
        return len(tokenizer.encode(think_content)) - 40
    return 0

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

def extract_predicted_tokens(text):
    # 使用正则表达式查找<think>标签内的token数量
    match = re.search(r'think for\s*(\d+)\s*tokens', text)
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


def reward_func(queries, prompts, labels, tokenizer, **kwargs):
    # queries is prompts + responses
    # labels is answers
    # print(len(queries))
    # print(prompts)
    rewards = []
    acc_rewards = []
    format_rewards = []
    # think_lengths = [extract_think_length(r, tokenizer) for r in responses]
    sum_length = 0
    max_length = 0
    count = 0
    lengths = []
    all_lengths = []

    # for query, prompt, answer in zip(queries, prompts, labels):
    #     response = query[len(prompt) - 7:]
    #     # print(f"response :{response} NNNNNNNNNNN")
    #     length = len(tokenizer.encode(response))
    #
    #     acc = accuracy_reward(response, answer)
    #     # format = format_reward(response)
    #
    #     if acc:
    #         sum_length += len(tokenizer.encode(response))
    #         count += 1
    #         lengths.append(length)
    #         all_lengths.append(length)
    #     else:
    #         max_length = length if length > max_length else max_length
    #         all_lengths.append(0)
    #
    # # print(lengths)
    # mean_length = sum_length / count if count > 0 else 0
    # median_length = statistics.median(lengths) if lengths else 0  # 如果lengths为空，则中位数为0
    #
    # if lengths:
    #     # 计算20%/80%分位数作为“合理length区间”
    #     lower_q = np.percentile(lengths, 30)
    #     upper_q = np.percentile(lengths, 70)
    #     # 你也可以调整为30%-70%或者使用众数±window
    # else:
    #     lower_q, upper_q = 0, 0

    # 获取当前组的prompt（因为所有prompt都相同，所以取第一个即可）
    # current_prompt = prompts[0]
    #
    # # 构建要写入的数据：每个 prompt + 当前 batch 的 median_length
    # log_data = {
    #     "prompt": current_prompt,
    #     "median_length": median_length
    # }
    #
    # log_data_0 = {
    #     "prompt": current_prompt,
    #     "median_length": mean_length
    # }

    # log_file = "/home/xy/median_length-2.json"
    # log_file_0 = "/home/xy/mean_length-2.json"

    # with open(log_file, "a+", encoding="utf-8") as f:
    #     # 移动到文件开头检查是否为空
    #     f.seek(0)
    #     first_char = f.read(1)
    #     f.seek(0, os.SEEK_END)  # 返回文件末尾以便追加数据
    #     if not first_char:  # 如果文件为空，则不添加换行符
    #         f.write(json.dumps(log_data, ensure_ascii=False))
    #     else:
    #         f.write('\n' + json.dumps(log_data, ensure_ascii=False))
    #
    # with open(log_file_0, "a+", encoding="utf-8") as f:
    #     # 移动到文件开头检查是否为空
    #     f.seek(0)
    #     first_char = f.read(1)
    #     f.seek(0, os.SEEK_END)  # 返回文件末尾以便追加数据
    #     if not first_char:  # 如果文件为空，则不添加换行符
    #         f.write(json.dumps(log_data_0, ensure_ascii=False))
    #     else:
    #         f.write('\n' + json.dumps(log_data_0, ensure_ascii=False))

    for query, prompt, answer in zip(queries, prompts, labels):
        response = query[len(prompt) - 7:]
        # print(f"response :{response} NNNNNNNNNNN")
        # think_length = extract_think_length(response, tokenizer)

        acc = accuracy_reward(response, answer)
        format = format_reward(response)
        length = len(tokenizer.encode(response))
        predict = 0.0
        len_reward = 0.0

        base_sigma = 20.0
        scale_factor = 0.4

        # if median_length != 0 and acc:
        #     # predict = max(1.0 - abs(length - median_length) * 0.002, 0.0)
        #     sigma = 120
        #
        #     predict = math.exp(-((length - median_length) ** 2) / (2 * (sigma ** 2)))
        # else:
        #     sigma = 120
        #
        #     predict = math.exp(-((length - max_length) ** 2) / (2 * (sigma ** 2))) if max_length > 0 else 0
        #     # predict = max(1.0 - abs(length - max_length) * 0.002, 0.0) if max_length > 0 else 0.0
        #
        #     # predict = random.random()


        # if acc and lower_q < upper_q:
        #     if lower_q <= length <= upper_q:
        #         len_reward = 1.0
        #     else:
        #         # 离区间越远奖励越低，超过±60会变为零
        #         margin = min(abs(length - lower_q), abs(length - upper_q))
        #         len_reward = max(0.0, 1.0 - margin / 100.0)
        # else:
        #     len_reward = 0.0

        # rewards.append(1.0 * acc + 0.7 * len_reward)
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