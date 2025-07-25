from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json

system_prompt = "You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here ."

input_file = 'mapping_path'

def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_prompt} User: {data[input_key]} Please think step by step and output the final answer within " + "\\boxed{}. Assistant: <think>"
        #
        # length = 0.0
        # with open(input_file, 'r', encoding='utf-8') as f:
        #     q2len = json.load(f)
        #
        #     length = q2len[prompt]
        #
        # if length == 0.0:
        #     length = 4096
        # else:
        #     length = int(length)
        #
        # prompt = prompt + f" I will answer the question with {length} tokens."

        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
