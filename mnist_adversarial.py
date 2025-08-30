import verifiers as vf
from datasets import load_dataset, Dataset

import re
import random

SYSTEM_PROMPT = """
You are a strict image judge for MNIST grayscale inputs.
You will receive a single flattened array of 784 grayscale values (row-major 28×28), each in [0,255].
Your job: (1) decide whether the input is adversarial or normal, and (2) identify the digit class X in {0,…,9}.
Adversarial = any input that appears perturbed to mislead a classifier; Normal = consistent MNIST-like digit with plausible strokes and contrast.
You must output only one line in the exact format: \\boxed{adversarial_X} or \\boxed{normal_X} (no other text, symbols, or whitespace).
"""

BOXED_RE = re.compile(r'^\s*\\boxed\{(adversarial|normal)_(\d)\}\s*$')

def load_environment(dataset_split: str = "test", size: int = 50, **kwargs) -> vf.Environment:
    '''
    Loads a custom environment.
    '''
    normal_dataset, adversarial_dataset = __load_dataset("wambosec/adversarial-mnist", dataset_split=dataset_split)
    eval_data = __build_eval_set(size, normal_dataset, adversarial_dataset)
    eval_data = [
        {
            "question": str(list(x.values())[1:-1]),
            "answer": f"\\boxed{{adversarial_{int(x['label'])}}}" if x["is_adversarial"] else f"\\boxed{{normal_{int(x['label'])}}}",
            "info": {},
            "task": "mnist_adversarial",
        }
        for x in eval_data
    ]

    eval_data = Dataset.from_list(eval_data)
    
    def __calculate_reward(completion, answer, **kwargs) -> float:
        '''
        Calculates the reward for a given answer and ground truth.
        +0.5 for correct status, +0.5 for correct label.
        '''
        completion = completion[0]["content"]
        parsed_completion = __parse_answer(completion)
        parsed_answer = __parse_answer(answer)
        reward = 0
        if parsed_answer["is_adversarial"] == parsed_completion["is_adversarial"]:
            reward += 0.5
        if parsed_answer["label"] == parsed_completion["label"]:
            reward += 0.5
        return reward

    rubric = vf.Rubric(
        funcs = [
            __calculate_reward,
        ],
        weights = [1.0],
    )
    
    vf_env = vf.SingleTurnEnv(
        dataset = eval_data,
        system_prompt = SYSTEM_PROMPT,
        parser = __parse_answer,
        rubric = rubric,
        **kwargs
    )
    
    return vf_env

def __load_dataset(dataset_name: str, dataset_split: str = "train") -> Dataset:
    '''
    Loads a custom dataset.
    '''
    # We focus on the test set for now, since we are only evaluating
    dataset = load_dataset(dataset_name, split=dataset_split)
    normal_dataset, adversarial_dataset = __split_dataset(dataset)
    return normal_dataset, adversarial_dataset

def __split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    '''
    Splits a dataset into a normal and adversarial set.
    '''
    normal_dataset = dataset.filter(lambda x: x["is_adversarial"] == 0)
    adversarial_dataset = dataset.filter(lambda x: x["is_adversarial"] == 1)
    return normal_dataset, adversarial_dataset

def __parse_answer(answer: str) -> tuple[bool, int]:
    """
    Returns: {"is_adversarial": bool, "label": int}
    Raises: ValueError on malformed input.
    """
    answer = answer.strip()
    m = BOXED_RE.match(answer)
    if not m:
        raise ValueError("Invalid format. Expected \\boxed{adversarial_X} or \\boxed{normal_X} with X in range(10).")
    status, digit = m.group(1), int(m.group(2))
    return {
        "is_adversarial": status == 1,
        "label": digit,
    }

def __build_eval_set(n: int, normal_dataset: Dataset, adversarial_dataset: Dataset) -> list:
    '''
    Builds an evaluation set from the normal and adversarial datasets.
    '''
    rng = random.Random(777)
    eval_set = []
    normal_indices = rng.sample(range(len(normal_dataset)), n)
    adversarial_indices = rng.sample(range(len(adversarial_dataset)), n)
    
    for i in range(n):
        normal_sample = normal_dataset[normal_indices[i]]
        adversarial_sample = adversarial_dataset[adversarial_indices[i]]
        eval_set.append(normal_sample)
        eval_set.append(adversarial_sample)
    random.shuffle(eval_set)
    return eval_set
