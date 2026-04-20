import argparse
import random

from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
import torch

from student.drgrpo_grader import pit_reward_fn
from student.evaluate import evaluate
from student.sec_7.dataloader import get_gsm_adversarial_dataloaders
from student.sec_7.defaults import MODEL_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    sampling_temperature: float = 0.7
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    test_dataloader = get_gsm_adversarial_dataloaders(
        dataset_path=args.dataset_path,
        n_prompts_per_rollout_batch=2,
        reduce=False
    )

    policy_model_name = MODEL_NAME
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map="mps",
    )

    eval_prompts = []
    eval_gts = []
    for batch in test_dataloader:
        eval_prompts.extend(batch["prompts"])
        eval_gts.extend(batch["ground_truths"])

    acc, reward = evaluate(policy, eval_prompts, eval_gts,
                           sampling_temperature=sampling_temperature,
                           sampling_max_tokens=sampling_max_tokens,
                           sampling_min_tokens=sampling_min_tokens,
                           stop_tokens=['</answer>'],
                           reward_fn=pit_reward_fn,
                           verbose=True
                           )

    print("acc", acc)