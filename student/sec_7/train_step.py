from typing import Callable, Literal

import numpy as np
import torch
import wandb
from torch import Tensor
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
import student.utils as utils
from student.evaluate import evaluate
from student.sec_4.run_experiment import load_policy_into_vllm_instance
from student.sec_7.sec7 import run_compute_policy_gradient_loss_util, run_masked_mean_util
from tests.conftest import rollout_responses, reward_fn, repeated_ground_truths


def run_grpo_microbatch_train_step_util(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length):
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio.
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            the policy gradient loss and its metadata.
    """

    print(">>>> GRADIENT CLIP STEP", gradient_accumulation_steps, "loss type", loss_type)

    loss_per_token, metadata = run_compute_policy_gradient_loss_util(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    # loss = loss_per_token.mean()

    print("loss initial 1 shape", loss_per_token.shape)
    masked_loss = run_masked_mean_util(loss_per_token, response_mask)
    masked_loss = masked_loss.mean()
    masked_loss = masked_loss / gradient_accumulation_steps

    masked_loss.backward()


    return masked_loss, metadata
    pass


def run_grpo_training(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=utils.DEVICE,
        # n_grpo_steps=3,
        # grad_accum_steps=32,
        eval_after=20,
        run_name=None,
):
    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    best_acc = -1
    print("Running eval once first")
    load_policy_into_vllm_instance(model_train, eval_vllm_model)
    acc = evaluate(eval_vllm_model, eval_prompts, eval_gts)
    wandb.log({"eval/accuracy": acc}, step=step_count)
    print('EVAL', acc)

    for step in range(n_grpo_steps):
        questions_batch = next(iter(dataloader))

        old_policy_model = model_train.eval()
        old_policy_model = old_policy_model.to(device)

        rollout_responses = []
        repeated_ground_truths = []



if __name__ == '__main__':
    policy_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # exact model for GRPO
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)

    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 16
    group_size: int = 8
    sampling_temperature: float = 0.7
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 64  # On-policy
    gradient_accumulation_steps: int = 128
    gpu_memory_utilization: float = 0.8
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )









