from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from torch.nn.utils.rnn import pad_sequence
import student.utils as utils
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