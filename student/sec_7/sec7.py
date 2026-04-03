from typing import Callable

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from torch.nn.utils.rnn import pad_sequence
import student.utils as utils
from tests.conftest import rollout_responses, reward_fn, repeated_ground_truths


def run_compute_group_normalized_rewards_util(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses,
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]],
            scores the rollout responses against the ground truths,
            producing a dict with keys
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy.
            The length of this list is
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples.
            The length of this list is `rollout_batch_size`,
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,):
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,):
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    rewards = []
    for i, rollout_response in enumerate(rollout_responses):
        reward_outp = reward_fn(rollout_response, repeated_ground_truths[i])
        reward = reward_outp['reward']
        rewards.append(reward)

    # mean_rwd = np.mean(rewards)
    # std_rwd = np.std(rewards)

    reward_tensor = torch.tensor(rewards)

    group_count = len(rollout_responses) // group_size
    grouped_reward_tensor = reward_tensor.view(group_count, group_size)

    mean_tensor = grouped_reward_tensor.mean(dim=1, keepdim=True)
    Advantage = grouped_reward_tensor - mean_tensor

    if normalize_by_std:
        group_stds = grouped_reward_tensor.std(dim=1, keepdim=True)
        Advantage = Advantage / (group_stds + advantage_eps)

    Advantage = Advantage.view(-1)


    return Advantage, reward_tensor, {"reward_total": sum(rewards), "max_reward": max(rewards), "mean_reward": sum(rewards)/len(rewards), "min_reward": min(rewards)}


def run_compute_naive_policy_gradient_loss_util(
        raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1):
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            the policy gradient per-token loss.
    """

    npgl = - raw_rewards_or_advantages * policy_log_probs

    return npgl

def run_compute_grpo_clip_loss_util(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1):
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length):
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length):
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss
                (used to compute clip fraction).
    """

    ratio = (policy_log_probs / old_log_probs)
    res = min(
        ratio * advantages, max(min(ratio, 1 + cliprange), 1 - cliprange)
    )

    return res, {"clip_loss": res, "ratio": ratio}



