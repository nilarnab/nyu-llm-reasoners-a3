from typing import Callable

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from torch.nn.utils.rnn import pad_sequence
import student.utils as utils
from tests.conftest import rollout_responses, reward_fn, repeated_ground_truths


# def run_compute_group_normalized_rewards_util(
#     reward_fn: Callable,
#     rollout_responses: list[str],
#     repeated_ground_truths: list[str],
#     group_size: int,
#     advantage_eps: float,
#     normalize_by_std: bool,
# ) -> tuple[torch.Tensor, dict[str, float]]:
#     """
#     Compute rewards for each group of rollout responses,
#     normalized by the group size.
#
#     For more on GRPO, see:
#         DeepSeekMath: https://arxiv.org/abs/2402.03300
#         DeepSeek-R1: https://arxiv.org/abs/2501.12948
#
#     Args:
#         reward_fn: Callable[[str, str], dict[str, float]],
#             scores the rollout responses against the ground truths,
#             producing a dict with keys
#             "reward", "format_reward", and "answer_reward".
#         rollout_responses: list[str], rollouts from the policy.
#             The length of this list is
#             `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
#         repeated_ground_truths: list[str], the ground truths for the examples.
#             The length of this list is `rollout_batch_size`,
#             because the ground truth for each example is repeated `group_size` times.
#         group_size: int, number of rollouts per group.
#         advantage_eps: float, epsilon to avoid division by zero
#             during group normalization.
#         normalize_by_std: bool, whether to normalize the rewards by
#             std(rewards).
#
#     Returns:
#         tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
#             torch.Tensor of shape (rollout_batch_size,):
#                 group-normalized rewards for each rollout response.
#             torch.Tensor of shape (rollout_batch_size,):
#                 raw rewards for each rollout response.
#             dict[str, float]: metadata for the rewards of the rollout batch.
#                 You may choose what you wish to log here
#                 (some statistics of the rewards, etc.).
#     """
#     rewards = []
#     for i, rollout_response in enumerate(rollout_responses):
#         reward_outp = reward_fn(rollout_response, repeated_ground_truths[i])
#         reward = reward_outp['reward']
#         rewards.append(reward)
#
#     mean_rwd = np.mean(rewards)
#     std_rwd = np.std(rewards)
#
#     reward_tensor = torch.tensor(rewards)
#     Advantage = (reward_tensor - mean_rwd) / (std_rwd + advantage_eps)
#
#     return Advantage, reward_tensor, {"reward_total": sum(rewards), "max_reward": max(rewards), "mean_reward": sum(rewards)/len(rewards), "min_reward": min(rewards)}


def run_compute_group_normalized_rewards_util(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    # Compute raw rewards for all responses
    rewards = []
    for i, rollout_response in enumerate(rollout_responses):  # fix: no ()
        reward_outp = reward_fn(rollout_response, repeated_ground_truths[i])
        rewards.append(reward_outp["reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    # Group-normalize: reshape into (n_groups, group_size)
    n_groups = len(rollout_responses) // group_size
    grouped = raw_rewards.view(n_groups, group_size)  # (n_groups, group_size)

    group_means = grouped.mean(dim=1, keepdim=True)   # (n_groups, 1)
    advantages = grouped - group_means                 # subtract group mean

    if normalize_by_std:
        group_stds = grouped.std(dim=1, keepdim=True)  # (n_groups, 1)
        advantages = advantages / (group_stds + advantage_eps)

    advantages = advantages.view(-1)  # flatten back to (rollout_batch_size,)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return advantages, raw_rewards, metadata

