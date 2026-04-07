from typing import Callable, Literal

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from torch import Tensor
from torch.utils.data import DataLoader
from torchgen.executorch.api.et_cpp import return_type
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
import student.utils as utils
from student.evaluate import evaluate
from student.sec_4.run_experiment import load_policy_into_vllm_instance, init_vllm, VLLM_DEVICE
from student.sec_4.sec4 import run_get_response_log_probs_util
from student.sec_7.sec7 import run_compute_policy_gradient_loss_util, run_masked_mean_util, \
    run_compute_group_normalized_rewards_util
from tests.conftest import rollout_responses, reward_fn, repeated_ground_truths



def get_countdown_dataloaders(dataset_path, n_prompts_per_rollout_batch, seed=42):
    def extract_question(text: str) -> str:
        if "User:" in text:
            text = text.split("User:")[-1]

        if "Show your work" in text:
            text = text.split("Show your work")[0]

        return text.strip()

    def format_prompt(question: str) -> str:
        return f"""Answer the following problem. Explain your reasoning step by step. When you are finished, give your answer in this format: <answer>(your answer)</answer>.

    Problem
    {question}

    Your solution should include a series of steps "Step X:" where each step is a mathematical operation and the final step ultimately leads to the target number or it should be a single equation that results in the target.

    Give your answer in the following format:
    <answer>
    (your answer)
    </answer>

    Where "(your answer)" is the list of steps to reach the target number or it should be a single equation that results in the target.

    For example:
    If the list of numbers was [1, 2, 3] and the target was 1, you could write:

    <answer>
    Step 1: 1 + 2 = 3
    Step 2: 3 / 3 = 1
    </answer>

    or

    <answer>
    (1 + 2) / 3
    </answer>

    Let's think step by step."""
    
    dataset = load_from_disk(dataset_path)

    def collate_fn(batch):
        prompts = []

        for item in batch:
            msgs = item["prompt"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")

            question = extract_question(user_msg)
            prompt = format_prompt(question)

            prompts.append(prompt)

        return {
            "prompts": prompts,
            "ground_truths": [str(item["target"]) for item in batch],
        }

    train_loader = DataLoader(
        dataset["train"],
        batch_size=n_prompts_per_rollout_batch,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        dataset["test"],  # update based on dataset.keys()
        batch_size=n_prompts_per_rollout_batch,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


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
        eval_after=5,
        run_name=None,

        # GRPO PARAMETERS
        n_grpo_steps: int = 200,
        advantage_eps: float = 1e-6,
        rollout_batch_size: int = 16,
        group_size: int = 8,
        sampling_temperature: float = 0.7,
        sampling_min_tokens: int = 4,
        sampling_max_tokens: int = 1024,
        epochs_per_rollout_batch: int = 1,  # On-policy
        train_batch_size: int = 64,  # On-policy
        gradient_accumulation_steps: int = 128,
        loss_type: Literal[
            "no_baseline",
            "reinforce_with_baseline",
            "grpo_clip",
        ] = "reinforce_with_baseline",
        use_std_normalization: bool = True,
        grpo_clip = 1.0,
):
    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    best_acc = -1
    print("Running eval once first")
    load_policy_into_vllm_instance(model_train, eval_vllm_model)
    acc, reward = evaluate(eval_vllm_model, eval_prompts, eval_gts)
    wandb.log({"eval/accuracy": acc}, step=step_count)
    print('EVAL', acc)

    for step in range(n_grpo_steps):
        questions_batch = next(iter(dataloader))

        old_policy_model = model_train.eval()
        old_policy_model = old_policy_model.to(device)
        load_policy_into_vllm_instance(old_policy_model, eval_vllm_model)

        rollout_responses = []
        repeated_ground_truths = []

        for question, gt in zip(questions_batch["prompts"], questions_batch["ground_truths"]):
            for _ in range(group_size):
                response = eval_vllm_model.generate(
                    question,
                    temperature=sampling_temperature,
                    min_tokens=sampling_min_tokens,
                    max_tokens=sampling_max_tokens,
                    stop_sequences=["</answer>"],
                )
                rollout_responses.append(response)
                repeated_ground_truths.append(gt)

        advantages, raw_rewards, metadata_rewards = run_compute_group_normalized_rewards_util(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )

        micro_train_batch_size = train_batch_size // gradient_accumulation_steps
        n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

        for epoch_id in range(epochs_per_rollout_batch):
            for i in range(n_microbatches_per_rollout_batch):
                start_index = i * micro_train_batch_size
                end_index = (i + 1) * micro_train_batch_size

                responses = rollout_responses[start_index:end_index]
                advantages = advantages[start_index:end_index]
                raw_rewards = raw_rewards[start_index:end_index]

                input_ids = tokenizer(responses, return_tensors="pt", padding=True).to(device).input_ids
                labels = input_ids.clone() # TODO: Have to understatd this one

                log_probs_dict = run_get_response_log_probs_util(
                    model=model_train,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True
                )

                log_prob = log_probs_dict['log_probs']
                entropy = log_probs_dict['token_entropy']
                resp_mask = (labels != tokenizer.pad_token_id).long()

                old_log_probs_dict = run_get_response_log_probs_util(
                    model=old_policy_model,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=False
                )
                old_log_probs = old_log_probs_dict['log_probs']

                loss, metadata_loss = run_grpo_microbatch_train_step_util(
                    policy_log_probs=log_prob,
                    response_mask=resp_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    raw_rewards=raw_rewards,
                    advantages=advantages,
                    old_log_probs=old_log_probs,
                    cliprange= grpo_clip,
                )
                true_loss = loss.item() * gradient_accumulation_steps
                wandb.log({
                    "train/loss": true_loss,
                    "train/entropy": entropy.mean().item()
                }, step=step_count)

        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step_count += 1

        if step_count % eval_after == 0:
            load_policy_into_vllm_instance(model_train, eval_vllm_model)
            acc, reward = evaluate(eval_vllm_model, eval_prompts, eval_gts)
            wandb.log({"eval/accuracy": acc}, step=step_count)
            reward_reportable = {f"eval/reward/{key}": reward[key] for key in reward}
            wandb.log(reward_reportable, step=step_count)
            print(f"Step {step_count} eval:", acc)


if __name__ == '__main__':
    policy_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)

    # Hyper parameters
    # =======
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
    # =====

    # wandb things
    run_name = f"GRPO-lt{loss_type}-ga{str(gradient_accumulation_steps)}-ngs{str(n_grpo_steps)}_lr{str(learning_rate)}"
    wandb.init(
        project=f"assignment-3-test",
        name=run_name,
        config={
            "model": "transformer",
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
        }
    )
    # ===

    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    # get the dataloaders
    train_dataloader, test_dataloader = get_countdown_dataloaders(
        "student/data/countdown/dataset", n_prompts_per_rollout_batch
    )

    print("Loading initvllm")
    eval_vllm_model = init_vllm(
        model_id=policy_model_name,
        device=VLLM_DEVICE,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=2,
    )
    print("==loaded==")

    # get eval data: prompt + gts
    eval_prompts = []
    eval_gts = []
    for batch in test_dataloader:
        eval_prompts.extend(batch["prompts"])
        eval_gts.extend(batch["ground_truths"])













