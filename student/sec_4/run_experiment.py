import os

from datasets import load_from_disk, Dataset, load_dataset
from torch import device
from torch.utils.data import DataLoader
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import torch
from unittest.mock import patch

from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

from student.evaluate import setup_logger, evaluate, load_prompt
from student.sec_4.sec4 import run_tokenize_prompt_and_output_util, run_get_response_log_probs_util, \
    run_sft_microbatch_train_step_util
from student.utils import DEVICE
import wandb



os.environ["WANDB_API_KEY"] = "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"
wandb.login(key=os.environ["WANDB_API_KEY"])


def get_eval_intellect_dataloader(dataset_path, example_count, batch_size):
    print('Intellect dataloader called')
    dataset = load_from_disk(dataset_path)
    if example_count:
        dataset = dataset.select(range(min(example_count, len(dataset))))

    prompts = []
    responses = []
    for ex in dataset:
        msgs = ex.get("messages", [])
        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
        responses.append(ex.get("ground_truth", ""))

    dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_eval_math_dataloader(example_count, batch_size, dataset_path="hiyouga/math12k"):
    prompt_template = load_prompt("intellect")

    math_ds = load_dataset(dataset_path, split="train")
    if example_count:
        math_ds = math_ds.select(range(min(example_count, len(math_ds))))

    prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
    gts = [ex["answer"] for ex in math_ds]
    pass


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
    return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
        model=model_id,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def run_sft_loop(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=DEVICE,
        epoch=3,
        grad_accum_steps=4,
        eval_after=20):


    model_train.train()
    step_count = 0
    optimizer.zero_grad()

    for epoch_id in range(epoch):
        for batch in dataloader:
            step_count += 1
            print("STEP COUNT", step_count)

            prompts = batch["prompt"]
            resps = batch["response"]

            res = run_tokenize_prompt_and_output_util(
                prompts, resps, tokenizer
            )
            print("tokenisze rpompt and output util done")

            input_ids = res['input_ids'].to(device)
            labels = res['labels'].to(device)
            response_mask = res['response_mask'].to(device)

            res_logprobs = run_get_response_log_probs_util(model_train, input_ids, labels, return_token_entropy=True)
            log_probs = res_logprobs['log_probs']
            entropy = res_logprobs['token_entropy']

            print("get response log probs done")

            # TODO: Understand normalize constant thing
            loss, metadata = run_sft_microbatch_train_step_util(log_probs,response_mask, grad_accum_steps, response_mask.sum(dim=-1))
            wandb.log({"eval/accuracy": loss, "entropy": entropy}, step=step_count)
            print("loss", loss)

            print("run sft microbatch done")

            if step_count % grad_accum_steps == 0:
                print("Taking grad accc step")
                optimizer.step()
                optimizer.zero_grad()

            if step_count % eval_after == 0:
                print("Running Eval")
                load_policy_into_vllm_instance(model_train, eval_vllm_model)
                acc = evaluate(eval_vllm_model, eval_prompts, eval_gts)
                wandb.log({"eval/accuracy": acc}, step=step_count)
                print('eval', acc)

        if step_count % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--dataset-type", type=str, default="INTELLECT")
    parser.add_argument("--intellect-train-path", default="student/data/intellect_math/train")
    parser.add_argument("--intellect-test-path", default="student/data/intellect_math/test")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--log-file", default="eval.log")

    args = parser.parse_args()

    logger = setup_logger(args.log_file)

    print("Loading training model")
    model_train = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)


    print("Loading tokenize")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("===loaded===")


    batch_size = 1
    example_count = 128
    learning_rate = 1e-4

    wandb.init(
        project=f"assignment-3-test",
        name=f"SFT4-batch_size{str(batch_size)}_lr{str(learning_rate)}",
        config={
            "model": "transformer",
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    )

    dataloader = get_eval_math_dataloader(example_count, batch_size) \
        if args.dataset_type == 'MATH' \
        else get_eval_intellect_dataloader(args.intellect_train_path, example_count, batch_size)

    optimizer = torch.optim.AdamW(model_train.parameters(), lr=learning_rate)

    print("Loading initvllm")
    eval_vllm_model = init_vllm(
        model_id=args.model,
        device=DEVICE,
        seed=2,
    )
    print("==loaded==")

    prompt_template = load_prompt("intellect")

    eval_prompts = []
    eval_gts = []
    if args.dataset_type == 'MATH':
        math_ds = load_dataset("hiyouga/math12k", split="test")
        if args.max_examples:
            math_ds = math_ds.select(range(min(args.max_examples, len(math_ds))))

        eval_prompts = [prompt_template + "\n\n" + ex["problem"] for ex in math_ds]
        eval_gts = [ex["answer"] for ex in math_ds]
    else:
        dataset = load_from_disk(args.intellect_test_path)
        if args.max_examples:
            dataset = dataset.select(range(min(args.max_examples, len(dataset))))

        for ex in dataset:
            msgs = ex.get("messages", [])
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            eval_prompts.append(sys_msg + "\n\n" + user_msg if sys_msg else user_msg)
            eval_gts.append(ex.get("ground_truth", ""))


    run_sft_loop(
        model_train,
        dataloader,
        tokenizer,
        optimizer,
        eval_vllm_model,
        eval_prompts,
        eval_gts,
        device=DEVICE,
        epoch=3,
        grad_accum_steps=4,
        eval_after=20)

    wandb.finish()


main()

