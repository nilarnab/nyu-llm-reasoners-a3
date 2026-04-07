from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
import torch


def get_countdown_dataloaders(dataset_path, n_prompts_per_rollout_batch, seed=42):
    dataset = load_from_disk(dataset_path)

    def collate_fn(batch):
        prompts = []
        for item in batch:
            msgs = item["prompt"]  # list of {"role": ..., "content": ...}
            sys_msg  = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            prompt = sys_msg + "\n\n" + user_msg if sys_msg else user_msg
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
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


if __name__ == '__main__':
    train_dataloader, _ = get_countdown_dataloaders(
        "student/data/countdown",1
    )

    batch = next(iter(train_dataloader))
    print(batch["prompts"][0])
    print("---")
    print(batch["ground_truths"][0])

