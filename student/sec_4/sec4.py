import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output_util(
        prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:

    input_ids = []
    labels = []
    masks = []
    max_len = 0

    for i in range(len(prompt_strs)):
        prompt_str = prompt_strs[i]
        output_str = output_strs[i]


        prompt_ids = tokenizer.encode(prompt_str)
        output_ids = tokenizer.encode(output_str)

        prompt_output_ids = prompt_ids + output_ids

        input_id_one = prompt_output_ids[:-1]
        max_len = max(max_len, len(input_id_one))
        targets_one = prompt_output_ids[1: ]

        response_mask = [0] * (len(prompt_ids) - 1)
        response_mask.extend([1] * len(output_ids))

        input_ids.append(input_id_one)
        labels.append(targets_one)
        masks.append(response_mask)

    input_ids_pad = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in input_ids]
    labels_pad = [x + [-100] * (max_len - len(x)) for x in labels]  # TODO: Try something else as well
    masks_pad = [x + [0] * (max_len - len(x)) for x in masks]

    input_ids_pad_tensor = torch.Tensor(input_ids_pad)
    labels_pad_tensor = torch.Tensor(labels_pad)
    masks_pad_tensor = torch.Tensor(masks_pad)

    return {
        "input_ids": input_ids_pad_tensor,
        "labels": labels_pad_tensor,
        "response": masks_pad_tensor
    }




    return {}