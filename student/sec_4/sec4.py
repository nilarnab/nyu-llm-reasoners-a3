import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from tests.conftest import response_mask
from torch.nn.utils.rnn import pad_sequence


def run_tokenize_prompt_and_output_util(
        prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:

    input_ids = []
    # labels = []
    masks = []
    max_len = 0

    for i in range(len(prompt_strs)):
        prompt_str = prompt_strs[i]
        output_str = output_strs[i]


        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_ids = tokenizer.encode(output_str, add_special_tokens=False)
        prompt_output_ids = prompt_ids + output_ids

        response_mask = [0]*len(prompt_ids) + [1]*len(output_ids)

        input_ids.append(torch.tensor(prompt_output_ids))
        masks.append(torch.tensor(response_mask))
    
    pad_id = tokenizer.pad_token_id

    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    ) 

    response_mask_padded = pad_sequence(
        masks, batch_first=True, padding_value=0
    ) 

    input_ids    = input_ids_padded[:, :-1]
    labels       = input_ids_padded[:, 1:]
    response_mask = response_mask_padded[:, 1:] 


    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }




    return {}