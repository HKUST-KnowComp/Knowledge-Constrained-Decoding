import torch


def zero_out_after_eos(sequence, eos_token_id):
    """In the batched greedy decode, the sequence generation is not stopped
    at eos. Manually zero-out the tokens after eos with padding tokens.
    """
    row, col = torch.where(sequence == eos_token_id)
    for i, j in zip(row, col):
        col_idx = j + 1
        sequence[i][col_idx:] = eos_token_id

    return sequence
