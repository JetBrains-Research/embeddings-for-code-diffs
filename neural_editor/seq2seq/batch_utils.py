from typing import Tuple, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def split_sequences_on_hunks(sequences: Tensor, new_hunk_index: int,
                             pad_index: int, sos_index: int, eos_index: int, device) -> Tuple[Tensor, Tensor]:
    # put all examples in a row
    sequences_in_row = sequences.reshape(-1)
    # remove batch padding, SOS and EOS tokens
    exclude_special_tokens_mask = (sequences_in_row != pad_index) & \
                                  (sequences_in_row != sos_index) & \
                                  (sequences_in_row != eos_index)
    sequences_in_row = sequences_in_row[exclude_special_tokens_mask]
    # get lengths of hunks and apply split based on them
    hunk_ids = torch.nonzero(sequences_in_row == new_hunk_index)
    split_sizes = (hunk_ids[1:] - hunk_ids[:-1]).view(-1).tolist()
    split_sizes.append(len(sequences_in_row) - hunk_ids[-1].item())
    # split into hunks
    hunks = list(torch.split(sequences_in_row, split_sizes))
    # append EOS token to sequences
    add_eos_tokens(hunks, eos_index, device)
    # add padding
    hunks = pad_sequence(hunks, batch_first=True,
                         padding_value=pad_index)
    # change new hunk token on SOS token
    hunks[:, 0] = sos_index
    # gather lengths
    lengths = torch.nonzero(hunks == eos_index)[:, 1].view(-1) + 1
    return hunks, lengths


def add_eos_tokens(hunks: List[Tensor], eos_index: int, device):
    eos_token = torch.tensor([eos_index]).to(device)
    for i, sequence_hunk in enumerate(hunks):
        hunks[i] = torch.cat((sequence_hunk, eos_token))
