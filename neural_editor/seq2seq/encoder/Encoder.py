from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence

from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import Batch
from neural_editor.seq2seq.BahdanauAttention import BahdanauAttention
from neural_editor.seq2seq.encoder import SrcEncoder


class Encoder(nn.Module):

    def __init__(self, src_encoder: SrcEncoder, edit_encoder: EditEncoder) -> None:
        super(Encoder, self).__init__()
        self.src_encoder = src_encoder
        self.edit_encoder = edit_encoder
        self.hidden_size = self.src_encoder.hidden_size * 2 + self.edit_encoder.hidden_size * 2
        self.edit_attention = BahdanauAttention(hidden_size=self.edit_encoder.hidden_size,
                                                key_size=self.edit_encoder.hidden_size * 2,
                                                query_size=None)
        self.src_attention = BahdanauAttention(hidden_size=self.src_encoder.hidden_size,
                                               key_size=self.src_encoder.hidden_size * 2,
                                               query_size=None)

    def forward(self, batch: Batch) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
        edit_final, src_final = self.encode(batch)
        src_output, _ = self.src_encoder(batch.src, batch.src_lengths)
        return edit_final, src_output, src_final

    def encode(self, batch: Batch) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        edit_hunks_final = self.edit_encoder.encode_edit(batch.diff_alignment_hunks,
                                                         batch.diff_prev_hunks, batch.diff_updated_hunks,
                                                         batch.diff_alignment_hunk_lengths)
        _, src_hunks_final = self.src_encoder(batch.src_hunks, batch.src_hunk_lengths)

        edit_final = (
            self.aggregate_with_attention(edit_hunks_final[0], batch.hunk_numbers, self.edit_attention),
            self.aggregate_with_attention(edit_hunks_final[1], batch.hunk_numbers, self.edit_attention)
        )
        src_final = (
            self.aggregate_with_attention(src_hunks_final[0], batch.hunk_numbers, self.src_attention),
            self.aggregate_with_attention(src_hunks_final[1], batch.hunk_numbers, self.src_attention)
        )

        return edit_final, src_final

    def aggregate_with_attention(self, hunks: Tensor, hunk_numbers_per_example: Tensor,
                                 attention: BahdanauAttention) -> Tensor:
        hunks_per_example, mask = self.reshape_hunks_with_batch_dimension(hunks, hunk_numbers_per_example)

        # unite batch size and numlayers dimensions
        hunks_per_example_reshaped = hunks_per_example.reshape(
            (-1, hunks_per_example.shape[2], hunks_per_example.shape[3]))
        mask_reshaped = mask.reshape((-1, 1, mask.shape[-1]))
        projection_key = attention.key_layer(hunks_per_example_reshaped)
        example_representation_reshaped, _ = attention(query=None, proj_key=projection_key,
                                                            value=hunks_per_example_reshaped, mask=mask_reshaped)
        example_representation = example_representation_reshaped.reshape((hunks_per_example.shape[0],
                                                                          hunks_per_example.shape[1],
                                                                          example_representation_reshaped.shape[1],
                                                                          example_representation_reshaped.shape[2]))
        return example_representation.squeeze(dim=2)

    def reshape_hunks_with_batch_dimension(self, hunks: Tensor, hunk_numbers_per_example: Tensor):
        # hunks: [NumLayers, TotalNumberOfHunks, H]
        hunks = hunks.permute(1, 0, 2)
        hunks = torch.split(hunks, hunk_numbers_per_example.tolist(), dim=0)
        hunks = pad_sequence(hunks, batch_first=False, padding_value=0)
        # hunks: [MaxNumberOfHunksInBatch, B, NumLayers, H]
        hunks = hunks.permute(2, 1, 0, 3)
        # hunks: [NumLayers, B, MaxNumberOfHunksInBatch, H]

        mask = torch.zeros((hunks.shape[0], hunks.shape[1], hunks.shape[2]), dtype=torch.bool)
        for batch_id, n in enumerate(hunk_numbers_per_example):
            mask[:, batch_id, :n] = True

        return hunks, mask

    def get_hidden_size(self) -> int:
        return self.hidden_size
