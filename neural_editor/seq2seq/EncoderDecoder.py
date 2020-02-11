from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import Generator, Batch
from neural_editor.seq2seq.decoder import Decoder
from neural_editor.seq2seq.encoder import Encoder


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, edit_encoder: EditEncoder,
                 embed: nn.Embedding, generator: Generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.edit_final = None
        self.encoder = encoder
        self.decoder = decoder
        self.edit_encoder = edit_encoder
        self.embed = embed
        self.generator = generator

    def forward(self, batch: Batch) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
        """
        Take in and process masked src and target sequences.
        Returns tuple of decoder states, hidden states of decoder, pre-output states.
        Pre-output combines output states with context and embedding of previous token
        :param batch: batch to process
        :return:  Tuple[
                 [B, TrgSeqLen, DecoderH],
                 Tuple[[NumLayers, B, DecoderH], [NumLayers, B, DecoderH]],
                 [B, TrgSeqLen, DecoderH]
        ]
        """
        edit_final, encoder_output, encoder_final = self.encode(batch)
        decoded = self.decode(batch, edit_final, encoder_output,
                              encoder_final, batch.src_mask,
                              batch.trg, batch.trg_mask, None)
        return decoded

    def set_edit_representation(self, sample: Batch) -> None:
        """
        Fixates edit_final vector. Used for one-shot learning.
        :param sample: sample from which construct edit representation, it is batch with size 1
        :return: nothing
        """
        self.edit_final = self.encode_edit(sample)

    def unset_edit_representation(self) -> None:
        """
        Unset edit representation. Turns off one-shot learning mode.
        :return: nothing
        """
        self.edit_final = None

    def encode_edit(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Returns edit representations (edit_final) of samples in the batch.
        :param batch: batch to encode
        :return: Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]]
        """
        return self.edit_encoder.encode_edit(batch)

    def encode(self, batch: Batch) -> Tuple[Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
        """
        Encodes edits and prev sequences
        :param batch: batch to process
        :return: Tuple[
            Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]],
            [B, SrcSeqLen, NumDirections * SrcEncoderH],
            Tuple[[NumLayers, B, NumDirections * SrcEncoderH], [NumLayers, B, NumDirections * SrcEncoderH]]
        ]
        """
        if self.edit_final is None:
            edit_final = self.encode_edit(batch)
        else:
            edit_final = self.edit_final
        encoder_output, encoder_final = self.encoder(self.embed(batch.src), batch.src_mask, batch.src_lengths)
        return edit_final, encoder_output, encoder_final

    def decode(self, batch: Batch, edit_final: Tuple[Tensor, Tensor],
               encoder_output: Tensor, encoder_final: Tuple[Tensor, Tensor],
               src_mask: Tensor, trg: Tensor, trg_mask: Tensor,
               states_to_initialize: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
        """
        :param edit_final: Tuple[
            [NumLayers, B, NumDirections * DiffEncoderH],
            [NumLayers, B, NumDirections * DiffEncoderH]
        ]
        :param encoder_output: [B, SrcSeqLen, NumDirections * SrcEncoderH]
        :param encoder_final: Tuple[
            [NumLayers, B, NumDirections * SrcEncoderH],
            [NumLayers, B, NumDirections * SrcEncoderH]
        ]
        :param src_mask: [B, 1, SrcSeqLen]
        :param trg: [B, TrgSeqLen]
        :param trg_mask: [B, TrgSeqLen]
        :param states_to_initialize: Tuple[[NumLayers, B, DecoderH], [NumLayers, B, DecoderH]] hidden and cell states
        :return: Tuple[
                 [B, TrgSeqLen, DecoderH],
                 Tuple[[NumLayers, B, DecoderH], [NumLayers, B, DecoderH]],
                 [B, TrgSeqLen, DecoderH]
        ]
        """
        return self.decoder(batch, self.embed(trg), edit_final, encoder_output, encoder_final,
                            src_mask, trg_mask, states_to_initialize)
