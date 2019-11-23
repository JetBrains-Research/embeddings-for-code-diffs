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
        self.encoder = encoder
        self.decoder = decoder
        self.edit_encoder = edit_encoder
        self.embed = embed
        self.generator = generator

    def forward(self, batch: Batch) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Take in and process masked src and target sequences.
        Returns tuple of decoder states, hidden states of decoder, pre-output states.
        Pre-output combines output states with context and embedding of previous token
        :param batch: batch to process
        :return: Tuple[[B, TrgSeqLen, DecoderH], [NumLayers, B, DecoderH], [B, TrgSeqLen, DecoderH]]
        """
        edit_output, edit_final, encoder_output, encoder_final = self.encode(batch)
        decoded = self.decode(edit_final, encoder_output,
                              encoder_final, batch.src_mask,
                              batch.trg, batch.trg_mask)
        return decoded

    def encode(self, batch: Batch) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
        """
        Encodes edits and prev sequences
        :param batch: batch to process
        :return: Tuple[
            [B, AlignedSeqLen, NumDirections * DiffEncoderH],
            Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]],
            [B, SrcSeqLen, NumDirections * SrcEncoderH],
            Tuple[[NumLayers, B, NumDirections * SrcEncoderH], [NumLayers, B, NumDirections * SrcEncoderH]]
        ]
        """
        diff_embedding = torch.cat(
            (self.embed(batch.diff_alignment), self.embed(batch.diff_prev), self.embed(batch.diff_updated)),
            dim=2
        )  # [B, SeqAlignedLen, EmbDiff + EmbDiff + EmbDiff]
        diff_embedding_mask = torch.cat(
            (batch.diff_alignment_mask, batch.diff_prev_mask, batch.diff_updated_mask),
            dim=2
        )  # [B, 1, AlignedSeqLen + AlignedSeqLen + AlignedSeqLen]
        # [B, AlignedSeqLen, NumDirections * DiffEncoderH]
        # Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]]
        edit_output, edit_final = self.edit_encoder(
            diff_embedding,
            diff_embedding_mask,
            batch.diff_alignment_lengths  # B * 1 * AlignedSeqLen
        )
        encoder_output, encoder_final = self.encoder(self.embed(batch.src), batch.src_mask, batch.src_lengths)
        return edit_output, edit_final, encoder_output, encoder_final

    def decode(self, edit_final: Tuple[Tensor, Tensor],
               encoder_output: Tensor, encoder_final: Tuple[Tensor, Tensor],
               src_mask: Tensor, trg: Tensor, trg_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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
        :return: Tuple[[B, TrgSeqLen, DecoderH], [NumLayers, B, DecoderH], [B, TrgSeqLen, DecoderH]]
        """
        return self.decoder(self.embed(trg), edit_final, encoder_output, encoder_final,
                            src_mask, trg_mask)
