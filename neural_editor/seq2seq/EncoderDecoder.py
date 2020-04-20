from typing import Tuple

import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch import Tensor
from torchtext import data

from neural_editor.seq2seq import Generator, Batch
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder import Decoder
from neural_editor.seq2seq.encoder import Encoder


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, embed: nn.Embedding, generator: Generator, config: Config) -> None:
        super(EncoderDecoder, self).__init__()
        self.edit_final = None
        self.encoded_train = None
        self.encoder = encoder
        self.decoder = decoder
        self.embed = embed
        self.generator = generator
        self.config = config
        self.train_dataset = None
        self.pad_index = None

    def forward(self, batch: Batch) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Take in and process masked src and target sequences.
        Returns tuple of decoder states, hidden states of decoder, pre-output states.
        Pre-output combines output states with context and embedding of previous token
        :param ignore_encoded_train: if we should ignore encoded train
        :param batch: batch to process
        :return:  Tuple[
                 [B, TrgSeqLen, DecoderH],
                 Tuple[[NumLayers, B, DecoderH], [NumLayers, B, DecoderH]],
                 [B, TrgSeqLen, DecoderH]
        ]
        """
        encoder_output, encoder_final = self.encode(batch)
        decoded = self.decode(encoder_output,
                              encoder_final, batch.src_mask,
                              batch.trg, batch.trg_mask, None)
        return decoded

    def encode(self, batch: Batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Encodes edits and prev sequences
        :param ignore_encoded_train: if we should ignore encoded train
        :param batch: batch to process
        :return: Tuple[
            Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]],
            [B, SrcSeqLen, NumDirections * SrcEncoderH],
            Tuple[[NumLayers, B, NumDirections * SrcEncoderH], [NumLayers, B, NumDirections * SrcEncoderH]]
        ]
        """
        encoder_output, encoder_final = self.encoder(self.embed(batch.src), batch.src_mask, batch.src_lengths)
        return encoder_output, encoder_final

    def decode(self, encoder_output: Tensor, encoder_final: Tuple[Tensor, Tensor],
               src_mask: Tensor, trg: Tensor, trg_mask: Tensor,
               states_to_initialize: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
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
        return self.decoder(self.embed(trg), encoder_output, encoder_final,
                            src_mask, trg_mask, states_to_initialize)
