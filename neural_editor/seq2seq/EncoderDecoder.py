from typing import Tuple

from torch import Tensor
from torch import nn

from neural_editor.seq2seq import Generator, Batch
from neural_editor.seq2seq.decoder import Decoder
from neural_editor.seq2seq.encoder import Encoder


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 target_embed: nn.Embedding, generator: Generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_embed = target_embed
        self.generator = generator
        self.edit_final = None

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
        edit_final, encoder_output, encoder_final = self.encoder(batch)
        decoded = self.decode(batch, edit_final, encoder_output,
                              encoder_final, batch.src_mask,
                              batch.trg, batch.trg_mask, None)
        return decoded

    def set_edit_representation(self, batch: Batch) -> None:
        self.edit_final, _, _ = self.encoder(batch)

    def unset_edit_representation(self) -> None:
        self.edit_final = None

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
        edit_final_current, src_output, src_final = self.encoder(batch)
        edit_final = self.edit_final if self.edit_final is not None else edit_final_current
        return edit_final, src_output, src_final

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
        return self.decoder(batch, self.target_embed(trg), edit_final, encoder_output, encoder_final,
                            src_mask, trg_mask, states_to_initialize)
