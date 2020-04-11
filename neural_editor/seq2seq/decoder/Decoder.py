import random
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import Embedding

from neural_editor.seq2seq import BahdanauAttention, Generator


# DONE_TODO: initialization = encoder output concatenate with edit representation.
# DONE_TODO: feed edit representation as input to decoder LSTM at each time step.
# TODO: consider Luong et al (2015). Looks like difference with current is: differenet attention (local vs global)
# TODO: add copying mechanism (Vinyals et al., 2015).
# TODO: large sequences need a lot of memory (out of memory)


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, generator: Generator, embedding: Embedding, emb_size: int, edit_representation_size: int,
                 hidden_size_encoder: int, hidden_size: int,
                 attention: BahdanauAttention,
                 teacher_forcing_ratio: float,
                 num_layers: int, dropout: float,
                 bridge: bool, use_edit_representation: bool) -> None:
        super(Decoder, self).__init__()

        self.generator = generator
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.use_edit_representation = use_edit_representation
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.rnn = nn.LSTM(emb_size + 2 * hidden_size_encoder + 2 * edit_representation_size, hidden_size,
                           num_layers, bidirectional=False,  # TODO_DONE: bidirectional=False?
                           batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size_encoder + 2 * edit_representation_size,
                                hidden_size, bias=True) if bridge else None  # 2 * EncoderH + 2 * EditH -> DecoderH

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size_encoder + emb_size,
                                          hidden_size, bias=False)  # DecoderH + 2 * EncoderH + Emb -> DecoderH

    def forward_step(self, edit_hidden: Tensor, prev_embed: Tensor, encoder_output: Tensor,
                     src_mask: Tensor, projection_key: Tensor,
                     hidden: Tensor, cell: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Perform a single decoder step (1 word)
        :arg edit_hidden: [NumLayers, B, NumDirections * DiffEncoderH]
        :arg prev_embed: [B, 1, EmbCode]
        :arg encoder_output: [B, SrcSeqLen, NumDirections * SrcEncoderH]
        :arg src_mask: [B, 1, SrcSeqLen]
        :arg projection_key: [B, SrcSeqLen, DecoderH]
        :arg hidden: [NumLayers, B, DecoderH]
        :arg cell: [NumLayers, B, DecoderH]
        :returns Tuple[[B, 1, DecoderH], [NumLayers, B, DecoderH], [NumLayers, B, DecoderH], [B, 1, DecoderH]]
        """

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [B, 1, DecoderH]
        # [B, 1, NumDirections * SrcEncoderH], [B, 1, SrcSeqLen]
        context, attn_probs = self.attention(
            query=query, proj_key=projection_key,
            value=encoder_output, mask=src_mask)

        # [B, 1, NumDirections * DiffEncoderH]
        edit_hidden_last_layer = edit_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1)
        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context, edit_hidden_last_layer], dim=2)
        # DONE_TODO: zeros or cell states from encoder
        # [B, 1, DecoderH], [NumLayers, B, DecoderH], [NumLayers, B, DecoderH]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # [B, 1, EmbCode + DecoderH + NumDirections * SrcEncoderH]
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)  # [B, 1, DecoderH]

        return output, hidden, cell, pre_output

    def forward(self, trg_embed: Tensor,
                edit_final: Tuple[Tensor, Tensor],
                encoder_output: Tensor, encoder_final: Tuple[Tensor, Tensor],
                src_mask: Tensor, trg_mask: Tensor,
                states_to_initialize: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        """
        Unroll the decoder one step at a time.
        :param trg_embed: [B, TrgSeqLen, EmbCode]
        :param edit_final: Tuple of [NumLayers, B, NumDirections * DiffEncoderH]
        :param encoder_output: [B, SrcSeqLen, NumDirections * SrcEncoderH]
        :param encoder_final: Tuple of [NumLayers, B, NumDirections * SrcEncoderH]
        :param src_mask: [B, 1, SrcSeqLen]
        :param trg_mask: [B, TrgSeqLen]
        :param states_to_initialize: Tuple[[[NumLayers, B, DecoderH]], [[NumLayers, B, DecoderH]]] hidden and cell
        :return: Tuple[
                 [B, TrgSeqLen, DecoderH],
                 Tuple[[NumLayers, B, DecoderH], [NumLayers, B, DecoderH]],
                 [B, TrgSeqLen, DecoderH]
        ]
        """
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False  # TODO: turn off teacher forcing if evaluation mode

        # the maximum number of steps to unroll the RNN
        max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        (edit_hidden, edit_cell) = edit_final  # Tuple of [NumLayers, B, NumDirections * DiffEncoderH]
        if not self.use_edit_representation:
            edit_hidden = torch.zeros_like(edit_hidden, requires_grad=False)
            edit_cell = torch.zeros_like(edit_cell, requires_grad=False)
        (encoder_hidden, encoder_cell) = encoder_final  # Tuple of [NumLayers, B, NumDirections * SrcEncoderH]

        if states_to_initialize is None:
            hidden = self.init_hidden(edit_hidden, encoder_hidden)  # [NumLayers, B, DecoderH]
            cell = self.init_hidden(edit_cell, encoder_cell)  # [NumLayers, B, DecoderH]
        else:
            hidden, cell = states_to_initialize

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        projection_key = self.attention.key_layer(encoder_output)  # [B, SrcSeqLen, DecoderH]

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            if use_teacher_forcing or i == 0:
                prev_embed = trg_embed[:, i].unsqueeze(1)  # [B, 1, EmbCode]
            else:
                _, top_i = self.generator(pre_output_vectors[-1]).squeeze().topk(1)
                prev_embed = self.embedding(top_i)  # TODO: stop decoding if EOS token? seems to me we should continue
            # [B, 1, DecoderH], [NumLayers, B, DecoderH], [NumLayers, B, DecoderH], [B, 1, DecoderH]
            output, hidden, cell, pre_output = self.forward_step(
                edit_hidden, prev_embed, encoder_output, src_mask, projection_key, hidden, cell)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)  # [B, TrgSeqLen, DecoderH]
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)  # [B, TrgSeqLen, DecoderH]
        return decoder_states, (hidden, cell), pre_output_vectors  # [B, N, D]

    def init_hidden(self, edit_final: Tensor, encoder_final: Tensor) -> Tensor:
        """
        Returns the initial decoder state, conditioned on the final encoder state.
        :param edit_final: [NumLayers, B, NumDirections * DiffEncoderH]
        :param encoder_final: [NumLayers, B, NumDirections * SrcEncoderH]
        :return: [NumLayers, B, DecoderH]
        """

        if encoder_final is None or edit_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(torch.cat((encoder_final, edit_final), dim=2)))
