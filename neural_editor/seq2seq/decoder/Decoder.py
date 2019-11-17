import torch
from torch import nn

# DONE_TODO: initialization = encoder output concatenate with edit representation.
# TODO: feed edit representation as input to decoder LSTM at each time step.
# TODO: consider Luong et al (2015). Looks like difference with current is: differenet attention (local vs global)
# TODO: add copying mechanism (Vinyals et al., 2015).
# TODO: large sequences need a lot of memory (out of memory)


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, edit_representation_size, hidden_size_encoder, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.LSTM(emb_size + 2 * hidden_size_encoder + 2 * edit_representation_size, hidden_size,
                           num_layers, bidirectional=False,  # TODO: bidirectional=False?
                           batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size_encoder + 2 * edit_representation_size,
                                hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size_encoder + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, edit_representation_final, prev_embed, encoder_hidden, src_mask, proj_key, hidden, cell_state):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context, edit_representation_final.transpose(0, 1)], dim=2)
        # DONE_TODO: zeros or cell states from encoder
        output, (hidden, cell_state) = self.rnn(rnn_input, (hidden, cell_state))

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, cell_state, pre_output

    def forward(self, trg_embed,
                edit_representation_final, edit_representation_cell_state,
                encoder_hidden, encoder_final, encoder_cell_state,
                src_mask, trg_mask, hidden=None, cell_state=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(edit_representation_final, encoder_final)
        if cell_state is None:
            cell_state = self.init_hidden(edit_representation_cell_state, encoder_cell_state)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, cell_state, pre_output = self.forward_step(
                edit_representation_final, prev_embed, encoder_hidden, src_mask, proj_key, hidden, cell_state)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, edit_representation_final, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None or edit_representation_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(torch.cat((encoder_final, edit_representation_final), dim=2)))
