import torch
from torch import nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, edit_encoder, embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.edit_encoder = edit_encoder
        self.embed = embed
        self.generator = generator

    def forward(self, batch):
        """Take in and process masked src and target sequences."""
        (encoder_hidden, encoder_final), _, edit_representation_final = self.encode(batch)
        decoded = self.decode(edit_representation_final, encoder_hidden,
                              encoder_final, batch.src_mask,
                              batch.trg, batch.trg_mask)
        return decoded

    def encode(self, batch):
        edit_representation_hidden, edit_representation_final = self.edit_encoder(
            torch.cat((self.embed(batch.diff_alignment), self.embed(batch.diff_prev), self.embed(batch.diff_updated)), dim=2),
            torch.cat((batch.diff_alignment_mask, batch.diff_prev_mask, batch.diff_updated_mask), dim=2),
            batch.diff_alignment_lengths
        )
        return self.encoder(self.embed(batch.src), batch.src_mask, batch.src_lengths), \
               edit_representation_hidden, edit_representation_final

    def decode(self, edit_representation, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.embed(trg), edit_representation, encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)
