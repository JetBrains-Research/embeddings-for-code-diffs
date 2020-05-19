from typing import Tuple

import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch import Tensor
from torchtext import data
import numpy as np

from edit_representation.sequence_encoding import EditEncoder
from neural_editor.seq2seq import Generator, Batch
from neural_editor.seq2seq.ClassifierBatch import ClassifierBatch
from neural_editor.seq2seq.classifier import GoodEditClassifier
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.datasets.dataset_utils import take_subset_from_dataset
from neural_editor.seq2seq.decoder import Decoder
from neural_editor.seq2seq.encoder import Encoder
from neural_editor.seq2seq.Batch import rebatch


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, edit_encoder: EditEncoder,
                 embed: nn.Embedding, generator: Generator, config: Config) -> None:
        super(EncoderDecoder, self).__init__()
        self.edit_final = None
        self.encoded_train = None
        self.encoder = encoder
        self.decoder = decoder
        self.edit_encoder = edit_encoder
        self.embed = embed
        self.generator = generator
        self.config = config
        self.train_dataset = None
        self.pad_index = None
        self.classifier: GoodEditClassifier = None
        self.metric = config['METRIC']

    def forward(self, batch: Batch, ignore_encoded_train) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
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
        edit_final, encoder_output, encoder_final = self.encode(batch, ignore_encoded_train)
        decoded = self.decode(edit_final, encoder_output,
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

    def set_training_data(self, train_dataset: data.Dataset, pad_index: int):
        self.train_dataset = train_dataset
        self.pad_index = pad_index
        self.update_training_vectors()

    def update_training_vectors(self) -> None:
        encoded_train_unsorted = {'src_hidden': [], 'edit_hidden': [], 'edit_cell': [], 'ids': []}
        data_iterator = data.Iterator(self.train_dataset, batch_size=self.config['BATCH_SIZE'], train=False,
                                      sort_within_batch=True,
                                      sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                      device=self.config['DEVICE'])
        data_iterator = [rebatch(self.pad_index, batch, self.config) for batch in data_iterator]

        for batch in data_iterator:
            (edit_hidden, edit_cell), _, (encoder_hidden, _) = self.encode(batch, ignore_encoded_train=True)
            encoded_train_unsorted['src_hidden'].append(encoder_hidden[-1].detach().cpu())
            encoded_train_unsorted['edit_hidden'].append(edit_hidden.detach().cpu())
            encoded_train_unsorted['edit_cell'].append(edit_cell.detach().cpu())
            encoded_train_unsorted['ids'].append(batch.ids.detach().cpu())
        encoded_train_unsorted['src_hidden'] = torch.cat(encoded_train_unsorted['src_hidden'], dim=0)
        encoded_train_unsorted['edit_hidden'] = torch.cat(encoded_train_unsorted['edit_hidden'], dim=1)
        encoded_train_unsorted['edit_cell'] = torch.cat(encoded_train_unsorted['edit_cell'], dim=1)
        encoded_train_unsorted['ids'] = torch.cat(encoded_train_unsorted['ids'], dim=0)
        ids_reverse = torch.argsort(encoded_train_unsorted['ids'])

        src_hidden_sorted = encoded_train_unsorted['src_hidden'][ids_reverse, :].numpy()
        encoded_train_sorted = {
            'edit_hidden': encoded_train_unsorted['edit_hidden'][:, ids_reverse, :],
            'edit_cell': encoded_train_unsorted['edit_cell'][:, ids_reverse, :],
            'nbrs': NearestNeighbors(n_neighbors=1, algorithm='brute', metric=self.metric, n_jobs=-1).fit(
                src_hidden_sorted)
        }
        self.encoded_train = encoded_train_sorted

    def unset_training_data(self) -> None:
        self.encoded_train = None
        self.train_dataset = None
        self.pad_index = None

    def get_neighbors(self, n_neighbors) -> None:
        return self.encoded_train['nbrs'].kneighbors(n_neighbors=n_neighbors, return_distance=False)

    def get_edit_final_from_train(self, src: Tensor, n_neighbors=None, src_indices=None) -> Tuple[Tensor, Tensor]:
        src = src.detach().cpu().numpy()
        # TODO: get rid of "if" by filtering on batch.ids
        # TODO_DONE: find out why distances are not zeros, reason: dropout in LSTM introduces randomness
        # TODO: but still some examples from the very first batch
        #   doesn't match (difference between encoders outputs is < 1e-2), therefore maybe it is just calculation
        #   errors, such examples are rare (~6 from 64)
        if n_neighbors is None:
            if self.training:
                indices = self.encoded_train['nbrs'].kneighbors(src, n_neighbors=2, return_distance=False)
                indices = indices[:, 1]
            else:
                if self.classifier is None:
                    indices = self.encoded_train['nbrs'].kneighbors(src, return_distance=False)
                    indices = indices[:, 0]
                else:
                    indices = self.encoded_train['nbrs'].kneighbors(src, n_neighbors=50, return_distance=False)
                    indices = self.resort_indices(src_indices, indices)
                    indices = indices[:, 0]
            if not self.config['BUILD_EDIT_VECTORS_EACH_QUERY']:
                return self.encoded_train['edit_hidden'][:, indices, :].to(self.config['DEVICE']), \
                       self.encoded_train['edit_cell'][:, indices, :].to(self.config['DEVICE'])
            return self.encode_edit(self.get_batch_from_ids(indices))
        else:
            indices = self.encoded_train['nbrs'].kneighbors(src, n_neighbors=n_neighbors, return_distance=False)
            if self.classifier is not None:
                indices = self.resort_indices(src_indices, indices)
            output = []
            for i in range(n_neighbors):
                i_indices = indices[:, i]
                if not self.config['BUILD_EDIT_VECTORS_EACH_QUERY']:
                    output.append((self.encoded_train['edit_hidden'][:, i_indices, :].to(self.config['DEVICE']),
                                   self.encoded_train['edit_cell'][:, i_indices, :].to(self.config['DEVICE'])))
                else:
                    output.append(self.encode_edit(self.get_batch_from_ids(i_indices)))
            return output, indices

    def get_batch_from_ids(self, indices):
        dataset = take_subset_from_dataset(self.train_dataset, indices)
        data_iterator = data.Iterator(dataset, batch_size=len(dataset), train=False,
                                      sort_within_batch=False,
                                      sort=False, repeat=False,
                                      device=self.config['DEVICE'])
        data_iterator = [rebatch(self.pad_index, batch, self.config) for batch in data_iterator]
        return data_iterator[0]

    def set_classifier(self, classifier: GoodEditClassifier):
        self.classifier = classifier
        self.classifier.eval()

    def unset_classifier(self):
        self.classifier = None

    def resort_indices(self, src_examples: Tensor, indices: np.ndarray) -> np.ndarray:
        sorted_indices = np.empty_like(indices)
        for src_i, neighbors_ids in enumerate(indices):
            src_with_padding = src_examples[src_i]
            src_without_padding = src_with_padding[src_with_padding != self.pad_index]
            src = torch.cat(len(neighbors_ids) * [src_without_padding.unsqueeze(dim=0)])
            originral_src = (src, torch.tensor([src_without_padding.shape[0]] * len(neighbors_ids)))

            neighbors_batch = self.get_batch_from_ids(neighbors_ids)
            edit_src = (neighbors_batch.src, neighbors_batch.src_lengths)

            classifier_batch = ClassifierBatch(originral_src, edit_src, trg=None, pad_index=self.pad_index)
            predicted = self.classifier.predict(classifier_batch)
            sorted_indices[src_i] = neighbors_ids[torch.argsort(predicted, descending=True).detach().cpu().numpy()]
        return sorted_indices

    def encode_edit(self, batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Returns edit representations (edit_final) of samples in the batch.
        :param batch: batch to encode
        :return: Tuple[[NumLayers, B, NumDirections * DiffEncoderH], [NumLayers, B, NumDirections * DiffEncoderH]]
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
        _, edit_final = self.edit_encoder(
            diff_embedding,
            diff_embedding_mask,
            batch.diff_alignment_lengths  # B * 1 * AlignedSeqLen
        )
        return edit_final

    def encode(self, batch: Batch, ignore_encoded_train=False, n_neighbors=None) -> Tuple[
        Tuple[Tensor, Tensor], Tensor, Tuple[Tensor, Tensor]]:
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
        if self.edit_final is not None:
            edit_final = self.edit_final
        elif self.encoded_train is not None and not ignore_encoded_train:
            edit_final = self.get_edit_final_from_train(encoder_final[0][-1], n_neighbors, batch.src)
        else:
            edit_final = self.encode_edit(batch)
        return edit_final, encoder_output, encoder_final

    def decode(self, edit_final: Tuple[Tensor, Tensor],
               encoder_output: Tensor, encoder_final: Tuple[Tensor, Tensor],
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
        return self.decoder(self.embed(trg), edit_final, encoder_output, encoder_final,
                            src_mask, trg_mask, states_to_initialize)
