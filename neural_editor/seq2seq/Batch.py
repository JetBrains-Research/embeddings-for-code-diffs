from typing import Tuple, Dict

import torch
import torchtext
from torch import Tensor
from torchtext.data import Dataset

from neural_editor.seq2seq.batch_utils import split_sequences_on_hunks
from neural_editor.seq2seq.config import Config


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    @staticmethod
    def create_oov_vocab(ids: Tensor, dataset: Dataset, config: Config) -> Tuple[Dict[str, int], Tensor]:
        trg_vocab = dataset.fields['trg'].vocab
        examples = [dataset[i] for i in ids]
        src_oov_vocab = {}
        cur_id = len(trg_vocab)
        oov_indices = []
        for example in examples:
            for src_token in example.src:
                if trg_vocab.stoi[src_token] == trg_vocab.unk_index:
                    if src_token not in src_oov_vocab:
                        src_oov_vocab[src_token] = cur_id
                        cur_id += 1
                    oov_indices.append(src_oov_vocab[src_token])
        return src_oov_vocab, torch.tensor(oov_indices).long().to(config['DEVICE'])

    @staticmethod
    def get_extended_target(trg_y: Tensor, ids: Tensor, dataset: Dataset, oov_vocab: Dict[str, int]) -> Tensor:
        unk_index = dataset.fields['trg'].vocab.unk_index
        trg = trg_y.clone().detach()
        for i in range(trg_y.shape[0]):
            sequence = dataset[ids[i]].trg
            for j in range(trg_y.shape[1]):
                if j < len(sequence):
                    token = sequence[j]
                    if trg[i][j] == unk_index and token in dataset[ids[i]].src:
                        trg[i][j] = oov_vocab[token]
        return trg

    @staticmethod
    def create_scatter_indices(src: Tensor, ids: Tensor, oov_indices: Tensor, pad_index: int,
                               dataset: Dataset) -> Tensor:
        # TODO: optimizer and get rid of this method
        # TODO: consider to exclude from scatter_indices <s> and </s> (but it looks like a bad idea)
        trg_vocab = dataset.fields['trg'].vocab
        examples = [dataset[i] for i in ids]
        scatter_indices = torch.zeros_like(src).fill_(pad_index)
        scatter_indices[:, 0] = src[:, 0]  # <s> token
        scatter_indices[:, -1] = src[:, -1]  # </s> token
        for i, example in enumerate(examples):
            for j, src_token in enumerate(example.src):
                scatter_indices[i][j + 1] = trg_vocab.stoi[src_token]  # j + 1 to mitigate <s> token
        scatter_indices[scatter_indices == trg_vocab.unk_index] = oov_indices
        return scatter_indices

    def __init__(self, src: Tuple[Tensor, Tensor], trg: Tuple[Tensor, Tensor],
                 diff_alignment: Tuple[Tensor, Tensor],
                 diff_prev: Tuple[Tensor, Tensor], diff_updated: Tuple[Tensor, Tensor],
                 ids: Tensor, original_ids: Tensor,
                 dataset: Dataset, config: Config) -> None:
        self.hunk_index = dataset.fields['src'].vocab.stoi[config['HUNK_TOKEN']]
        self.pad_index = dataset.fields['src'].vocab.stoi[config['PAD_TOKEN']]
        self.sos_index = dataset.fields['src'].vocab.stoi[config['SOS_TOKEN']]
        self.eos_index = dataset.fields['src'].vocab.stoi[config['EOS_TOKEN']]

        src, src_lengths = src  # B * SrcSeqLen, B
        # TODO_DONE: remove first sos token, it makes results worse
        # src = src[1:]
        # src_lengths = src_lengths - 1
        self.ids = ids
        self.original_ids = original_ids
        self.oov_vocab, self.oov_indices = Batch.create_oov_vocab(ids, dataset, config)
        self.oov_vocab_reverse = {value: key for key, value in self.oov_vocab.items()}
        self.oov_num = len(self.oov_vocab)
        self.scatter_indices = Batch.create_scatter_indices(src, ids, self.oov_indices, self.pad_index, dataset)

        self.diff_alignment, self.diff_alignment_lengths = diff_alignment  # B * SeqAlignedLen, B
        self.diff_alignment_mask = (self.diff_alignment != self.pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_prev, self.diff_prev_lengths = diff_prev  # B * SeqAlignedLen, B
        self.diff_prev_mask = (self.diff_prev != self.pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_updated, self.diff_updated_lengths = diff_updated  # B * SeqAlignedLen, B
        self.diff_updated_mask = (self.diff_updated != self.pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen

        self.src = src  # B * SrcSeqLen
        self.src_lengths = src_lengths  # B
        self.src_mask = (src != self.pad_index).unsqueeze(-2)  # B * 1 * SrcSeqLen
        self.nseqs = src.size(0)

        self.hunk_numbers = (self.src == self.hunk_index).sum(dim=-1)
        self.diff_alignment_hunks, self.diff_alignment_hunk_lengths = \
            split_sequences_on_hunks(self.diff_alignment, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index, config['DEVICE'])
        self.diff_prev_hunks, self.diff_prev_hunk_lengths = \
            split_sequences_on_hunks(self.diff_prev, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index, config['DEVICE'])
        self.diff_updated_hunks, self.diff_updated_hunk_lengths = \
            split_sequences_on_hunks(self.diff_updated, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index, config['DEVICE'])
        self.src_hunks, self.src_hunk_lengths = \
            split_sequences_on_hunks(self.src, self.hunk_index, self.pad_index,
                                     self.sos_index, self.eos_index, config['DEVICE'])

        self.trg = None
        self.trg_y = None
        self.trg_y_extended_vocab = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]  # B * (TrgSeqLen - 1), removing eos from sequences
            self.trg_lengths = trg_lengths  # B
            self.trg_y = trg[:, 1:]  # B * (TrgSeqLen - 1), removing sos from sequences
            self.trg_mask = (self.trg_y != self.pad_index)  # B * (TrgSeqLen - 1)
            self.trg_y_extended_vocab = Batch.get_extended_target(self.trg_y, ids, dataset, self.oov_vocab)
            self.ntokens = (self.trg_y != self.pad_index).data.sum().item()

        self.src = self.src.to(config['DEVICE'])
        self.src_mask = self.src_mask.to(config['DEVICE'])

        if trg is not None:
            self.trg = self.trg.to(config['DEVICE'])
            self.trg_y = self.trg_y.to(config['DEVICE'])
            self.trg_y_extended_vocab = self.trg_y_extended_vocab.to(config['DEVICE'])
            self.trg_mask = self.trg_mask.to(config['DEVICE'])

    def __len__(self) -> int:
        return self.nseqs


def rebatch(batch: torchtext.data.Batch, dataset: Dataset, config: Config) -> Batch:
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    # These fields are added dynamically by PyTorch
    return Batch(batch.src, batch.trg, batch.diff_alignment,
                 batch.diff_prev, batch.diff_updated, batch.ids, batch.original_ids, dataset, config)
