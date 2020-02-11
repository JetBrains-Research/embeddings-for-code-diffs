from typing import Tuple, Dict, List

import torch
from torch import Tensor
from torchtext.data import Dataset

from neural_editor.seq2seq.config import Config


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    @staticmethod
    def create_oov_vocab(ids: Tensor, dataset: Dataset, config: Config) -> Tuple[Dict[str, int], Tensor]:
        vocab = dataset.fields['src'].vocab
        examples = [dataset[i] for i in ids]
        src_oov_vocab = {}
        cur_id = len(vocab)
        oov_indices = []
        for example in examples:
            for src_token in example.src:
                if vocab.stoi[src_token] == vocab.unk_index:
                    if src_token not in src_oov_vocab:
                        src_oov_vocab[src_token] = cur_id
                        cur_id += 1
                    oov_indices.append(src_oov_vocab[src_token])
        return src_oov_vocab, torch.tensor(oov_indices).long().to(config['DEVICE'])

    @staticmethod
    def get_extended_target(trg_y: Tensor, ids: Tensor, dataset: Dataset, oov_vocab: Dict[str, int]) -> Tensor:
        unk_index = dataset.fields['src'].vocab.unk_index
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
    def create_scatter_indices(src: Tensor, oov_indices: Tensor, dataset: Dataset) -> Tensor:
        unk_index = dataset.fields['src'].vocab.unk_index
        scatter_indices = src.clone().detach()
        scatter_indices[scatter_indices == unk_index] = oov_indices
        return scatter_indices

    def __init__(self, src: Tuple[Tensor, Tensor], trg: Tuple[Tensor, Tensor],
                 diff_alignment: Tuple[Tensor, Tensor],
                 diff_prev: Tuple[Tensor, Tensor], diff_updated: Tuple[Tensor, Tensor],
                 ids: Tensor, dataset: Dataset,
                 pad_index: int, config: Config) -> None:
        src, src_lengths = src  # B * SrcSeqLen, B
        # TODO: remove first sos token
        # src = src[1:]
        # src_lengths = src_lengths - 1

        self.oov_vocab, self.oov_indices = Batch.create_oov_vocab(ids, dataset, config)
        self.oov_vocab_reverse = {value: key for key, value in self.oov_vocab.items()}
        self.oov_num = len(self.oov_vocab)
        self.scatter_indices = Batch.create_scatter_indices(src, self.oov_indices, dataset)

        self.diff_alignment, self.diff_alignment_lengths = diff_alignment  # B * SeqAlignedLen, B
        self.diff_alignment_mask = (self.diff_alignment != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_prev, self.diff_prev_lengths = diff_prev  # B * SeqAlignedLen, B
        self.diff_prev_mask = (self.diff_prev != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen
        self.diff_updated, self.diff_updated_lengths = diff_updated  # B * SeqAlignedLen, B
        self.diff_updated_mask = (self.diff_updated != pad_index).unsqueeze(-2)  # B * 1 * SeqAlignedLen

        self.src = src  # B * SrcSeqLen
        self.src_lengths = src_lengths  # B
        self.src_mask = (src != pad_index).unsqueeze(-2)  # B * 1 * SrcSeqLen
        self.nseqs = src.size(0)

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
            self.trg_mask = (self.trg_y != pad_index)  # B * (TrgSeqLen - 1)
            self.trg_y_extended_vocab = Batch.get_extended_target(self.trg_y, ids, dataset, self.oov_vocab)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        self.src = self.src.to(config['DEVICE'])
        self.src_mask = self.src_mask.to(config['DEVICE'])

        if trg is not None:
            self.trg = self.trg.to(config['DEVICE'])
            self.trg_y = self.trg_y.to(config['DEVICE'])
            self.trg_y_extended_vocab = self.trg_y_extended_vocab.to(config['DEVICE'])
            self.trg_mask = self.trg_mask.to(config['DEVICE'])

    def __len__(self) -> int:
        return self.nseqs
