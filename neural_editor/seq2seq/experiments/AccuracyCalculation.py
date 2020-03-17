from typing import List

from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.train_utils import rebatch, calculate_top_k_accuracy, \
    create_greedy_decode_method_with_batch_support


class AccuracyCalculation:
    def __init__(self, model: EncoderDecoder, target_field: Field, max_len: int, greedy: bool, config: Config) -> None:
        super().__init__()
        self.model = model
        self.trg_vocab: Vocab = target_field.vocab
        self.pad_index: int = self.trg_vocab.stoi[config['PAD_TOKEN']]
        sos_index: int = self.trg_vocab.stoi[config['SOS_TOKEN']]
        self.eos_index: int = self.trg_vocab.stoi[config['EOS_TOKEN']]
        self.config = config
        self.beam_size = self.config['BEAM_SIZE']
        self.topk_values = [1] if greedy else self.config['TOP_K']
        if greedy:
            self.decode_method = create_greedy_decode_method_with_batch_support(
                self.model, max_len, sos_index, self.eos_index,
                self.trg_vocab.unk_index, len(self.trg_vocab)
            )
        else:
            self.decode_method = create_decode_method(
                self.model, max_len, sos_index, self.eos_index,
                self.trg_vocab.unk_index, len(self.trg_vocab), self.beam_size,
                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                verbose=False
            )
        self.batch_size = self.config['TEST_BATCH_SIZE'] if greedy else 1
        self.greedy = greedy

    def conduct(self, dataset: Dataset, dataset_label: str) -> List[List[List[str]]]:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...')
        if self.greedy:
            # TODO: deal with iterators and shuffling and BLEU calculation
            data_iterator = data.Iterator(dataset, batch_size=self.batch_size, train=False,
                                          shuffle=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          device=self.config['DEVICE'])
        else:
            data_iterator = data.Iterator(dataset, batch_size=self.batch_size,
                                          sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        correct_all_k, total, max_top_k_predicted = \
            calculate_top_k_accuracy(self.topk_values,
                                     [rebatch(self.pad_index, batch, dataset, self.config) for batch in data_iterator],
                                     self.decode_method, self.trg_vocab, self.eos_index)
        for correct_top_k, k in zip(correct_all_k, self.topk_values):
            print(f'Top-{k} accuracy: {correct_top_k} / {total} = {correct_top_k / total}')
        return max_top_k_predicted
