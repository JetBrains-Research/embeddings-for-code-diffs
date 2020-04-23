import os
from typing import List

from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch, rebatch_iterator
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.train_utils import calculate_top_k_accuracy, create_greedy_decode_method_top_k_edits, \
    create_greedy_decode_method


def save_predicted(max_top_k_predicted: List[List[List[str]]], dataset_name: str, k: int, config: Config) -> None:
    root_to_save = os.path.join(config['OUTPUT_PATH'], 'predictions')
    if not os.path.isdir(root_to_save):
        os.mkdir(root_to_save)

    top_k_file_lines = []
    for predictions in max_top_k_predicted:
        top_k_file_lines.append('====NEW EXAMPLE====')
        for prediction in predictions[:k]:
            top_k_file_lines.append(' '.join(prediction))
    top_k_file_lines.append('========END========')

    top_k_path = os.path.join(root_to_save, f'{dataset_name}_predicted_top_{k}.txt')
    with open(top_k_path, 'w') as top_k_file:
        top_k_file.write('\n'.join(top_k_file_lines))


class AccuracyCalculation:
    def __init__(self, model: EncoderDecoder, diffs_field: Field, train_dataset: Dataset, config: Config) -> None:
        super().__init__()
        self.model = model
        self.vocab: Vocab = diffs_field.vocab
        self.pad_index: int = self.vocab.stoi[config['PAD_TOKEN']]
        sos_index: int = self.vocab.stoi[config['SOS_TOKEN']]
        self.eos_index: int = self.vocab.stoi[config['EOS_TOKEN']]
        self.config = config
        self.beam_size = self.config['BEAM_SIZE']
        self.topk_values = self.config['TOP_K']
        self.max_top_k = max(self.topk_values)
        num_iterations = self.config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
        self.train_dataset = train_dataset
        self.beam_search = create_decode_method(self.model, num_iterations, sos_index, self.eos_index, self.beam_size,
                                                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                                                verbose=False)
        self.greedy_decode_top_k = create_greedy_decode_method_top_k_edits(self.model, num_iterations,
                                                                           sos_index, self.eos_index,
                                                                           self.max_top_k)
        self.greedy_decode = create_greedy_decode_method(self.model, num_iterations, sos_index, self.eos_index)

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...', flush=True)
        dataset_name = '_'.join(dataset_label.lower().split())
        data_iterator = rebatch_iterator(
            data.Iterator(dataset, batch_size=1, train=False,
                          sort=False, shuffle=False, sort_within_batch=False, repeat=False,
                          device=self.config['DEVICE']), self.pad_index, self.config)
        batched_data_iterator = rebatch_iterator(
            data.Iterator(dataset, batch_size=self.config['BATCH_SIZE'], train=False,
                          sort=False,
                          shuffle=False,
                          sort_within_batch=True,
                          sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                          device=self.config['DEVICE']), self.pad_index, self.config)

        self.model.set_training_data(self.train_dataset, self.pad_index)
        self.measure_performance(batched_data_iterator, dataset_name, 'top_k_edits_greedy_bug_fixing',
                                 self.greedy_decode_top_k, len(dataset))
        self.measure_performance(data_iterator, dataset_name, 'bug_fixing', self.beam_search, len(dataset))
        self.model.unset_training_data()

        self.measure_performance(batched_data_iterator, dataset_name, 'greedy_default_objective',
                                 self.greedy_decode, len(dataset))
        self.measure_performance(data_iterator, dataset_name, 'default_objective', self.beam_search, len(dataset))

    def measure_performance(self, data_iterator, dataset_name, method_name, decode_method, dataset_len):
        print(f'{method_name} ACCURACY')
        max_top_k_predicted = self.run(data_iterator, decode_method, dataset_len)
        save_predicted(max_top_k_predicted, f'{dataset_name}_{method_name}', k=1, config=self.config)

    def run(self, data_iterator: data.Iterator, decode_method, dataset_len) -> List[List[List[str]]]:
        correct_all_k, total, max_top_k_predicted = \
            calculate_top_k_accuracy(self.topk_values,
                                     data_iterator, decode_method,
                                     self.vocab, self.eos_index, dataset_len)
        for correct_top_k, k in zip(correct_all_k, self.topk_values):
            print(f'Top-{k} accuracy: {correct_top_k} / {total} = {correct_top_k / total}')
        return max_top_k_predicted
