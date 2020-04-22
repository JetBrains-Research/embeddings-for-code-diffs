import os
from typing import List

from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.train_utils import calculate_top_k_accuracy


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
        num_iterations = self.config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
        self.train_dataset = train_dataset
        self.beam_search = create_decode_method(self.model, num_iterations, sos_index, self.eos_index, self.beam_size,
                                                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                                                verbose=False)

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...')
        dataset_name = '_'.join(dataset_label.lower().split())
        data_iterator = data.Iterator(dataset, batch_size=1,
                                      sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        print('BUG FIXING ACCURACY')
        self.model.set_training_data(self.train_dataset, self.pad_index)
        max_top_k_predicted_bug_fixing = self.run(data_iterator)
        save_predicted(max_top_k_predicted_bug_fixing, f'{dataset_name}_bug_fixing', k=1, config=self.config)
        self.model.unset_training_data()
        print('TRAINING OBJECTIVE ACCURACY')
        max_top_k_predicted_default = self.run(data_iterator)
        save_predicted(max_top_k_predicted_default, f'{dataset_name}_default', k=1, config=self.config)

    def run(self, data_iterator: data.Iterator) -> List[List[List[str]]]:
        data_iterator = [rebatch(self.pad_index, batch, self.config) for batch in data_iterator]
        correct_all_k, total, max_top_k_predicted = \
            calculate_top_k_accuracy(self.topk_values,
                                     data_iterator, self.beam_search,
                                     self.vocab, self.eos_index, len(data_iterator))
        for correct_top_k, k in zip(correct_all_k, self.topk_values):
            print(f'Top-{k} accuracy: {correct_top_k} / {total} = {correct_top_k / total}')
        return max_top_k_predicted
