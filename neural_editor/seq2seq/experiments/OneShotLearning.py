from typing import List, Dict, Any

from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
import numpy as np

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.BatchBeamSearch import BatchedBeamSearch
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.train_utils import rebatch, calculate_top_k_accuracy, \
    print_examples_decode_method


class OneShotLearning:
    def __init__(self, model: EncoderDecoder, diffs_field: Field, config: Config) -> None:
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
        self.beam_search = create_decode_method(self.model, num_iterations, sos_index, self.eos_index, self.beam_size,
                                                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                                                verbose=False)

    def conduct(self, dataset: Dataset, classes: List[str], dataset_label: str) -> None:
        print(f'Start conducting one shot learning experiment for {dataset_label}...')
        data_iterator = data.Iterator(dataset, batch_size=1,
                                      sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        current_class = None
        correct_top_k_same_edit_representation = np.array([0 for _ in range(len(self.topk_values))])
        total_same_edit_representation = 0
        correct_top_k_other_edit_representation = np.array([0 for _ in range(len(self.topk_values))])
        total_other_edit_representation = 0

        correct_examples = []
        incorrect_examples = []
        for i, batch in enumerate(data_iterator):
            batch = rebatch(self.pad_index, batch, self.config)
            y = classes[i]
            if y != current_class:
                self.model.set_edit_representation(batch)
                current_class = y
                correct_top_k, _ = calculate_top_k_accuracy(self.topk_values, [batch],
                                                            self.beam_search, self.eos_index)
                correct_top_k_same_edit_representation += correct_top_k
                total_same_edit_representation += 1
                is_correct = False if correct_top_k[0] == 0 else True
                correct_examples.append({'class': y, 'golden': (batch, is_correct), 'others': []})
                incorrect_examples.append({'class': y, 'golden': (batch, is_correct), 'others': []})
            else:
                correct_top_k, _ = calculate_top_k_accuracy(self.topk_values, [batch],
                                                            self.beam_search, self.eos_index)
                correct_top_k_other_edit_representation += correct_top_k
                total_other_edit_representation += 1
                if correct_top_k[0] == 0:
                    incorrect_examples[-1]['others'].append(batch)
                else:
                    correct_examples[-1]['others'].append(batch)

        for correct_top_specific_k, k in zip(correct_top_k_same_edit_representation, self.topk_values):
            print(f'Top-{k} accuracy on {dataset_label} for same edit representations: '
                  f'{correct_top_specific_k} / {total_same_edit_representation} = '
                  f'{correct_top_specific_k / total_same_edit_representation}')
        print()
        for correct_top_specific_k, k in zip(correct_top_k_other_edit_representation, self.topk_values):
            print(f'Top-{k} accuracy on {dataset_label} for other edit representations: '
                  f'{correct_top_specific_k} / {total_other_edit_representation} = '
                  f'{correct_top_specific_k / total_other_edit_representation}')
        self.print_examples(correct_examples, is_correct=True)
        self.print_examples(incorrect_examples, is_correct=False)

        self.model.unset_edit_representation()

    def print_examples(self, examples: List[Dict[str, Any]], is_correct: bool) -> None:
        print('================')
        print(f'{"Correct" if is_correct else "Incorrect"} Examples')
        for class_correct_examples in examples:
            if len(class_correct_examples['others']) == 0:
                continue

            print(f'Class: {class_correct_examples["class"]}')

            golden_example = class_correct_examples["golden"]
            print(f'Golden example ({golden_example[1]}):')
            self.model.set_edit_representation(golden_example[0])
            print_examples_decode_method([golden_example[0]], self.model, self.vocab, self.config,
                                         n=1, color='green' if golden_example[1] else 'red',
                                         decode_method=self.beam_search)
            print('+++++++++++++++')
            print_examples_decode_method(class_correct_examples['others'], self.model, self.vocab, self.config,
                                         n=len(class_correct_examples['others']),
                                         color='green' if is_correct else 'red',
                                         decode_method=self.beam_search)  # TODO: decode -> decode_several vs greedy decode
            print('---------------')
            self.model.unset_edit_representation()
        print('================')
