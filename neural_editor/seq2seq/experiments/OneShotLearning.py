from typing import List, Dict, Any

import torch
from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import rebatch, calculate_accuracy, print_examples


class OneShotLearning:
    def __init__(self, model: EncoderDecoder, diffs_field: Field) -> None:
        super().__init__()
        self.model = model
        self.vocab: Vocab = diffs_field.vocab
        self.pad_index: int = self.vocab.stoi[CONFIG['PAD_TOKEN']]

    def conduct(self, dataset: Dataset, classes: List[str], dataset_label: str) -> None:
        print(f'Start conducting one shot learning experiment for {dataset_label}...')
        data_iterator = data.Iterator(dataset, batch_size=1,
                                      sort=False, train=False, shuffle=False, device=CONFIG['DEVICE'])
        current_class = None
        correct_same_edit_representation = 0
        total_same_edit_representation = 0
        correct_other_edit_representation = 0
        total_other_edit_representation = 0

        was_training_before = self.model.training
        self.model.eval()
        with torch.no_grad():
            correct_examples = []
            incorrect_examples = []
            for i, batch in enumerate(data_iterator):
                batch = rebatch(self.pad_index, batch)
                y = classes[i]
                if y != current_class:
                    self.model.set_edit_representation(batch)
                    current_class = y
                    correct = calculate_accuracy([batch], self.model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'], self.vocab)
                    correct_same_edit_representation += correct
                    total_same_edit_representation += 1
                    is_correct = False if correct == 0 else True
                    correct_examples.append({'class': y, 'golden': (batch, is_correct), 'others': []})
                    incorrect_examples.append({'class': y, 'golden': (batch, is_correct), 'others': []})
                else:
                    correct = calculate_accuracy([batch], self.model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'], self.vocab)
                    correct_other_edit_representation += correct
                    total_other_edit_representation += 1
                    if correct == 0:
                        incorrect_examples[-1]['others'].append(batch)
                    else:
                        correct_examples[-1]['others'].append(batch)

        print(f'Accuracy on {dataset_label} for same edit representations: '
              f'{correct_same_edit_representation} / {total_same_edit_representation} = '
              f'{correct_same_edit_representation / total_same_edit_representation}')
        print(f'Accuracy on {dataset_label} for other edit representations: '
              f'{correct_other_edit_representation} / {total_other_edit_representation} = '
              f'{correct_other_edit_representation / total_other_edit_representation}')
        self.print_examples(correct_examples, is_correct=True)
        self.print_examples(incorrect_examples, is_correct=False)

        self.model.unset_edit_representation()
        self.model.training = was_training_before

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
            print_examples([golden_example[0]], self.model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'],
                           self.vocab, n=1, color='green' if golden_example[1] else 'red')
            print('+++++++++++++++')
            print_examples(class_correct_examples['others'], self.model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'],
                           self.vocab, n=len(class_correct_examples['others']), color='green' if is_correct else 'red')
            print('---------------')
            self.model.unset_edit_representation()
        print('================')
