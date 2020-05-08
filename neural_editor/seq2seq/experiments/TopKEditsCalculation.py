import os
from typing import List, Tuple, Iterator, Optional

from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab
import numpy as np

from edit_representation.sequence_encoding.Differ import Differ
from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch, rebatch_iterator
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.experiments.AccuracyCalculation import save_predicted
from neural_editor.seq2seq.experiments.NearestNeighbor import create_levenshtein_metric
from neural_editor.seq2seq.train_utils import calculate_top_k_accuracy, create_greedy_decode_method_top_k_edits, \
    create_greedy_decode_method, create_greedy_decode_method_top_k_edits_with_indices, remove_eos, lookup_words


class TopKEditsCalculation:
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

        self.greedy_decode_top_k = create_greedy_decode_method_top_k_edits_with_indices(self.model, num_iterations,
                                                                                        sos_index, self.eos_index,
                                                                                        self.max_top_k)

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting top-k edits calculation experiment for {dataset_label}...', flush=True)
        dataset_name = '_'.join(dataset_label.lower().split())
        data_iterator = rebatch_iterator(
            data.Iterator(dataset, batch_size=1, train=False,
                          sort=False, shuffle=False, sort_within_batch=False, repeat=False,
                          device=self.config['DEVICE']), self.pad_index, self.config)

        self.model.set_training_data(self.train_dataset, self.pad_index)
        indices, correct_mask, max_top_k_predicted = \
            self.calculate_top_k_correct(data_iterator, len(dataset))
        save_predicted(max_top_k_predicted, f'{dataset_name}_top_k_edits_calculation', k=self.max_top_k, config=self.config)
        print(f'Test examples: {[example.ids for example in dataset.examples]}')
        print(f'Nearest neighbors:\n{indices}')
        print(f'Correct nearest neighbors:\n')
        for i, example_correct_mask in enumerate(correct_mask):
            print(f'{dataset.examples[i].ids}: {indices[i][example_correct_mask]}')
        k = 3
        for i in range(len(dataset)):
            print('\n====NEW TEST EXAMPLE====')
            print(f'Test example {dataset.examples[i].ids}')
            print(f'prev:')
            print(f'{" ".join(dataset.examples[i].src)}')
            print(f'updated:')
            print(f'{" ".join(dataset.examples[i].trg)}')
            print(f'\nprevs and updateds in train by top-{k} nearest vectors')
            for train_idx in indices[i][:k]:
                print(f'{" ".join(self.train_dataset.examples[train_idx].src)}')
                print(f'{" ".join(self.train_dataset.examples[train_idx].trg)}')
                print()
            print(f'\nprevs and updateds in train by correct nearest vectors')
            distances_correct = [idx_in_indices for idx_in_indices in range(len(indices[i])) if correct_mask[i][idx_in_indices]]
            print(f'distances for correct: {distances_correct}')
            for train_correct_idx in indices[i][correct_mask[i]]:
                print(f'{" ".join(self.train_dataset.examples[train_correct_idx].src)}')
                print(f'{" ".join(self.train_dataset.examples[train_correct_idx].trg)}')
                print()
        self.model.unset_training_data()

    def calculate_top_k_correct(self, dataset_iterator: Iterator, dataset_len: int) -> Tuple[np.ndarray, np.ndarray, List[List[List[str]]]]:
        max_top_k_results: List[Optional[List[List[str]]]] = [None] * dataset_len
        indices = np.full((dataset_len, self.max_top_k), -1)
        correct_mask = np.full(indices.shape, False)
        for example_id, batch in enumerate(dataset_iterator):
            targets = remove_eos(batch.trg_y.cpu().numpy(), self.eos_index)
            results, indices_in_batch = self.greedy_decode_top_k(batch)
            for example_idx_in_batch in range(len(results)):
                # example_id = batch.ids[example_idx_in_batch].item()
                target = targets[example_idx_in_batch]
                example_nearest_indices = indices_in_batch[example_idx_in_batch]
                indices[example_id] = example_nearest_indices
                example_top_k_results = results[example_idx_in_batch][:self.max_top_k]
                decoded_tokens = [lookup_words(result, self.vocab)
                                  for result in example_top_k_results]
                max_top_k_results[example_id] = decoded_tokens
                for i, result in enumerate(example_top_k_results):
                    correct_mask[example_id][i] = (len(result) == len(target) and np.all(result == target))
        return indices, correct_mask, max_top_k_results
