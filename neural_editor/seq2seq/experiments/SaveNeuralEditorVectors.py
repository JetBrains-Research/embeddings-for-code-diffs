import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from torchtext import data
from torchtext.data import Field, Dataset

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.Batch import rebatch
from neural_editor.seq2seq.config import Config


class SaveNeuralEditorVectors:
    def __init__(self, model: EncoderDecoder, diffs_field: Field, config: Config) -> None:
        super().__init__()
        self.model = model
        self.pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
        self.config = config
        self.batch_size = self.config['TEST_BATCH_SIZE']
        self.commit_hashes = [l.split(':')[0]
                              for l in Path(self.config['COMMIT_HASHES_PATH']).read_text().splitlines(keepends=False)] \
            if self.config['COMMIT_HASHES_PATH'] is not None else None
        self.output_path = Path(self.config['OUTPUT_PATH'])

    def conduct(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, dataset_label: str) -> None:
        print(f'Start saving neural editor vectors experiment for {dataset_label}...', flush=True)
        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        dataset_iterators = {}
        for data_type, dataset in datasets.items():
            data_iterator = data.Iterator(dataset, batch_size=self.batch_size, train=False,
                                          shuffle=False,
                                          sort=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          device=self.config['DEVICE'])
            dataset_iterators[data_type] = [rebatch(batch, dataset, self.config) for batch in data_iterator]

        dataset_dicts = self.get_dataset_dicts(dataset_iterators)
        self.save_dataset_dicts(dataset_dicts)

    def get_dataset_dicts(self, dataset_iterators):
        dataset_dicts = {'src': defaultdict(lambda: {}), 'edit': defaultdict(lambda: {}),
                         'both': defaultdict(lambda: {})}
        for name, dataset_iterator in dataset_iterators.items():
            for i, batch in enumerate(dataset_iterator):
                (edit_hidden, _), _, (encoder_hidden, _) = self.model.encode(batch)
                edit_hidden = edit_hidden[-1].detach().cpu().numpy()
                encoder_hidden = encoder_hidden[-1].detach().cpu().numpy()
                both_hidden = np.concatenate((edit_hidden, encoder_hidden), axis=-1)
                for j in range(len(batch)):
                    original_id = batch.original_ids[j]
                    commit_hash = self.commit_hashes[original_id] if self.commit_hashes is not None else str(original_id.item())
                    dataset_dicts['edit'][name][commit_hash] = edit_hidden[j]
                    dataset_dicts['src'][name][commit_hash] = encoder_hidden[j]
                    dataset_dicts['both'][name][commit_hash] = both_hidden[j]

        for vector_type in dataset_dicts:
            train_dict = dataset_dicts[vector_type]['train']
            val_dict = dataset_dicts[vector_type]['val']
            test_dict = dataset_dicts[vector_type]['test']
            dataset_dicts[vector_type]['train_val'] = {**train_dict, **val_dict}
            dataset_dicts[vector_type]['all'] = {**train_dict, **val_dict, **test_dict}

        return dataset_dicts

    def save_dataset_dicts(self, dataset_dicts):
        path1 = self.output_path.joinpath('ne_vectors')
        path1.mkdir(exist_ok=True)
        for vector_type in dataset_dicts:
            path2 = path1.joinpath(vector_type)
            path2.mkdir(exist_ok=True)
            for data_type in dataset_dicts[vector_type]:
                path3 = path2.joinpath(data_type)
                path3.mkdir(exist_ok=True)
                pickle.dump(dataset_dicts[vector_type][data_type], path3.joinpath('ne_vectors.pkl').open(mode='wb'))
                pickle.dump(dataset_dicts[vector_type][data_type], path3.joinpath('ne_vectors_2.pkl').open(mode='wb'),
                            protocol=2)
