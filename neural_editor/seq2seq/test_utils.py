import os
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.train_utils import print_examples, rebatch, calculate_accuracy


def map_classes_to_colors(classes: List[str]) -> Tuple[List[int], Dict[int, str]]:
    dictionary = {}
    colors = []
    for cls in classes:
        if cls not in dictionary:
            dictionary[cls] = len(dictionary)
        colors.append(dictionary[cls])
    return colors, dict([(value, key) for key, value in dictionary.items()])


def visualize_tsne(representations: torch.Tensor, classes: List[str], filename: str, config: Config) -> None:
    representations = representations.numpy()
    representations_2d = TSNE(n_components=2, init='pca', random_state=config['SEED']).fit_transform(representations)
    df = pd.DataFrame(dict(x=representations_2d[:, 0], y=representations_2d[:, 1], classes=classes))
    sns.lmplot('x', 'y', data=df, hue='classes' if classes is not None else None, fit_reg=False)
    plt.savefig(os.path.join(config['OUTPUT_PATH'], filename), bbox_inches='tight')
    plt.clf()


def save_perplexity_plot(perplexities: List[List[float]], labels: List[str], filepath: str, config: Config) -> None:
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
        plt.legend()
    plt.savefig(os.path.join(config['OUTPUT_PATH'], filepath))
    plt.clf()


def load_defects4j_dataset(diffs_field: Field, config: Config) -> Tuple[Dataset, List[str]]:
    dataset = CodeChangesTokensDataset(config['DEFECTS4J_PATH'], diffs_field,
                                       add_reverse_examples_ratio=0, config=config)
    classes = Path(config['DEFECTS4J_PATH']).joinpath('classes.txt').read_text().splitlines(keepends=False)
    return dataset, classes


def load_labeled_dataset(path: str, diffs_field: Field, config: Config) -> Tuple[Dataset, List[str]]:
    dataset = CodeChangesTokensDataset(path, diffs_field, add_reverse_examples_ratio=0, config=config)
    classes = Path(path).joinpath('classes.txt').read_text().splitlines(keepends=False)
    return dataset, classes


def output_accuracy_on_defects4j(model: EncoderDecoder, defects4j_data: Dataset,
                                 diffs_field: Field, config: Config) -> None:
    pad_index: int = diffs_field.vocab.stoi[config['PAD_TOKEN']]
    vocab: Vocab = diffs_field.vocab
    # TODO: delete eval and no_grad
    model.eval()
    with torch.no_grad():
        print_examples_iterator = data.Iterator(defects4j_data, batch_size=1, train=False, sort=False,
                                                repeat=False, device=config['DEVICE'])
        print(f'===Defects4J EXAMPLES===')
        print_examples((rebatch(pad_index, x, config) for x in print_examples_iterator),
                       model, config['TOKENS_CODE_CHUNK_MAX_LEN'],
                       vocab, config, n=len(print_examples_iterator))
        accuracy_iterator = data.Iterator(defects4j_data, batch_size=config['TEST_BATCH_SIZE'], train=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          repeat=False,
                                          device=config['DEVICE'])
        accuracy = calculate_accuracy((rebatch(pad_index, t, config) for t in accuracy_iterator),
                                      model,
                                      config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                      vocab, config)
        print(f'Accuracy on Defects4J: {accuracy}')
