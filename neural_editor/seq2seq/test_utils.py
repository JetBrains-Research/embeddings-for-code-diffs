from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import print_examples, rebatch, calculate_accuracy


def map_classes_to_colors(classes: List[str]) -> Tuple[List[int], Dict[int, str]]:
    dictionary = {}
    colors = []
    for cls in classes:
        if cls not in dictionary:
            dictionary[cls] = len(dictionary)
        colors.append(dictionary[cls])
    return colors, dict([(value, key) for key, value in dictionary.items()])


def visualize_tsne(representations: torch.Tensor, classes: List[str]) -> None:
    representations = PCA(n_components=20).fit_transform(representations.numpy())
    representations_2d = TSNE(n_components=2).fit_transform(representations)
    df = pd.DataFrame(dict(x=representations_2d[:, 0], y=representations_2d[:, 1], classes=classes))
    sns.lmplot('x', 'y', data=df, hue='classes' if classes is not None else None, fit_reg=False)
    plt.show()


def plot_perplexity(perplexities: List[float], labels: List[str]) -> None:
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
        plt.legend()
    plt.show()


def load_defects4j_dataset(diffs_field: Field) -> Tuple[Dataset, List[str]]:
    dataset = CodeChangesTokensDataset(CONFIG['DEFECTS4J_PATH'], diffs_field)
    classes = Path(CONFIG['DEFECTS4J_PATH']).joinpath('classes.txt').read_text().splitlines(keepends=False)
    return dataset, classes


def output_accuracy_on_defects4j(model: EncoderDecoder, defects4j_data: Dataset, diffs_field: Field) -> None:
    pad_index: int = diffs_field.vocab.stoi[CONFIG['PAD_TOKEN']]
    vocab: Vocab = diffs_field.vocab
    model.eval()
    with torch.no_grad():
        print_examples_iterator = data.Iterator(defects4j_data, batch_size=1, train=False, sort=False,
                                                repeat=False, device=CONFIG['DEVICE'])
        print(f'===Defects4J EXAMPLES===')
        print_examples((rebatch(pad_index, x) for x in print_examples_iterator),
                       model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'],
                       vocab, n=len(print_examples_iterator))
        accuracy_iterator = data.Iterator(defects4j_data, batch_size=CONFIG['TEST_BATCH_SIZE'], train=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          repeat=False,
                                          device=CONFIG['DEVICE'])
        accuracy = calculate_accuracy((rebatch(pad_index, t) for t in accuracy_iterator),
                                      model,
                                      CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'],
                                      vocab)
        print(f'Accuracy on Defects4J: {accuracy}')
