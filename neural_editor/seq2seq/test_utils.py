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

from datasets.dataset_utils import create_filter_predicate_on_length
from neural_editor.seq2seq import EncoderDecoder
from datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.train_utils import print_examples, calculate_accuracy
from neural_editor.seq2seq.Batch import rebatch


def save_predicted(max_top_k_predicted: List[List[List[str]]], dataset_name: str, config: Config) -> None:
    top_1_file_lines = []
    top_k_file_lines = []
    max_k = config['TOP_K'][-1]
    for predictions in max_top_k_predicted:
        top_1_file_lines.append("" if len(predictions) == 0 else ' '.join(predictions[0]))
        top_k_file_lines.append('====NEW EXAMPLE====')
        for prediction in predictions[:max_k]:
            top_k_file_lines.append(' '.join(prediction))

    top_1_path = os.path.join(config['OUTPUT_PATH'], f'{dataset_name}_predicted_top_1.txt')
    top_k_path = os.path.join(config['OUTPUT_PATH'], f'{dataset_name}_predicted_top_{max_k}.txt')
    with open(top_1_path, 'w') as top_1_file, open(top_k_path, 'w') as top_k_file:
        top_1_file.write('\n'.join(top_1_file_lines))
        top_k_file.write('\n'.join(top_k_file_lines))


def map_classes_to_colors(classes: List[str]) -> Tuple[List[int], Dict[int, str]]:
    dictionary = {}
    colors = []
    for cls in classes:
        if cls not in dictionary:
            dictionary[cls] = len(dictionary)
        colors.append(dictionary[cls])
    return colors, dict([(value, key) for key, value in dictionary.items()])


def save_metrics_plot(metrics: List[List[float]], labels: List[str], title: str, filepath: str, config: Config,
                      xlabel='', ylabel='') \
        -> None:
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for values, label in zip(metrics, labels):
        plt.plot(values, label=label)
        plt.legend()
    plt.savefig(os.path.join(config['OUTPUT_PATH'], filepath))
    plt.clf()


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


def save_metric_plot(metric_values: List[float], label: str, filepath: str, config: Config) -> None:
    """plot a metrics for each epoch"""
    plt.title(f"{label} per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.plot(metric_values)
    plt.savefig(os.path.join(config['OUTPUT_PATH'], filepath))
    plt.clf()


def save_patchnet_metric_plot(metric_values: List[float], label: str, root: str) -> None:
    plt.title(f"{label} per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.plot(metric_values)
    plt.savefig(os.path.join(root, f'{label}.png'))
    plt.clf()


def load_defects4j_dataset(diffs_field: Field, config: Config) -> Tuple[Dataset, List[str]]:
    filter_predicate = create_filter_predicate_on_length(config['TOKENS_CODE_CHUNK_MAX_LEN'])
    dataset = CodeChangesTokensDataset(config['DEFECTS4J_PATH'], diffs_field, config, filter_predicate)
    classes = Path(config['DEFECTS4J_PATH']).joinpath('classes.txt').read_text().splitlines(keepends=False)
    return dataset, classes


def load_labeled_dataset(path: str, diffs_field: Field, config: Config) -> Tuple[Dataset, List[str]]:
    filter_predicate = create_filter_predicate_on_length(config['TOKENS_CODE_CHUNK_MAX_LEN'])
    dataset = CodeChangesTokensDataset(path, diffs_field, config, filter_predicate)
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
        print_examples((rebatch(x, defects4j_data, config) for x in print_examples_iterator),
                       model, config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1,
                       vocab, vocab, config, n=len(print_examples_iterator))
        accuracy_iterator = data.Iterator(defects4j_data, batch_size=config['TEST_BATCH_SIZE'], train=False,
                                          sort_within_batch=True,
                                          sort_key=lambda x: (len(x.src), len(x.trg)),
                                          repeat=False,
                                          device=config['DEVICE'])
        accuracy = calculate_accuracy((rebatch(t, defects4j_data, config) for t in accuracy_iterator),
                                      model,
                                      config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1,
                                      vocab, config)
        print(f'Accuracy on Defects4J: {accuracy}')
