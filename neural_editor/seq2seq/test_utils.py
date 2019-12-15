from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torchtext import data
from torchtext.data import Dataset, Field
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import print_examples, rebatch, calculate_accuracy


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
