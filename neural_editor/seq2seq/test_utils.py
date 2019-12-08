from typing import List, Generator

import matplotlib.pyplot as plt
from torchtext.vocab import Vocab
import numpy as np
from tqdm.auto import tqdm

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import greedy_decode


def plot_perplexity(perplexities: List[float], labels: List[str]) -> None:
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
        plt.legend()
    plt.show()


def calculate_accuracy(dataset_iterator: List,
                       model: EncoderDecoder,
                       max_len: int,
                       vocab: Vocab) -> float:
    sos_index = vocab.stoi[CONFIG['SOS_TOKEN']]
    eos_index = vocab.stoi[CONFIG['EOS_TOKEN']]

    correct = 0
    total = 0
    for batch in tqdm(dataset_iterator):
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        trg = trg[:-1] if trg[-1] == eos_index else trg

        result, _ = greedy_decode(model, batch, max_len, sos_index, eos_index)
        if np.all(trg == result):
            print('Found one correct')
            correct += 1
        total += 1
    return correct / total
