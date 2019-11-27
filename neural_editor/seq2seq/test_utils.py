from typing import List

import matplotlib.pyplot as plt


def plot_perplexity(perplexities: List[float], labels: List[str]) -> None:
    """plot perplexities"""
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    for perplexity_values, label in zip(perplexities, labels):
        plt.plot(perplexity_values, label=label)
        plt.legend()
    plt.show()
