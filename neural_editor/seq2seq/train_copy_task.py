import numpy as np
import torch
from torch import nn


from neural_editor.seq2seq.Batch import Batch
from neural_editor.seq2seq.SimpleLossCompute import SimpleLossCompute
from neural_editor.seq2seq.train_utils import USE_CUDA, make_model_mt, run_epoch, print_examples_mt, plot_perplexity


def data_gen(num_words=11, batch_size=16, num_batches=100, length=10, pad_index=0, sos_index=1):
    """Generate random data for a src-tgt copy task."""
    for i in range(num_batches):
        data = torch.from_numpy(
          np.random.randint(1, num_words, size=(batch_size, length)))
        data[:, 0] = sos_index
        data = data.cuda() if USE_CUDA else data
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


def train_copy_task():
    """Train the simple copy task."""
    num_words = 11
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model_mt(num_words, num_words, emb_size=32, hidden_size=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.0003)
    eval_data = list(data_gen(num_words=num_words, batch_size=1, num_batches=100))

    dev_perplexities = []

    if USE_CUDA:
        model.cuda()

    for epoch in range(10):
        print("Epoch %d" % epoch)

        # train
        model.train()
        data = data_gen(num_words=num_words, batch_size=32, num_batches=100)
        run_epoch(data, model,
                  SimpleLossCompute(model.generator, criterion, optim))

        # evaluate
        model.eval()
        with torch.no_grad():
            perplexity = run_epoch(eval_data, model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Evaluation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)
            print_examples_mt(eval_data, model, n=2, max_len=9)

    return dev_perplexities


if __name__ == "__main__":
    # train the copy task
    dev_perplexities = train_copy_task()
    plot_perplexity(dev_perplexities)
