import pprint
from typing import Tuple, List

import torch
from torch import nn
from torchtext import data
from torchtext.data import Field, Dataset

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.SimpleLossCompute import SimpleLossCompute
from neural_editor.seq2seq.datasets.CodeChangesDataset import CodeChangesTokensDataset
from neural_editor.seq2seq.datasets.dataset_utils import load_datasets
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import print_data_info, make_model, \
    run_epoch, rebatch, print_examples, plot_perplexity


def load_data(verbose: bool) -> Tuple[Dataset, Dataset, Dataset, Field]:
    diffs_field: Field = data.Field(batch_first=True, lower=CONFIG['LOWER'], include_lengths=True,
                                    unk_token=CONFIG['UNK_TOKEN'], pad_token=CONFIG['PAD_TOKEN'],
                                    init_token=CONFIG['SOS_TOKEN'],
                                    eos_token=CONFIG['EOS_TOKEN'])  # TODO: init_token=None?
    train_data, val_data, test_data = load_datasets(CodeChangesTokensDataset,
                                                    CONFIG['DATASET_ROOT'], diffs_field,
                                                    filter_pred=lambda x: len(vars(x)['src']) <= CONFIG[
                                                        'TOKENS_CODE_CHUNK_MAX_LEN'] and
                                                                          len(vars(x)['trg']) <= CONFIG[
                                                                              'TOKENS_CODE_CHUNK_MAX_LEN'])
    # TODO: consider building 2 vocabularies: one for (src, trg), second for diffs
    diffs_field.build_vocab(train_data.src, train_data.trg,
                            train_data.diff_alignment, train_data.diff_prev,
                            train_data.diff_updated, min_freq=CONFIG['TOKEN_MIN_FREQ'])
    if verbose:
        print_data_info(train_data, val_data, test_data, diffs_field)
    return train_data, val_data, test_data, diffs_field


def train(model: EncoderDecoder,
          train_data: Dataset, val_data: Dataset, diffs_field: Field,
          print_every: int) -> List[float]:
    """Train a model on LearningToRepresentEdits"""
    # TODO: add early stopping and choosing best model on eval
    train_iter = data.BucketIterator(train_data, batch_size=CONFIG['BATCH_SIZE'], train=True,
                                     shuffle=False,  # TODO: make shuffle reproducible
                                     sort_within_batch=True,
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=CONFIG['DEVICE'])
    val_iter = data.Iterator(val_data, batch_size=1, train=False, sort=False, repeat=False,
                             device=CONFIG['DEVICE'])

    # optionally add label smoothing; see the Annotated Transformer
    # TODO: why it is 0, maybe padding doesn't work because no tokenizing
    pad_index: int = diffs_field.vocab.stoi[CONFIG['PAD_TOKEN']]
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    dev_perplexities = []

    batches_num: int = len(train_iter)
    for epoch in range(CONFIG['MAX_NUM_OF_EPOCHS']):
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(pad_index, b) for b in train_iter),
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optimizer),
                                     batches_num,
                                     print_every=print_every)

        model.eval()
        with torch.no_grad():
            print_examples((rebatch(pad_index, x) for x in val_iter),
                           model, CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'] + 10,
                           diffs_field.vocab, n=3)

            dev_perplexity = run_epoch((rebatch(pad_index, t) for t in val_iter),
                                       model, SimpleLossCompute(model.generator, criterion, None),
                                       batches_num, print_every=print_every)
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)

    return dev_perplexities


def run_experiment() -> None:
    pprint.pprint(CONFIG)
    train_dataset, val_dataset, test_dataset, diffs_field = load_data(verbose=CONFIG['VERBOSE'])
    model: EncoderDecoder = make_model(len(diffs_field.vocab),
                                       edit_representation_size=CONFIG['EDIT_REPRESENTATION_SIZE'],
                                       emb_size=CONFIG['WORD_EMBEDDING_SIZE'],
                                       hidden_size_encoder=CONFIG['ENCODER_HIDDEN_SIZE'],
                                       hidden_size_decoder=CONFIG['DECODER_HIDDEN_SIZE'],
                                       num_layers=CONFIG['NUM_LAYERS'],
                                       dropout=CONFIG['DROPOUT'],
                                       use_bridge=CONFIG['USE_BRIDGE'])
    # TODO: fix PyCharm highlighting
    dev_perplexities = train(model, train_dataset, val_dataset, diffs_field,
                             print_every=CONFIG['PRINT_EVERY_iTH_BATCH'])
    plot_perplexity(dev_perplexities)


if __name__ == "__main__":
    run_experiment()
