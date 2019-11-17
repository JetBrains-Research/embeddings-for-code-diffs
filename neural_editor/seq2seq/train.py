# For data loading.
import torch
from torch import nn
from torchtext import data
import pprint

from neural_editor.seq2seq.SimpleLossCompute import SimpleLossCompute
from neural_editor.seq2seq.datasets.LearningToRepresentEditsJson import LearningToRepresentEditsTokenStrings
from neural_editor.seq2seq.train_config import CONFIG
from neural_editor.seq2seq.train_utils import print_data_info, make_model, \
    run_epoch, rebatch, print_examples, plot_perplexity


def load_data(verbose=False):
    diffs_field = data.Field(batch_first=True, lower=CONFIG['LOWER'], include_lengths=True,
                             unk_token=CONFIG['UNK_TOKEN'], pad_token=CONFIG['PAD_TOKEN'],
                             init_token=CONFIG['SOS_TOKEN'], eos_token=CONFIG['EOS_TOKEN'])  # TODO: init_token=None?
    train_data, val_data, test_data = LearningToRepresentEditsTokenStrings.splits(
        CONFIG['DATASET_ROOT'], diffs_field,
        filter_pred=lambda x: len(vars(x)['src']) <= CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'] and
                              len(vars(x)['trg']) <= CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'])
    # TODO: consider building 2 vocabularies: one for (src, trg), second for diffs
    diffs_field.build_vocab(train_data.src, train_data.trg,
                            train_data.diff_alignment, train_data.diff_prev,  # TODO: try to delete alignment
                            train_data.diff_updated, min_freq=CONFIG['TOKEN_MIN_FREQ'])
    if verbose:
        print_data_info(train_data, val_data, test_data, diffs_field)
    return train_data, val_data, test_data, diffs_field


def train(model, train_data, val_data, diffs_field, print_every=100):
    """Train a model on LearningToRepresentEdits"""
    # TODO: add early stopping and choosing best model on eval
    train_iter = data.BucketIterator(train_data, batch_size=64, train=True,
                                     shuffle=False, sort_within_batch=True,
                                     sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                     device=CONFIG['DEVICE'])
    val_iter = data.Iterator(val_data, batch_size=1, train=False, shuffle=False, sort=False, repeat=False,
                             device=CONFIG['DEVICE'])

    if CONFIG['USE_CUDA']:
        model.cuda()

    # optionally add label smoothing; see the Annotated Transformer
    # TODO: why it is 0, maybe padding doesn't work because no tokenizing
    pad_index = diffs_field.vocab.stoi[CONFIG['PAD_TOKEN']]
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optim = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

    dev_perplexities = []

    batches_num = len(train_iter)
    for epoch in range(CONFIG['MAX_NUM_OF_EPOCHS']):
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch((rebatch(pad_index, b) for b in train_iter),
                                     model,
                                     SimpleLossCompute(model.generator, criterion, optim),
                                     batches_num,
                                     print_every=print_every)

        model.eval()
        with torch.no_grad():   
            print_examples((rebatch(pad_index, x) for x in val_iter),
                           model, max_len=CONFIG['TOKENS_CODE_CHUNK_MAX_LEN'] + 10,
                           n=3, src_vocab=diffs_field.vocab, trg_vocab=diffs_field.vocab)

            dev_perplexity = run_epoch((rebatch(pad_index, t) for t in val_iter),
                                       model,
                                       SimpleLossCompute(model.generator, criterion, None), batches_num)
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)

    return dev_perplexities


def run_experiment():
    pprint.pprint(CONFIG)
    train_data, val_data, test_data, diffs_field = load_data(verbose=CONFIG['VERBOSE'])
    model = make_model(len(diffs_field.vocab),
                       emb_size=CONFIG['WORD_EMBEDDING_SIZE'], hidden_size_encoder=CONFIG['ENCODER_HIDDEN_SIZE'],
                       hidden_size_decoder=CONFIG['DECODER_HIDDEN_SIZE'],
                       num_layers=1, dropout=0.2)  # TODO: num_layers
    dev_perplexities = train(model, train_data, val_data, diffs_field, print_every=CONFIG['PRINT_EVERY_EPOCH'])
    plot_perplexity(dev_perplexities)


if __name__ == "__main__":
    run_experiment()
