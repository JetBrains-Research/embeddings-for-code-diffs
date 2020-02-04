from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.search import create_decode_method
from neural_editor.seq2seq.train_utils import rebatch, calculate_top_k_accuracy


class AccuracyCalculation:
    def __init__(self, model: EncoderDecoder, diffs_field: Field, config: Config) -> None:
        super().__init__()
        self.model = model
        vocab: Vocab = diffs_field.vocab
        self.pad_index: int = vocab.stoi[config['PAD_TOKEN']]
        sos_index: int = vocab.stoi[config['SOS_TOKEN']]
        self.eos_index: int = vocab.stoi[config['EOS_TOKEN']]
        self.config = config
        self.beam_size = self.config['BEAM_SIZE']
        self.topk_values = self.config['TOP_K']
        num_iterations = self.config['TOKENS_CODE_CHUNK_MAX_LEN'] + 1
        self.beam_search = create_decode_method(self.model, num_iterations, sos_index, self.eos_index, self.beam_size,
                                                self.config['NUM_GROUPS'], self.config['DIVERSITY_STRENGTH'],
                                                verbose=False)

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...')
        data_iterator = data.Iterator(dataset, batch_size=1,
                                      sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        correct_all_k, total = \
            calculate_top_k_accuracy(self.topk_values,
                                     [rebatch(self.pad_index, batch, self.config) for batch in data_iterator],
                                     self.beam_search, self.eos_index)
        for correct_top_k, k in zip(correct_all_k, self.topk_values):
            print(f'Top-{k} accuracy: {correct_top_k} / {total} = {correct_top_k / total}')
