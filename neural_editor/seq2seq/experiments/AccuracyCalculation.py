from torchtext import data
from torchtext.data import Field, Dataset
from torchtext.vocab import Vocab

from neural_editor.seq2seq import EncoderDecoder
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.decoder.BatchBeamSearch import BatchedBeamSearch
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
        self.beam_search = BatchedBeamSearch(self.beam_size, self.model, sos_index, self.eos_index, self.config)

    def conduct_on_single_dataset(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting accuracy calculation experiment for {dataset_label}...')
        data_iterator = data.Iterator(dataset, batch_size=1,
                                      sort=False, train=False, shuffle=False, device=self.config['DEVICE'])
        correct_top_1, correct, total = \
            calculate_top_k_accuracy(self.beam_size,
                                     [rebatch(self.pad_index, batch, self.config) for batch in data_iterator],
                                     self.beam_search.decode, self.eos_index)
        print(f'Top-1 accuracy: {correct_top_1} / {total} = {correct_top_1 / total}')
        print(f'Top-{self.beam_size} accuracy: {correct} / {total} = {correct / total}')
