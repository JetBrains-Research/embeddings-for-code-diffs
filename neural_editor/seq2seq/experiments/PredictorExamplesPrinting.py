from torchtext import data
from torchtext.data import Dataset

from neural_editor.seq2seq import EncoderPredictor
from neural_editor.seq2seq.PredictorBatch import rebatch_predictor
from neural_editor.seq2seq.config import Config
from neural_editor.seq2seq.train_utils import lookup_words_no_copying_mechanism


class PredictorExamplesPrinting:
    def __init__(self, model: EncoderPredictor, n: int, config: Config) -> None:
        super().__init__()
        self.model = model
        self.n = n
        self.config = config

    def conduct(self, dataset: Dataset, dataset_label: str) -> None:
        print(f'Start conducting predictor examples printing experiment for {dataset_label}...', flush=True)
        data_iter = data.Iterator(dataset, batch_size=1, train=False, sort=False,
                                  sort_within_batch=False, repeat=False, shuffle=False,
                                  device=self.config['DEVICE'])
        vocab = dataset.fields['src'].vocab
        for i, batch in enumerate(data_iter, 1):
            batch = rebatch_predictor(batch)
            print(f'\n====EXAMPLE {i}====')
            print(f'ID: {batch.ids[0].data.item()}')
            src = batch.src.cpu().numpy()[0, 1:-1]
            updated = batch.updated.cpu().numpy()[0, 1:-1]
            diff_alignment = batch.diff_alignment.cpu().numpy()[0, 1:-1]
            diff_prev = batch.diff_prev.cpu().numpy()[0, 1:-1]
            diff_updated = batch.diff_updated.cpu().numpy()[0, 1:-1]
            print("Prev           : " + " ".join(lookup_words_no_copying_mechanism(src, vocab)))
            print("Updated        : " + " ".join(lookup_words_no_copying_mechanism(updated, vocab)))
            print(f'Diff alignment: ' + " ".join(lookup_words_no_copying_mechanism(diff_alignment, vocab)))
            print(f'Diff prev     : ' + " ".join(lookup_words_no_copying_mechanism(diff_prev, vocab)))
            print(f'Diff updated  : ' + " ".join(lookup_words_no_copying_mechanism(diff_updated, vocab)))
            print(f'Ground truth  : {batch.trg.data.item()}')
            probs = self.model.predict(batch)
            print(f'Predicted     : {self.model.predict_labels(probs).data.item()} ({probs.data.item()})')
            if i == self.n:
                break


