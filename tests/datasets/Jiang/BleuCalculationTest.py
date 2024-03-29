import unittest

from neural_editor.seq2seq.experiments.BleuCalculation import BleuCalculation


class Example:

    def __init__(self, trg) -> None:
        super().__init__()
        self.trg = trg


class BleuChecking(unittest.TestCase):
    CONFIG = {'BLEU_PERL_SCRIPT_PATH': '../../../neural_editor/seq2seq/experiments/multi-bleu.perl'}
    BLEU_CALCULATION_EXPERIMENT = BleuCalculation(CONFIG)

    def test_same_predicted_as_target(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Fix typo'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_bleu_zero(self):
        predictions = [
            [
                'Completely incorrect guess .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Totally incorrect guess .'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_only_one_example(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split())
        ]
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_no_predictions(self):
        predictions = [
            [], []
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        BleuChecking.BLEU_CALCULATION_EXPERIMENT.conduct(predictions, dataset, 'test')

    def test_same_predicted_as_target_bleu_score(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Fix typo'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(100.0, bleu_score)

    def test_bleu_zero_bleu_score(self):
        predictions = [
            [
                'Completely incorrect guess .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ],
            [
                'Totally incorrect guess .'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)

    def test_only_one_example_bleu_score(self):
        predictions = [
            [
                'Fix android_media_AudioSystem_getMasterMute return type .'.split(),
                'Not fix android_media_AudioSystem_getMasterMute return type .'.split()
            ]
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split())
        ]
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(100.0, bleu_score)

    def test_no_predictions_bleu_score(self):
        predictions = [
            [], []
        ]
        dataset = [
            Example('Fix android_media_AudioSystem_getMasterMute return type .'.split()),
            Example('Fix typo'.split())
        ]
        bleu_score = BleuChecking.BLEU_CALCULATION_EXPERIMENT.get_bleu_score(predictions, dataset)
        self.assertEqual(0.0, bleu_score)


if __name__ == '__main__':
    unittest.main()
