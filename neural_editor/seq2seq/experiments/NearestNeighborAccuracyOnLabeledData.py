from typing import List

from torchtext.data import Dataset

from neural_editor.seq2seq.experiments import NearestNeighbor


class NearestNeighborAccuracyOnLabeledData:
    def __init__(self, nearest_neighbor_calculator: NearestNeighbor) -> None:
        super().__init__()
        self.nearest_neighbor_calculator = nearest_neighbor_calculator

    def conduct(self, dataset: Dataset, classes: List[str], dataset_label: str):
        print(f'Start conducting nearest neighbor accuracy on labeled data experiment for {dataset_label}...')
        nbrs_result = self.nearest_neighbor_calculator.find(dataset_train=dataset, dataset_test=None)
        for vector_type in nbrs_result:
            for distance_type in nbrs_result[vector_type]:
                correct = [i for i, neighbor_id in enumerate(nbrs_result[vector_type][distance_type])
                           if classes[i] == classes[neighbor_id]]
                print(f'vector_type: {vector_type}, distance_type: {distance_type}, '
                      f'accuracy: {len(correct)} / {len(classes)} = {len(correct) / len(classes)}')
