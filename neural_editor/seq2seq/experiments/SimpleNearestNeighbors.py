import numpy as np


class SimpleNearestNeighbors:
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric
        self.X_train = None

    def fit(self, X_train):
        self.X_train = X_train
        return self

    def kneighbors(self, X_test, return_distance: bool):
        neigh_dist, neigh_ind = [], []
        X_test_to_run = self.X_train if X_test is None else X_test
        for i, x in enumerate(X_test_to_run):
            best_dist = float("inf")
            best_ind = -1
            for j, y in enumerate(self.X_train):
                if X_test is None and i == j:
                    continue
                dist = self.metric(x, y)
                if dist < best_dist:
                    best_dist = dist
                    best_ind = j
            neigh_dist.append([best_dist])
            neigh_ind.append([best_ind])
        neigh_dist, neigh_ind = np.array(neigh_dist), np.array(neigh_ind)
        return (neigh_dist, neigh_ind) if return_distance else neigh_ind

