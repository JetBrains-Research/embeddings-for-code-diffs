import os
import sys
from os import mkdir

import numpy as np
from sklearn.model_selection import KFold, train_test_split

SEED = 542


def read_data(root: str) -> np.ndarray:
    data = []
    with open(os.path.join(root, 'diff.txt'), 'r') as diff_file, \
            open(os.path.join(root, 'msg.txt'), 'r') as msg_file, \
            open(os.path.join(root, 'prev.txt'), 'r') as prev_file, \
            open(os.path.join(root, 'updated.txt'), 'r') as updated_file:
        for diff_line, msg_line, prev_line, updated_line in zip(diff_file, msg_file, prev_file, updated_file):
            diff_line, msg_line, prev_line, updated_line = \
                diff_line.strip(), msg_line.strip(), prev_line.strip(), updated_line.strip()
            data.append((diff_line, msg_line, prev_line, updated_line))
    return np.array(data)


def read_dataset(root: str) -> np.ndarray:
    train = read_data(os.path.join(root, 'train'))
    val = read_data(os.path.join(root, 'val'))
    test = read_data(os.path.join(root, 'test'))
    return np.concatenate([train, val, test], axis=0)


def write_fold(root, i, datasets):
    path = os.path.join(root, str(i + 1))
    mkdir(path)
    for data_type, data in datasets.items():
        data_type_path = os.path.join(path, data_type)
        mkdir(data_type_path)
        files_to_write = ['diff.txt', 'msg.txt', 'prev.txt', 'updated.txt']
        for i, filename in enumerate(files_to_write):
            with open(os.path.join(data_type_path, filename), 'w') as file:
                file.write('\n'.join(data[:, i]))


def main():
    if len(sys.argv) != 3:
        print("Args: <k in k fold> <path to dataset>")
        exit(1)
    k = int(sys.argv[1])
    root = sys.argv[2]
    dataset = read_dataset(root)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    k_fold_root = os.path.join(root, 'k_folds')
    mkdir(k_fold_root)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        X_train, X_test = dataset[train_index], dataset[test_index]
        X_train, X_val = train_test_split(X_train, test_size=1 / (k - 1), random_state=SEED)
        write_fold(k_fold_root, i, {'train': X_train, 'val': X_val, 'test': X_test})


if __name__ == "__main__":
    main()
