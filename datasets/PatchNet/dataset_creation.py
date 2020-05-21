import sys
from pathlib import Path

from datasets.PatchNet.PatchNetDataset import PatchNetDataset


def main() -> None:
    if len(sys.argv) != 4:
        print('Usage: <root where to save processed data> <path to file with description of dataset> '
              '<path to local copy of linux git repository>')
        exit(1)
    root = Path(sys.argv[1])
    dataset_description_file = Path(sys.argv[2])
    linux_repository_filepath = Path(sys.argv[3])
    if not root.is_dir():
        print(f'No such directory: {root.absolute()}')
    if not dataset_description_file.is_file():
        print(f'No such file: {dataset_description_file.absolute()}')
        exit(1)
    if not linux_repository_filepath.is_dir():
        print(f'No such directory: {linux_repository_filepath.absolute()}')
    patch_net_dataset = PatchNetDataset(root, dataset_description_file, linux_repository_filepath)
    patch_net_dataset.print_statistics()
    patch_net_dataset.write_data()


if __name__ == "__main__":
    main()
