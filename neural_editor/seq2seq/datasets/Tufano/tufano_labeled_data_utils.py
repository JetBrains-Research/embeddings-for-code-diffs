import sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def draw_hist_for_classes(path_to_data: Path) -> None:
    classes = path_to_data.joinpath('classes.txt').read_text().splitlines()
    classes_counter = defaultdict(lambda: 0)
    for cls in classes:
        classes_counter[cls] += 1
    print(len(classes_counter))
    for pair in reversed(sorted(classes_counter.items(), key=lambda x: x[1])):
        print(f'{pair[0]}: {pair[1]}')
    plt.hist(classes)
    plt.show()


def main() -> None:
    dataset_root = Path(sys.argv[1])
    draw_hist_for_classes(dataset_root)


if __name__ == "__main__":
    main()
