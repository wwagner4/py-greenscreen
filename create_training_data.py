import itertools as it
import os.path as osp
import os
from pathlib import Path
from typing import Tuple, Iterable, List, Any

import matplotlib.pylab as pl
import numpy as np


class TrainFileNames:

    def __init__(self, green: str, transp: str):
        self.green = green
        self.transp = transp


class TrainImages:

    def __init__(self, green: np.array, transp: np.array):
        self.green = green
        self.transp = transp


# f: A function returning an Iterable
def flatmap(f, list_of_list: Iterable[Any]) -> Iterable[Any]:
    return it.chain.from_iterable(map(f, list_of_list))


def create_training_img_arrays(green_transp_name: TrainFileNames) -> TrainImages:
    img: np.array = pl.imread(green_transp_name.green)
    imt: np.array = pl.imread(green_transp_name.transp)[:, :, -1:]  # use only the transparent value
    return TrainImages(img, imt)


def core_indices(rows: int, cols: int, delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(delta, rows - delta):
        for j in range(delta, cols - delta):
            yield (i, j)


def square_indices_rows(delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            yield (i, j)


def square_indices_cols(delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            yield (j, i)


def square_indices_rows_cols(delta: int) -> Iterable[Tuple[int, int]]:
    return flatmap(lambda l: l, [square_indices_rows(delta), square_indices_cols(delta)])


def create_training_data(green_transp_name: TrainFileNames, delta: int, around_idxs: List[Tuple[int, int]]) -> \
        Iterable[np.array]:
    train_images = create_training_img_arrays(green_transp_name)
    rows = train_images.green.shape[0]
    cols = train_images.green.shape[1]
    core_idxs: Iterable[Tuple[int, int]] = core_indices(rows, cols, delta)
    return create_training_data1(train_images, core_idxs, around_idxs)


def create_training_data1(
        train_images: TrainImages,
        core_idxs: Iterable[Tuple[int, int]], around_idxs: List[Tuple[int, int]]) -> Iterable[np.array]:
    for r, c in core_idxs:
        greens = np.empty(0, dtype=float)
        for i, j in around_idxs:
            r1 = r + i
            c1 = c + j
            green = train_images.green[r1, c1]
            greens = np.hstack((greens, green))
        transp = train_images.transp[r, c, 0]
        yield np.hstack((greens, transp))


def write_file(out_file_name: str, data: Iterable[np.array]):
    def array_to_string(arr: np.array) -> str:
        _line = ''.join(['%5.3f;' % num for num in arr])
        return _line[:-1]

    with open(out_file_name, 'w') as f:
        for i, line in enumerate(data):
            if i % 1000 == 0 and i > 0:
                print("wrote {} lines".format(i))
            f.write(array_to_string(line) + "\n")


def run():
    home_dir = Path.home()
    work_dir = osp.join(home_dir, 'work', 'work-greenscreen')
    print("work_dir: '{}'".format(work_dir))
    if not osp.exists(work_dir):
	    os.makedirs(work_dir)

    out_file = osp.join(work_dir, 'data_img100.csv')

    img_dir = "res/img100"
    names = [
        TrainFileNames(
            osp.join(img_dir, 'bsp1_green.png'),
            osp.join(img_dir, 'bsp1_transp.png')),
        TrainFileNames(
            osp.join(img_dir, 'bsp2_green.png'),
            osp.join(img_dir, 'bsp2_transp.png'))]

    delta = 10

    around_idxs = list(square_indices_rows_cols(delta))
    datas = flatmap(lambda train_file_names: create_training_data(train_file_names, delta, around_idxs), names)
    write_file(out_file, datas)
    print("wrote data to'{}'".format(out_file))


run()
