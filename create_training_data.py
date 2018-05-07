import itertools as it
import os.path as osp
from typing import Tuple, Iterable, List, Any

import matplotlib.pylab as pl
import numpy as np


# f: A function returning an Iterable
def flatmap(f, list_of_list: Iterable[Any]) -> Iterable[Any]:
    return it.chain.from_iterable(map(f, list_of_list))


def create_training_img_arrays(green_transp_name: Tuple[str, str]) -> Tuple[np.array, np.array]:
    img: np.array = pl.imread(green_transp_name[0])
    imt: np.array = pl.imread(green_transp_name[1])[:, :, -1:]  # use only the transparent value
    return img, imt


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


def create_training_data(green_transp_name: Tuple[str, str], delta: int, around_idxs: List[Tuple[int, int]]) -> \
        Iterable[np.array]:
    green_img, transp_img = create_training_img_arrays(green_transp_name)
    rows = green_img.shape[0]
    cols = green_img.shape[1]
    core_idxs: Iterable[Tuple[int, int]] = core_indices(rows, cols, delta)
    return create_training_data1(green_img, transp_img, core_idxs, around_idxs)


def create_training_data1(
        g: np.array, t: np.array,
        core_idxs: Iterable[Tuple[int, int]], around_idxs: List[Tuple[int, int]]) -> Iterable[np.array]:
    for r, c in core_idxs:
        greens = np.empty(0, dtype=float)
        for i, j in around_idxs:
            r1 = r + i
            c1 = c + j
            green = g[r1, c1]
            greens = np.hstack((greens, green))
        transp = t[r, c, 0]
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
    out_file = "/Users/wwagner4/work/work-greenscreen/data01.csv"

    img_dir = "res/img100"
    imgg1 = osp.join(img_dir, 'bsp1_green.png')
    imgt1 = osp.join(img_dir, 'bsp1_transp.png')
    imgg2 = osp.join(img_dir, 'bsp2_green.png')
    imgt2 = osp.join(img_dir, 'bsp2_transp.png')
    names = [(imgg1, imgt1), (imgg2, imgt2)]

    delta = 10

    around_idxs = list(square_indices_rows_cols(10))
    datas = flatmap(lambda green_transp_name: create_training_data(green_transp_name, delta, around_idxs), names)
    write_file(out_file, datas)
    print("wrote data to'{}'".format(out_file))


run()
