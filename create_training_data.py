import itertools as it
import os.path as osp
from typing import Tuple, Iterable, List

import matplotlib.pylab as pl
import numpy as np

imgDir = "res/img100"


def flatmap(f, list_of_list):
    return it.chain.from_iterable(map(f, list_of_list))


def create_training_img_arrays(img_name: str) -> Tuple[np.array, np.array]:
    fnamg = "bsp{}_0.png".format(img_name)
    fnamt = "bsp{}_1.png".format(img_name)
    img: np.array = pl.imread(osp.join(imgDir, fnamg))
    imt: np.array = pl.imread(osp.join(imgDir, fnamt))[:, :, -1:]  # use only the transparent value
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


def create_training_data(name: str, delta: int, around_idxs: List[Tuple[int, int]]) -> Iterable[np.array]:
    green_img, transp_img = create_training_img_arrays(name)
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


def run():
    names = ['1', '2']
    delta = 10
    around_idxs = list(square_indices_rows_cols(delta))
    datas = flatmap(lambda name: create_training_data(name, delta, around_idxs), names)
    for idx, line in enumerate(datas):
        print("{} {}".format(idx, line))


def test():
    si = square_indices_rows_cols(1)
    for i in si:
        print(i)


# test()
run()

'''
for idx, line in enumerate(create_training_data('1', 10)):
    print("{} {}".format(idx, line))

for x, y in square_indices(7):
    print("{}, {}".format(x, y))


    
    
for i in range(0, 100):
    for j in range(0, 133):
        x = t[i, j]
        print("{} {} {}".format(i, j, x))



g, t = create_training_img_arrays('1')
dimg = g.shape[0:2]
print("green {}".format(dimg))
dimt = t.shape[0:2]
print("transp {}".format(dimt))


'''
