import itertools as it
import os
import os.path as osp
from pathlib import Path
from typing import Tuple, Iterable, Any, List

import matplotlib.pylab as pl
import numpy as np


class Dim:

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

    def __str__(self):
        return "<Dim rows:{} cols:{}>".format(self.rows, self.cols)


class TrainFileNames:

    def __init__(self, green: str, transp: str):
        self.green = green
        self.transp = transp

    def __str__(self):
        return "<TrainFileNames green:{} transp:{}>".format(self.green, self.transp)


class Conf:

    def __init__(self,
                 dim: Dim,
                 delta: int,
                 data_file_type: str,  # Can be 'h5' or 'csv'
                 train_file_names: List[TrainFileNames],
                 around_indices: List[Tuple[int, int]]):
        self.dim = dim
        self.delta = delta
        self.data_file_type = data_file_type
        self.train_file_names = train_file_names
        self.around_indices = around_indices


# f: A function returning an Iterable
def flatmap(f, list_of_list: Iterable[Any]) -> Iterable[Any]:
    return it.chain.from_iterable(map(f, list_of_list))


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


def work_file(name: str, _dir: str = None) -> str:
    home_dir = Path.home()
    if _dir is None:
        work_dir = osp.join(home_dir, 'work', 'work-greenscreen')
    else:
        work_dir = osp.join(home_dir, 'work', 'work-greenscreen', _dir)
    if not osp.exists(work_dir):
        print("created work dir: '{}'".format(work_dir))
        os.makedirs(work_dir)
    return osp.join(work_dir, name)


def csv_file(_id: str) -> str:
    name = "data_{}.csv".format(_id)
    return work_file(name)


def h5_file(_id: str) -> str:
    name = "data_{}.h5".format(_id)
    return work_file(name)


def model_file(_id: str) -> str:
    name = "model_{}.h5".format(_id)
    return work_file(name)


def conf(_id: str) -> Conf:
    if _id == 'img100':
        img_dir = "res/img100"
        _delta = 10
        return Conf(
            dim=Dim(100, 133),
            delta=_delta,
            data_file_type='h5',
            train_file_names=[
                TrainFileNames(
                    green=osp.join(img_dir, 'bsp1_green.png'),
                    transp=osp.join(img_dir, 'bsp1_transp.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'bsp2_green.png'),
                    transp=osp.join(img_dir, 'bsp2_transp.png'))],
            around_indices=list(square_indices_rows_cols(_delta)))
    elif _id == 'img500':
        img_dir = "res/img500"
        _delta = 10
        return Conf(
            dim=Dim(500, 667),
            delta=_delta,
            data_file_type='h5',
            train_file_names=[
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen01.png'),
                    transp=osp.join(img_dir, 'trainTransp01.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen02.png'),
                    transp=osp.join(img_dir, 'trainTransp02.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen03.png'),
                    transp=osp.join(img_dir, 'trainTransp03.png'))],
            around_indices=list(square_indices_rows_cols(_delta)))
    else:
        raise ValueError("invalid id '{}'".format(_id))


def load_image(path: str, _dim: Dim) -> np.array:
    def validate(img: np.array):
        rows = img.shape[0]
        cols = img.shape[1]
        if rows != _dim.rows or cols != _dim.cols:
            msg = "Illegal dimension of image {}: {}/{}. expected: {}/{}" \
                .format(path, rows, cols, _dim.rows, _dim.cols)
            raise AssertionError(msg)

    re = pl.imread(path)
    validate(re)
    return re


def create_features(img: np.array, row: int, col: int, idx_rel: List[Tuple[int, int]]) -> np.array:
    re = np.empty(0, dtype=float)
    for row_off, col_off in idx_rel:
        row1 = row + row_off
        col1 = col + col_off
        green = img[row1, col1]
        re = np.hstack((re, green))
    return re
