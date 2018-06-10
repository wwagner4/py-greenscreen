import itertools as it
import os
import os.path as osp
import random as ran
from typing import Tuple, Iterable, Any, List, Dict

import matplotlib.pylab as pl
import numpy as np
from keras import Model
from keras.optimizers import Optimizer


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
                 _id: str,
                 dim: Dim,
                 delta: int,
                 img_dir: str,
                 train_file_names: List[TrainFileNames],
                 around_indices: List[Tuple[int, int]],
                 model: Model,  # function to model
                 optimizer: Optimizer):
        self.id = _id
        self.dim = dim
        self.delta = delta
        self.train_file_names = train_file_names
        self.img_dir = img_dir
        self.around_indices = around_indices
        self.model = model
        self.optimizer = optimizer


def flatmap(f, list_of_list: Iterable[Any]) -> Iterable[Any]:
    """maps the elements of a list of list of elements to a liet of elements
    f: a function returning an iterable"""
    return it.chain.from_iterable(map(f, list_of_list))


def categorize(iterable: Iterable, size: int,
               size1: int, size2: int,
               l1: str = 'a', l2: str = 'a', l3: str = 'c') -> Iterable[Tuple]:
    """categorizes an iterable in three categories
    size: Size of the iterable. Must be known in advace
    size1: Size of the first category
    size2: Size of the second category. Third category is the rest.
    l1, l2, l3: Names of the categories
    returns an iterable of Tuples (category, obj)"""
    assert size1 <= size, "size1 > size. {}".format(size1)
    assert size2 <= size, "size2 > size. {}".format(size2)
    assert size1 + size2 <= size, "size1 + size2 > size. {}, {}".format(size1, size2)

    b0 = size1
    b1 = size1 + size2

    def cdict() -> Dict[int, Any]:
        idxs = np.arange(0, size)
        ran.shuffle(idxs)
        re = {}
        for i, idx in enumerate(idxs):
            if i < b0:
                re[idx] = l1
            elif i < b1:
                re[idx] = l2
            else:
                re[idx] = l3
        return re

    cd = cdict()
    for _idx, obj in enumerate(iterable):
        yield (cd[_idx], obj)


def core_indices(rows: int, cols: int, delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(delta, rows - delta):
        for j in range(delta, cols - delta):
            yield (i, j)


def work_file(_work_dir: str, name: str, _dir: str = None) -> str:
    """Returns path to a file in the workdirectory or to a file in a subdirectory if _dir is defined.
    The work directory is created if it does not exist
    """
    if _dir is None:
        _work_dir = osp.join(_work_dir)
    else:
        _work_dir = osp.join(_work_dir, _dir)
    if not osp.exists(_work_dir):
        print("created work dir: '{}'".format(_work_dir))
        os.makedirs(_work_dir)
    return osp.join(_work_dir, name)


def work_dir(_work_dir: str, name: str) -> str:
    """Returns the path to a directory in the work directory. the directory is created if it does not exist"""
    _dir = osp.join(_work_dir, name)
    if not osp.exists(_dir):
        print("created work dir: '{}'".format(_dir))
        os.makedirs(_dir)
    return _dir


def csv_file(_work_dir: str, _id: str) -> str:
    name = "data_{}.csv".format(_id)
    return work_file(_work_dir, name)


def h5_file(_work_dir: str, _id: str) -> str:
    name = "data_{}.h5".format(_id)
    return work_file(_work_dir, name)


def model_file(_work_dir: str, _id: str) -> str:
    name = "model_{}.h5".format(_id)
    return work_file(_work_dir, name)


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
    re = np.zeros((len(idx_rel) * 3,), dtype=np.float32)
    idx = 0
    for row_off, col_off in idx_rel:
        row1 = row + row_off
        col1 = col + col_off
        green = img[row1, col1]
        re[idx] = green[0]
        re[idx + 1] = green[1]
        re[idx + 2] = green[2]
        idx += 3
    return re


def features_shape(cfg: Conf) -> Tuple[int, int]:
    """Returns the shape of the features for one image"""
    r1 = cfg.dim.rows - 2 * cfg.delta
    c1 = cfg.dim.cols - 2 * cfg.delta
    rows = r1 * c1
    cols = len(cfg.around_indices) * 3  # Three because there is RGB
    return rows, cols
