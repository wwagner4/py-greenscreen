import itertools as it
import os
import os.path as osp
from pathlib import Path
from typing import Tuple, Iterable, Any, List

import matplotlib.pylab as pl
import numpy as np
from keras.optimizers import Optimizer
from keras import Model


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


# f: A function returning an Iterable
def flatmap(f, list_of_list: Iterable[Any]) -> Iterable[Any]:
    return it.chain.from_iterable(map(f, list_of_list))


def core_indices(rows: int, cols: int, delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(delta, rows - delta):
        for j in range(delta, cols - delta):
            yield (i, j)


def work_file(name: str, _dir: str = None) -> str:
    """Returns path to a file in the workdirectory or to a file in a subdirectory if _dir is defined.
    The work directory is created if it does not exist
    """
    home_dir = Path.home()
    if _dir is None:
        _work_dir = osp.join(home_dir, 'work', 'work-greenscreen')
    else:
        _work_dir = osp.join(home_dir, 'work', 'work-greenscreen', _dir)
    if not osp.exists(_work_dir):
        print("created work dir: '{}'".format(_work_dir))
        os.makedirs(_work_dir)
    return osp.join(_work_dir, name)


def work_dir(name: str) -> str:
    """Returns the path to a directory in the work directory. the directory is created if it does not exist"""
    home_dir = Path.home()
    _dir = osp.join(home_dir, 'work', 'work-greenscreen', name)
    if not osp.exists(_dir):
        print("created work dir: '{}'".format(_dir))
        os.makedirs(_dir)
    return _dir


def csv_file(_id: str) -> str:
    name = "data_{}.csv".format(_id)
    return work_file(name)


def h5_file(_id: str) -> str:
    name = "data_{}.h5".format(_id)
    return work_file(name)


def model_file(_id: str) -> str:
    name = "model_{}.h5".format(_id)
    return work_file(name)


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
