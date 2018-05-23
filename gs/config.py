from typing import Tuple, Iterable

import gs.common as co
import os.path as osp

from keras import Model
from keras.layers import Dense
from keras.models import Sequential


def conf(_id: str, root_dir: str) -> co.Conf:
    if _id == 'img100':
        img_dir = osp.join(root_dir, "res", _id)
        _delta = 10
        return co.Conf(
            _id=_id,
            dim=co.Dim(100, 133),
            delta=_delta,
            img_dir=img_dir,
            train_file_names=[
                co.TrainFileNames(
                    green=osp.join(img_dir, 'bsp1_green.png'),
                    transp=osp.join(img_dir, 'bsp1_transp.png')),
                co.TrainFileNames(
                    green=osp.join(img_dir, 'bsp2_green.png'),
                    transp=osp.join(img_dir, 'bsp2_transp.png'))],
            around_indices=list(square_indices_rows_cols(_delta)),
            model=model_a)
    elif _id == 'img500':
        img_dir = osp.join(root_dir, "res", _id)
        _delta = 10
        return co.Conf(
            _id=_id,
            dim=co.Dim(500, 667),
            delta=_delta,
            img_dir=img_dir,
            train_file_names=[
                co.TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen01.png'),
                    transp=osp.join(img_dir, 'trainTransp01.png')),
                co.TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen02.png'),
                    transp=osp.join(img_dir, 'trainTransp02.png')),
                co.TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen03.png'),
                    transp=osp.join(img_dir, 'trainTransp03.png'))],
            around_indices=list(square_indices_rows_cols(_delta)),
            model=model_a)
    else:
        raise ValueError("invalid id '{}'".format(_id))


def model_a() -> Model:
    model = Sequential()
    model.add(Dense(1000, input_dim=2646, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def square_indices_rows(delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            yield (i, j)


def square_indices_cols(delta: int) -> Iterable[Tuple[int, int]]:
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            yield (j, i)


def square_indices_rows_cols(delta: int) -> Iterable[Tuple[int, int]]:
    return co.flatmap(lambda l: l, [square_indices_rows(delta), square_indices_cols(delta)])
