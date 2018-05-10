import os.path as osp
from typing import Tuple, Iterable, List

import matplotlib.pylab as pl
import numpy as np

import common as co


def create(out_file: str):
    class Dim:

        def __init__(self, rows: int, cols: int):
            self.rows = rows
            self.cols = cols

    class TrainFileNames:

        def __init__(self, green: str, transp: str):
            self.green = green
            self.transp = transp

    class TrainImages:

        def __init__(self, file_names: TrainFileNames, dim: Dim):
            def validate(img: np.array, name: str):
                rows = img.shape[0]
                cols = img.shape[1]
                if rows != dim.rows or cols != dim.cols:
                    msg = "Illegal dimension of image {}: {}/{}. expected: {}/{}" \
                        .format(name, rows, cols, dim.rows, dim.cols)
                    raise AssertionError(msg)

            green: np.array = pl.imread(file_names.green)
            validate(green, file_names.green)
            transp: np.array = pl.imread(file_names.transp)[:, :, -1:]  # use only the transparent value
            validate(transp, file_names.transp)
            self.green = green
            self.transp = transp

    def create_rows(names: TrainFileNames, bord: int, idx_rel: List[Tuple[int, int]], dim: Dim) -> Iterable[np.array]:
        train_images = TrainImages(names, dim)
        idx_core: Iterable[Tuple[int, int]] = co.core_indices(dim.rows, dim.cols, bord)
        for row, col in idx_core:
            greens = np.empty(0, dtype=float)
            for row_off, col_off in idx_rel:
                row1 = row + row_off
                col1 = col + col_off
                green = train_images.green[row1, col1]
                greens = np.hstack((greens, green))
            transp = train_images.transp[row, col, 0]
            yield np.hstack((greens, transp))

    def write_file(file_name: str, data: Iterable[np.array]):
        def array_to_string(arr: np.array) -> str:
            _line = ''.join(['%7.5f;' % num for num in arr])
            return _line[:-1]

        with open(file_name, 'w') as f:
            for i, line in enumerate(data):
                if i % 1000 == 0 and i > 0:
                    print("wrote {} lines".format(i))
                f.write(array_to_string(line) + "\n")

    def create_img100():

        img_dir = "res/img100"
        dim = Dim(100, 133)
        names = [
            TrainFileNames(
                green=osp.join(img_dir, 'bsp1_green.png'),
                transp=osp.join(img_dir, 'bsp1_transp.png')),
            TrainFileNames(
                green=osp.join(img_dir, 'bsp2_green.png'),
                transp=osp.join(img_dir, 'bsp2_transp.png'))]

        bord = 10

        around_idxs = list(co.square_indices_rows_cols(bord))
        datas = co.flatmap(lambda train_file_names: create_rows(train_file_names, bord, around_idxs, dim), names)
        write_file(out_file, datas)
        print("wrote data to'{}'".format(out_file))

    create_img100()
