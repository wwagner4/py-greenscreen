import os.path as osp
from typing import Tuple, Iterable, List

import numpy as np

from gs import common as co


def create(_id: str):
    class TrainFileNames:

        def __init__(self, green: str, transp: str):
            self.green = green
            self.transp = transp

        def __str__(self):
            return "<TrainFileNames green:{} transp:{}>".format(self.green, self.transp)

    class TrainImages:

        def __init__(self, file_names: TrainFileNames, dim: co.Dim):
            self.green = co.load_image(file_names.green, dim)
            self.transp = co.load_image(file_names.transp, dim)[:, :, -1:]  # use only the transparent value

    def create_rows(
            names: TrainFileNames, bord: int, idx_rel: List[Tuple[int, int]], dim: co.Dim) -> Iterable[np.array]:
        print("create rows from {}".format(names))
        train_images = TrainImages(names, dim)
        idx_core: Iterable[Tuple[int, int]] = co.core_indices(dim.rows, dim.cols, bord)
        for row, col in idx_core:
            features = co.create_features(train_images.green, row, col, idx_rel)
            labels = train_images.transp[row, col, 0]
            yield np.hstack((features, labels))

    def write_file(file_name: str, data: Iterable[np.array]):
        def array_to_string(arr: np.array) -> str:
            _line = ''.join(['%7.5f;' % num for num in arr])
            return _line[:-1]

        with open(file_name, 'w') as f:
            for i, line in enumerate(data):
                if i % 1000 == 0 and i > 0:
                    print("wrote {} lines".format(i))
                f.write(array_to_string(line) + "\n")

    def train_file_names(img_dir: str):
        return {
            "img100": [
                TrainFileNames(
                    green=osp.join(img_dir, 'bsp1_green.png'),
                    transp=osp.join(img_dir, 'bsp1_transp.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'bsp2_green.png'),
                    transp=osp.join(img_dir, 'bsp2_transp.png'))],
            "img500": [
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen01.png'),
                    transp=osp.join(img_dir, 'trainTransp01.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen02.png'),
                    transp=osp.join(img_dir, 'trainTransp02.png')),
                TrainFileNames(
                    green=osp.join(img_dir, 'trainGreen03.png'),
                    transp=osp.join(img_dir, 'trainTransp03.png'))]
        }

    def create_img100():

        img_dir = "res/{}".format(_id)
        names = train_file_names(img_dir)[_id]
        dim = co.dim(_id)
        bord = 10

        around_idxs = list(co.square_indices_rows_cols(bord))
        datas = co.flatmap(lambda _train_file_names: create_rows(_train_file_names, bord, around_idxs, dim), names)
        out_file = co.csv_file(_id)
        write_file(out_file, datas)
        print("wrote data to'{}'".format(out_file))

    create_img100()
