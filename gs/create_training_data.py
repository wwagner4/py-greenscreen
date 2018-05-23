from typing import Tuple, Iterable, List

import h5py
import numpy as np

from gs import common as co
from gs import config as cf


def create(_id: str, root_dir: str):
    cfg = cf.conf(_id, root_dir)

    class TrainImages:

        def __init__(self, file_names: co.TrainFileNames, dim: co.Dim):
            self.green = co.load_image(file_names.green, dim)
            self.transp = co.load_image(file_names.transp, dim)[:, :, -1:]  # use only the transparent value

    def create_rows(
            names: co.TrainFileNames, idx_rel: List[Tuple[int, int]], conf: co.Conf) -> Iterable[np.array]:
        print("create rows from {}".format(names))
        train_images = TrainImages(names, conf.dim)
        idx_core: Iterable[Tuple[int, int]] = co.core_indices(conf.dim.rows, conf.dim.cols, conf.delta)
        for row, col in idx_core:
            features = co.create_features(train_images.green, row, col, idx_rel)
            labels = train_images.transp[row, col, 0]
            yield np.hstack((features, labels))

    def write_h5(_id: str, _data: Iterable[np.array]) -> str:
        path = co.h5_file(_id)
        print("writing to {}".format(path))
        img_cnt = len(cfg.train_file_names)
        r, c = co.features_shape(cfg)
        rows = r * img_cnt  # Multiply with the amount of images
        print("ds shape {} {}".format(rows * img_cnt, c))
        with h5py.File(path, 'w', libver='latest') as file:
            dsx = file.create_dataset(name="dsx", shape=(rows, c), dtype=float)
            dsy = file.create_dataset(name="dsy", shape=(rows, 1), dtype=float)
            for i, line in enumerate(_data):
                if i % 5000 == 0 and i > 0:
                    print("wrote {} lines to {}".format(i, path))
                dsx[i] = line[:-1]
                dsy[i] = line[-1:]
        return path

    def run():
        nams = cfg.train_file_names
        around_idxs = list(cfg.around_indices)
        datas = co.flatmap(lambda _nams: create_rows(_nams, around_idxs, cfg), nams)
        out_file = write_h5(_id, datas)
        print("wrote data to'{}'".format(out_file))

    run()
