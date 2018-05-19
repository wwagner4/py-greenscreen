from typing import Tuple, Iterable, List

import h5py
import numpy as np

from gs import common as co


def create(_id: str):
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

    def write(_id: str, conf: co.Conf, data: Iterable[np.array]) -> str:
        def write_csv(_id: str, data: Iterable[np.array]) -> str:
            def array_to_string(arr: np.array) -> str:
                _line = ''.join(['%7.5f;' % num for num in arr])
                return _line[:-1]

            path = co.csv_file(_id)
            print("writing to {}".format(path))
            with open(path, 'w') as f:
                for i, line in enumerate(data):
                    if i % 1000 == 0 and i > 0:
                        print("wrote {} lines".format(i))
                    f.write(array_to_string(line) + "\n")
            return path

        def write_h5(_id: str, data: Iterable[np.array], conf: co.Conf) -> str:
            path = co.h5_file(_id)
            print("writing to {}".format(path))
            dim = conf.dim
            delta = conf.delta
            img_cnt = len(conf.train_file_names)
            r1 = dim.rows - 2 * delta
            c1 = dim.cols - 2 * delta
            rows = r1 * c1 * img_cnt
            cols = len(conf.around_indices) * 3 + 1  # features plus one label
            print("ds shape {} {}".format(rows, cols))
            with h5py.File(path, 'w', libver='latest') as file:
                ds = file.create_dataset(name="dx", shape=(rows, cols), dtype=float)
                for i, line in enumerate(data):
                    if i % 1000 == 0 and i > 0:
                        print("wrote {} lines".format(i))
                    ds[i] = np.array([line])
            return path

        t = conf.data_file_type
        if t == 'csv':
            return write_csv(_id, data)
        elif t == 'h5':
            return write_h5(_id, data, conf)
        else:
            raise NameError("data file type can only be 'csv' or 'h5'. {}".format(t))

    def run():
        cfg = co.conf(_id)
        nams = cfg.train_file_names
        delt = cfg.delta
        around_idxs = list(co.square_indices_rows_cols(delt))
        datas = co.flatmap(lambda _nams: create_rows(_nams, around_idxs, cfg), nams)
        out_file = write(_id, cfg, datas)
        print("wrote data to'{}'".format(out_file))

    run()
