from typing import Tuple, Iterable, List

import h5py
import numpy as np

from gs import common as co


def create(work_dir: str, cfg: co.Conf):

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

    def write_h5(_id: str, data: Iterable[np.array]) -> str:
        path = co.h5_file(work_dir, _id)
        print("writing to {}".format(path))
        img_cnt = len(cfg.train_file_names)
        r, c = co.features_shape(cfg)
        rows = r * img_cnt  # Multiply with the amount of images
        rows_train = int(rows * 0.6)
        rows_cross = int(rows * 0.2)
        rows_test = rows - rows_train - rows_cross
        print("ds shape {} {}".format(rows * img_cnt, c))
        with h5py.File(path, 'w', libver='latest') as file:
            dsx = file.create_dataset(name="dsx", shape=(rows_train, c), dtype=float)
            dsy = file.create_dataset(name="dsy", shape=(rows_train, 1), dtype=float)
            dsx_cross = file.create_dataset(name="dsx_cross", shape=(rows_cross, c), dtype=float)
            dsy_cross = file.create_dataset(name="dsy_cross", shape=(rows_cross, 1), dtype=float)
            dsx_test = file.create_dataset(name="dsx_train", shape=(rows_test, c), dtype=float)
            dsy_test = file.create_dataset(name="dsy_train", shape=(rows_test, 1), dtype=float)
            cat_data = co.categorize(data, rows, rows_train, rows_cross, l1="train", l2="cross", l3="test")
            i_train = 0
            i_cross = 0
            i_test = 0
            for cat, line in cat_data:
                if cat == 'train':
                    if i_train % 1000 == 0 and i_train > 0:
                        print("wrote {} training data".format(i_train))
                    dsx[i_train] = line[:-1]
                    dsy[i_train] = line[-1:]
                    i_train += 1
                elif cat == 'cross':
                    if i_cross % 1000 == 0 and i_cross > 0:
                        print("wrote {} cross validation data".format(i_cross))
                    dsx_cross[i_cross] = line[:-1]
                    dsy_cross[i_cross] = line[-1:]
                    i_cross += 1
                else:
                    if i_test % 1000 == 0 and i_test > 0:
                        print("wrote {} test data".format(i_test))
                    dsx_test[i_test] = line[:-1]
                    dsy_test[i_test] = line[-1:]
                    i_test += 1
        return path

    def run():
        nams = cfg.train_file_names
        around_idxs = list(cfg.around_indices)
        datas = co.flatmap(lambda _nams: create_rows(_nams, around_idxs, cfg), nams)
        out_file = write_h5(cfg.id, datas)
        print("wrote data to'{}'".format(out_file))

    run()
