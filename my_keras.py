import os.path as osp
from pathlib import Path
from typing import Tuple

import numpy as np


def read_csv(file: str) -> Tuple[np.array, np.array]:
    data = np.genfromtxt(file, delimiter=';')
    _x = data[:, :-1]
    _y = data[:, -1:]
    return _x, _y


home_dir = Path.home()
work_dir = osp.join(home_dir, 'work/work-greenscreen')
csv_file = osp.join(work_dir, 'data_img100_1000.csv')
print("reading from '{}'".format(csv_file))
x, y = read_csv(csv_file)
print("x {}".format(x.shape))
print("y {}".format(y.shape))
