import os
import os.path as osp
from typing import List

import matplotlib.pylab as pl
import numpy as np

imgDir = "res/img100"


def images() -> List[str]:
    return os.listdir(imgDir)


def create_training_data(img_name: str) -> object:
    fnamg = "bsp{}_0.png".format(img_name)
    fnamt = "bsp{}_1.png".format(img_name)
    img: np.array = pl.imread(osp.join(imgDir, fnamg))
    print("img green shape {}".format(img.shape))
    imt: np.array = pl.imread(osp.join(imgDir, fnamt))[:, :, -1:]  # use only the transparent value
    print("img transp shape {} {}".format(imt.shape, type(imt)))
    return img, imt


'''
g, t = create_training_data('1')
for i in range(0, 100):
    for j in range(0, 133):
        x = t[i, j]
        print("{} {} {}".format(i, j, x))



g, t = create_training_data('1')
dim = g.shape[0:2]
print("green {}".format(dimg))
dimt = t.shape[0:2]
print("transp {}".format(dimt))

'''
