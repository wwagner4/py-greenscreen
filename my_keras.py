import os
import os.path as osp
from pathlib import Path
from typing import Tuple

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def read_csv(file: str) -> Tuple[np.array, np.array]:
    data = np.genfromtxt(file, delimiter=';')
    _x = data[:, :-1]
    _y = data[:, -1:]
    return _x, _y


home_dir = Path.home()
work_dir = osp.join(home_dir, 'work', 'work-greenscreen')
print("work_dir: '{}'".format(work_dir))
if not osp.exists(work_dir):
    os.makedirs(work_dir)
csv_file = osp.join(work_dir, 'data_img100.csv')
print("reading from '{}'".format(csv_file))
x, y = read_csv(csv_file)
print("x {}".format(x.shape))
print("y {}".format(y.shape))

model = Sequential()
model.add(Dense(1000, input_dim=2646, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
print("Defined model {}".format(model))
adam = Adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print("Compiled model {}".format(model))

model.fit(x, y, epochs=10, batch_size=20)
print("Fit model {}".format(model))

model_file = osp.join(work_dir, "model-img100.h5")
model.save(model_file)
print("Saved model to {}".format(model_file))
