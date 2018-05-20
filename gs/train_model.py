from typing import Tuple

import numpy as np
from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from gs import common as co
import h5py


def model_a() -> Model:
    model = Sequential()
    model.add(Dense(1000, input_dim=2646, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train(_id: str):
    def read_csv() -> Tuple[np.array, np.array]:
        path = co.csv_file(_id)
        print("reading from '{}'".format(path))
        data = np.genfromtxt(path, delimiter=';')
        _x = data[:, :-1]
        _y = data[:, -1:]
        return _x, _y

    def read_h5() -> Tuple[np.array, np.array]:
        path = co.h5_file(_id)
        print("reading from '{}'".format(path))
        with h5py.File(path, 'r', libver='latest') as f:
            data = f['dx']
            _x = data[:, :-1]
            _y = data[:, -1:]
        return _x, _y

    def read() -> Tuple[np.array, np.array]:
        conf = co.conf(_id)
        t = conf.data_file_type
        if t == 'csv':
            return read_csv()
        elif t == 'h5':
            return read_h5()
        else:
            raise NameError("Invalid data file type {}. Correct values would be 'csv' or 'h5'".format(t))

    def run():
        x, y = read()
        print("x {}".format(x.shape))
        print("y {}".format(y.shape))

        model = model_a()
        print("Defined model {}".format(model.to_yaml()))
        print("--------------------------------------------")
        adam = Adam(lr=0.0005)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        print("Compiled model {}".format(model))

        model.fit(x, y, epochs=4, batch_size=20)
        print("Fit model {}".format(model))

        model_file = co.model_file(_id)
        model.save(model_file)

        print("Saved model to {}".format(model_file))

    run()
