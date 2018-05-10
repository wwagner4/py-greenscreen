from typing import Tuple

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


def train(csv_file: str, model_file: str):
    def read_csv(file: str) -> Tuple[np.array, np.array]:
        data = np.genfromtxt(file, delimiter=';')
        _x = data[:, :-1]
        _y = data[:, -1:]
        return _x, _y

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

    model.fit(x, y, epochs=6, batch_size=20)
    print("Fit model {}".format(model))

    model.save(model_file)
    print("Saved model to {}".format(model_file))
