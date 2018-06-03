from keras import Model

import gs.config as cfg
import gs.common as co
import gs.plot as pl
import keras.optimizers as opt
import h5py
import random as ran

epoches = list(range(1, 6))
learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]

_ran = ran.Random()


def validate(model: Model) -> float:
    return 10 + _ran.random() * 22


def run():
    _cfg = cfg.conf('img100', '..')

    def train(lr: float) -> pl.DataRow:
        _cfg.optimizer = opt.Adam(lr=lr)

        path = co.h5_file(_cfg.id)
        print("reading from '{}'".format(path))
        with h5py.File(path, 'r', libver='latest') as h5_file:
            x = h5_file['dsx']
            y = h5_file['dsy']
            print("x {}".format(x.shape))
            print("y {}".format(y.shape))

            model = _cfg.model()
            print("Defined model {}".format(model.to_yaml()))
            print("--------------------------------------------")
            model.compile(loss='binary_crossentropy', optimizer=_cfg.optimizer, metrics=['accuracy'])
            print("Compiled model {}".format(model))

            data = []
            for i in epoches:
                # model.fit(x, y, epochs=1, batch_size=20, shuffle='batch')
                score = validate(model)
                data.append(pl.XY(i, score))
        return pl.DataRow(data=data, name="{:6.4f}".format(lr))

    data_rows = []
    for _lr in learning_rates:
        data_rows.append(train(_lr))

    file = co.work_file(name="test1.png", _dir='opt')
    dia = pl.Dia(data=data_rows, title="Optimize Learning Rates")
    pl.plot_dia(dia, file=file)
    print("plot result to {}".format(file))


run()
