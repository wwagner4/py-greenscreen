from typing import Tuple

import h5py
from keras.callbacks import History

import gs.common as co
import gs.config as cfg
import gs.plot as pl


def run(_id: str, root_dir: str, work_dir: str):
    epoche_cnt = 10
    runid = "bob001"

    epoches = list(range(0, epoche_cnt))
    _cfg = cfg.conf(_id, root_dir)

    def _train(h5_file) -> Tuple:

        x = h5_file['dsx']
        y = h5_file['dsy']
        xcv = h5_file['dsx_cross']
        ycv = h5_file['dsy_cross']

        model = _cfg.model()
        model.compile(loss='binary_crossentropy', optimizer=_cfg.optimizer, metrics=['accuracy'])
        data = []
        datat = []
        for epoche in epoches:
            hist: History = model.fit(x, y, epochs=1, batch_size=_cfg.batch_size, shuffle='batch', verbose=0)
            scoret = hist.history['acc'][0]
            score = model.evaluate(xcv, ycv,  batch_size=_cfg.batch_size, verbose=0)[1]
            print("score batch size: {} epoche {}/{} acc[train: {:.4f}, eval: {:.4f}]"
                  .format(_cfg.batch_size, epoche + 1, epoche_cnt, scoret, score))
            data.append(pl.XY(epoche, scoret))
            datat.append(pl.XY(epoche, score))

        return (
            pl.DataRow(data=datat, name="train"),
            pl.DataRow(data=data, name="eval"))

    def _train01(_h5_file) -> pl.Dia:
        data_rows = []
        title = "optimize model"
        drt, dr = _train(_h5_file)
        data_rows.append(drt)
        data_rows.append(dr)
        return pl.Dia(data=data_rows, title=title,
                      xaxis=pl.Axis(title="epoche"),
                      yaxis=pl.Axis(lim=(0.8, 1.0)))

    path = co.h5_file(_work_dir=work_dir, _id=_cfg.id)
    print("reading from '{}'".format(path))
    with h5py.File(path, 'r', libver='latest') as _h5_file:

        dia = _train01(_h5_file)

        file = co.work_file(_work_dir=work_dir, name="opt_extend_model_{}.png".format(runid), _dir='opt')
        pl.plot_dia(dia, file=file, img_size=(1000, 800))
        print("plot result to {}".format(file))
