import gs.config as cfg
import gs.common as co
import gs.plot as pl
import keras.optimizers as opt
import h5py


def _run():
    _id = "adam_lr"
    title = "adam learning rate"
    runid = "001"
    epoche_cnt = 4
    learning_rates = [0.00012, 0.00011, 0.00010, 0.00009]
    # learning_rates = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # learning_rates = [0.001, 0.0005]

    epoches = list(range(0, epoche_cnt))
    _cfg = cfg.conf('img100', '..')
    lrmax = len(learning_rates)

    def _train(lr: float, lrnr: int, h5_file) -> pl.DataRow:
        _cfg.optimizer = opt.Adam(lr=lr)

        x = h5_file['dsx']
        y = h5_file['dsy']
        xcv = h5_file['dsx_cross']
        ycv = h5_file['dsy_cross']

        model = _cfg.model()
        model.compile(loss='binary_crossentropy', optimizer=_cfg.optimizer, metrics=['accuracy'])
        data = []
        for i in epoches:
            print("evaluating lr: {}/{} {:.5f} epoche {}/{}".format(lrnr + 1, lrmax, lr, i + 1, epoche_cnt))
            model.fit(x, y, epochs=1, batch_size=20, shuffle='batch')
            score = model.evaluate(xcv, ycv, batch_size=20)[1]
            print("score lr: {}/{} {:.5f} epoche {}/{} {:.2f}".format(lrnr + 1, lrmax, lr, i + 1, epoche_cnt, score))
            data.append(pl.XY(i, score))

        return pl.DataRow(data=data, name="{:7.5f}".format(lr))

    path = co.h5_file(_cfg.id)
    print("reading from '{}'".format(path))
    with h5py.File(path, 'r', libver='latest') as _h5_file:
        data_rows = []
        for _lrnr, _lr in enumerate(learning_rates):
            data_rows.append(_train(_lr, _lrnr, _h5_file))

        file = co.work_file(name="opt_{}_{}.png".format(_id, runid), _dir='opt')
        dia = pl.Dia(data=data_rows, title=title)
        pl.plot_dia(dia, file=file)
        print("plot result to {}".format(file))
