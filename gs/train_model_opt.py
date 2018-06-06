import gs.config as cfg
import gs.common as co
import gs.plot as pl
import keras.optimizers as opt
import h5py


def _run():
    epoche_cnt = 10
    batch_sizes = [5, 10, 20, 40]
    runid = "w001"
    learning_rates = [0.00020, 0.00015, 0.00010, 0.00005]

    epoches = list(range(0, epoche_cnt))
    _cfg = cfg.conf('img100', '..')
    lrmax = len(learning_rates)

    def _train(lr: float, lrnr: int, batch_size: int, h5_file) -> pl.DataRow:
        _cfg.optimizer = opt.Adam(lr=lr)

        x = h5_file['dsx']
        y = h5_file['dsy']
        xcv = h5_file['dsx_cross']
        ycv = h5_file['dsy_cross']

        model = _cfg.model()
        model.compile(loss='binary_crossentropy', optimizer=_cfg.optimizer, metrics=['accuracy'])
        data = []
        for epoche in epoches:
            model.fit(x, y, epochs=1, batch_size=batch_size, shuffle='batch', verbose=0)
            score = model.evaluate(xcv, ycv, batch_size=batch_size, verbose=0)[1]
            print("score batch size: {} lr: {}/{} {:.5f} epoche {}/{} {:.4f}"
                  .format(batch_size, lrnr + 1, lrmax, lr, epoche + 1, epoche_cnt, score))
            data.append(pl.XY(epoche, score))

        return pl.DataRow(data=data, name="{:7.5f}".format(lr))

    def _train01(batch_size: int, _h5_file) -> pl.Dia:
        data_rows = []
        title = "adam, batch size {}".format(batch_size)
        for _lrnr, _lr in enumerate(learning_rates):
            data_rows.append(_train(_lr, _lrnr, batch_size, _h5_file))
        return pl.Dia(data=data_rows, title=title,
                      xaxis=pl.Axis(title="epoche"),
                      yaxis=pl.Axis(lim=(0.8, 1.0)))

    path = co.h5_file(_cfg.id)
    print("reading from '{}'".format(path))
    with h5py.File(path, 'r', libver='latest') as _h5_file:

        dias = []
        for bs in batch_sizes:
            dia = _train01(bs, _h5_file)
            dias.append(dia)

        file = co.work_file(name="opt_adam_{}.png".format(runid), _dir='opt')
        pl.plot_multi_dia(dias, rows=2, cols=2, file=file, img_size=(3000, 2000))
        print("plot result to {}".format(file))
