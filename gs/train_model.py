import h5py

from gs import common as co


def train(work_dir: str, cfg: co.Conf):
    def run():
        path = co.h5_file(work_dir, cfg.id)
        print("reading from '{}'".format(path))
        with h5py.File(path, 'r', libver='latest') as h5_file:
            x = h5_file['dsx']
            y = h5_file['dsy']
            print("x {}".format(x.shape))
            print("y {}".format(y.shape))

            model = cfg.model()
            print("Defined model {}".format(model.to_yaml()))
            print("--------------------------------------------")
            model.compile(loss='binary_crossentropy', optimizer=cfg.optimizer, metrics=['accuracy'])
            print("Compiled model {}".format(model))

            model.fit(x, y, epochs=4, batch_size=cfg.batch_size, shuffle='batch', verbose=0)
            print("Fit model {}".format(model))

            model_file = co.model_file(work_dir, cfg.id)
            model.save(model_file)

            print("Saved model to {}".format(model_file))

    run()
