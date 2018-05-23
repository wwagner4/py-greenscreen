from gs import common as co
import h5py


def train(cfg: co.Conf):
    def run():
        path = co.h5_file(cfg.id)
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

            model.fit(x, y, epochs=4, batch_size=20, shuffle='batch')
            print("Fit model {}".format(model))

            model_file = co.model_file(cfg.id)
            model.save(model_file)

            print("Saved model to {}".format(model_file))

    run()
