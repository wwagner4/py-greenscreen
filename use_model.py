import keras.models as km

import common as co


def use(model_file: str):
    print("using model: '{}'".format(model_file))
    model = km.load_model(model_file)
    print("loaded model: '{}'".format(model))

    dim = co.Dim(100, 133)
    print("dim: {}".format(dim))

    delta = 10
    icore = co.core_indices(dim.rows, dim.cols, delta=delta)
    idiff = list(co.square_indices_rows_cols(delta=delta))
