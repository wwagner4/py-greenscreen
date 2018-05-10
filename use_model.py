import keras.models as km

import common as co


def use(model_file: str):
    print("using model: '{}'".format(model_file))
    model = km.load_model(model_file)
    print("loaded model: '{}'".format(model))

    dim = co.Dim(100, 133)
    print("dim: {}".format(dim))

    delta = 10

    img = co.load_image("res/img100/DSCN1834.png", dim)

    icore = co.core_indices(dim.rows, dim.cols, delta=delta)
    idiff = list(co.square_indices_rows_cols(delta=delta))
    for r, c in icore:
        print("({},{}) {}".format(r, c, img[r, c]))
