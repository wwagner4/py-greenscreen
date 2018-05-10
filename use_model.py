import keras.models as km

import common as co


def use(model_file: str):
    print("using model: '{}'".format(model_file))
    model = km.load_model(model_file)
    print("loaded model: '{}'".format(model))

    dim = co.Dim(100, 133)
    print("dim {}".format(dim))
