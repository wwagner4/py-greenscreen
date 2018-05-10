import keras.models as km


def use(model_file: str):
    print("using model: '{}'".format(model_file))
    model = km.load_model(model_file)
    print("loaded model: '{}'".format(model))
