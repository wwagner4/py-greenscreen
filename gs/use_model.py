from gs import common as co
import keras.models as km
import numpy as np
import os
import os.path as osp
from PIL import Image


def use(model_file: str):
    def plot_image(img: np.array, path: str):
        img1 = (img * 256).astype(np.uint8)
        image = Image.fromarray(img1, mode='RGBA')
        image.save(path, format='PNG')

    def use_file(model: km.Sequential, img_file: str, dim: co.Dim, delta: int):
        print("predicting {}".format(img_file))
        name = osp.basename(img_file)
        base, _ = osp.splitext(name)
        print("base name {}".format(base))

        img = co.load_image(img_file, dim)

        timg = np.zeros((dim.rows, dim.cols, 4), dtype=np.float32)

        icore = co.core_indices(dim.rows, dim.cols, delta=delta)
        idiff = list(co.square_indices_rows_cols(delta=delta))
        cnt = 0
        for r, c in icore:
            f = co.create_features(img, r, c, idiff)
            prediction = model.predict(np.array(np.array([f])))[0]
            col: np.array = img[r, c]
            col1 = np.insert(col, 3, prediction)
            timg[r, c] = col1
            if cnt % 500 == 0 and cnt > 0:
                print("made {} predictions for {}. latest value {}".format(cnt, img_file, prediction))
            cnt = cnt + 1

        out = co.work_file(base + ".png")
        plot_image(timg, out)
        print("Plot image {}".format(out))

    def use_img100():
        print("using model: '{}'".format(model_file))
        model: km.Sequential = km.load_model(model_file)
        print("loaded model: '{}'".format(model))

        dim = co.Dim(100, 133)
        print("dim: {}".format(dim))

        delta = 10

        for file in os.listdir(osp.join("res", "img100")):
            if file.startswith("DSCN"):
                img_file = os.path.join("res", "img100", file)
                use_file(model, img_file, dim, delta)

    use_img100()
