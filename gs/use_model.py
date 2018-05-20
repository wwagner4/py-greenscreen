from gs import common as co
import keras.models as km
import numpy as np
import os
import os.path as osp
from PIL import Image


def use(_id: str, timestamp: str):
    cfg = co.conf(_id)
    in_dir = osp.join("res", _id)
    out_dir = co.work_dir("{}_{}".format(_id, timestamp))

    def plot_image(img: np.array, path: str):
        img1 = (img * 256).astype(np.uint8)
        image = Image.fromarray(img1, mode='RGBA')
        image.save(path, format='PNG')

    def use_file(model: km.Sequential, img_file: str):
        print("predicting {}".format(img_file))
        name = osp.basename(img_file)
        base, _ = osp.splitext(name)
        print("base name {}".format(base))

        img = co.load_image(img_file, cfg.dim)

        xrows, xcols = co.features_shape(cfg)
        x = np.zeros((xrows, xcols), dtype=np.float32)

        icore = co.core_indices(cfg.dim.rows, cfg.dim.cols, delta=cfg.delta)
        cnt = 0
        for r, c in icore:
            features = co.create_features(img, r, c, cfg.around_indices)
            x[cnt] = features
            if cnt % 5000 == 0 and cnt > 0:
                print("Added {} features to the feature vextor x".format(cnt))
            cnt = cnt + 1

        print("Created the feature vector x: {}".format(x.shape))
        predictions = model.predict(x)
        print("Created predictions: {}".format(predictions.shape))

        timg = np.zeros((cfg.dim.rows, cfg.dim.cols, 4), dtype=np.float32)
        icore = co.core_indices(cfg.dim.rows, cfg.dim.cols, delta=cfg.delta)
        cnt = 0
        for r, c in icore:
            col: np.array = img[r, c]
            col1 = np.insert(col, 3, predictions[cnt])
            timg[r, c] = col1
            if cnt % 5000 == 0 and cnt > 0:
                print("wrote {} predictions for {}. latest value {}".format(cnt, img_file, predictions[r]))
            cnt = cnt + 1

        out = osp.join(out_dir, base + ".png")
        print("Plot image {}".format(out))
        plot_image(timg, out)

    def run():
        model_file = co.model_file(_id)
        print("using model: '{}'".format(model_file))
        model: km.Sequential = km.load_model(model_file)
        print("input directory: '{}'".format(in_dir))
        print("output directory: '{}'".format(out_dir))
        for file in os.listdir(in_dir):
            print("conduct file {}".format(file))
            if file.startswith("DSCN"):
                img_file = osp.join(in_dir, file)
                use_file(model, img_file)

    run()
