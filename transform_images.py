from os import *
from os.path import join

from wand.image import Image


def is_jpg_file(file_name: str):
    return file_name.upper().endswith('JPG')


def is_png_file(file_name: str):
    return file_name.upper().endswith('PNG')


def transform_file(di: str, fi: str):
    base = fi[:-4]
    fipng = base + ".png"
    print("transforming {} --> {}".format(fi, fipng))

    with Image(filename=join(di, fi)) as img:
        img.format = 'png'
        img.save(filename=join(di, fipng))


def scale_file(di: str, fi: str):
    print("scaling {}".format(fi))

    with Image(filename=join(di, fi)) as img:
        k = 100.0 / img.height
        w1 = int(k * img.width)
        h1 = int(k * img.height)
        print("resize {} to {} x {}".format(fi, w1, h1))
        img.resize(w1, h1)
        img.save(filename=join(di, fi))


def transform_dir(di: str):
    jpg_images = [f for f in listdir(di) if is_jpg_file(f)]
    for file in jpg_images:
        transform_file(di, file)


def scale_dir(di: str):
    images = [f for f in listdir(di) if is_png_file(f)]
    for file in images:
        scale_file(di, file)


def deletejpg_dir(di: str):
    images = [f for f in listdir(di) if is_jpg_file(f)]
    for file in images:
        fn = join(di, file)
        print("deleting {}".format(fn))
        remove(fn)


def transform_images(di: str):
    transform_dir(di)
    scale_dir(di)
    deletejpg_dir(di)


transform_images("/Users/wwagner4/Pictures/Diverses/Wolfis Experimente/Green Screen/100")
