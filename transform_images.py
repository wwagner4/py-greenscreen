from os import listdir
from os.path import join

from wand.image import Image


def is_jpg_file(file_name: str):
    return file_name.upper().endswith('JPG')


def transform(di: str, fi: str):
    base = fi[:-4]
    print("transforming image '{}' base '{}' in {}".format(fi, base, di))
    with Image(filename=join(di, fi)) as img:
        img.format = 'png'
        img.save(filename=join(di, base + ".png"))



def transform_images(di: str):
    jpg_images = [f for f in listdir(di) if is_jpg_file(f)]
    for file in jpg_images:
        transform(di, file)


transform_images("/Users/wwagner4/Pictures/Diverses/Wolfis Experimente/Green Screen/100")
