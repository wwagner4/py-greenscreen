from os import listdir


def is_jpg_file(file_name: str):
    return file_name.upper().endswith('JPG')


def transform_images(di: str):
    print("transforming images in '{}'".format(di))
    onlyfiles = [f for f in listdir(di) if is_jpg_file(f)]
    for file in onlyfiles:
        print("File: '{}'".format(file))


transform_images("/Users/wwagner4/Pictures/Diverses/Wolfis Experimente/Green Screen/100")
