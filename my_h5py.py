import h5py
import numpy as np


def write():
    with h5py.File("/Users/wwagner4/work/work-greenscreen/dummy001.h5py", 'a', libver='latest') as dummy001:
        print("file %s" % dummy001)

        a1 = np.arange(0, 17, 4)

        d1 = dummy001.create_dataset(name="d6", data=a1)
        print("d1 %s" % d1)

        print("dummy001 len %s" % len(dummy001))
        for k in dummy001.keys():
            print(k)

        for v in dummy001.values():
            print(v)

        dummy001.flush()


def read():
    with h5py.File("/Users/wwagner4/work/work-greenscreen/dummy001.h5py", 'a', libver='latest') as dummy001:
        print("file --> %s" % dummy001)

        print("dummy001 len %s" % len(dummy001))
        for k in dummy001.keys():
            print(k)

        for v in dummy001.values():
            print(v)

        d = dummy001['d2']
        print("d[0] %s" % d[1])
        for v in d:
            print("v = %10d" % v)


read()
