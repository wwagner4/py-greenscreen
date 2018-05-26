from typing import List

import matplotlib.pyplot as plt
import gs.common as co


class Data:
    def __init__(self, grp: str, x: float, y: float):
        self.grp = grp
        self.x = x
        self.y = y


def plot(data: List[Data]):
    def extract(_data, key: str) -> float:
        return vars(_data).get(key)

    def extract_x(_data) -> float:
        return extract(_data, 'x')

    def extract_y(_data) -> float:
        return extract(_data, 'y')

    xs = list(map(extract_x, data))
    ys = list(map(extract_y, data))

    for x in xs:
        print("x:{}".format(x))
    for y in ys:
        print("y:{}".format(y))

    fig = plt.figure()
    fig.add_subplot(111)

    plt.plot(xs, ys, linewidth=0.5, label='label')
    plt.legend()

    file = co.work_file('p1.png')
    fig.savefig(fname=file, dpi=300, papertype='a7', format='png')
    print("saved img to {}".format(file))


def run1():
    d1 = Data("a", 0.0, 0.0)
    vd1 = vars(d1)
    for k in vd1.keys():
        v = vd1.get(k)
        print("{} = {}".format(k, v))

    data_a = [
        Data('a', 0.0, 0.1),
        Data('a', 1.0, 0.2),
        Data('a', 2.0, 0.5),
        Data('a', 3.0, 0.7),
        Data('a', 4.0, 0.6),
        Data('a', 5.0, 0.8)
    ]
    plot(data_a)


run1()
