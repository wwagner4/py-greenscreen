from typing import Iterable

import matplotlib.pyplot as plt
import gs.common as co
import numpy as np


class XY:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return "XY[x:{} y:{}]".format(self.x, self.y)

    __repr__ = __str__


class DataRow:
    def __init__(self, data: Iterable[XY], name: str = ""):
        self.name = name
        self.data = list(data)

    def __str__(self):
        return "DataRow[name:{} data:{}]".format(self.name, self.data)

    __repr__ = __str__


class Dia:
    def __init__(self, data: Iterable[DataRow], title: str = ""):
        self.title = title
        self.data = data

    def __str__(self):
        return "Dia[title:{} data:{}]".format(self.title, self.title)

    __repr__ = __str__


def plot_dia(dia: Dia, file: str):
    def extract(_data, key: str) -> float:
        return vars(_data).get(key)

    def extract_x(_data) -> float:
        return extract(_data, 'x')

    def extract_y(_data) -> float:
        return extract(_data, 'y')

    fig = plt.figure()
    fig.add_subplot(111)
    for dr in dia.data:
        xs = list(map(extract_x, dr.data))
        ys = list(map(extract_y, dr.data))
        plt.plot(xs, ys, linewidth=0.5, label=dr.name)
        plt.legend()
        if dia.title:
            plt.title(dia.title)

    fig.savefig(fname=file, dpi=300, papertype='a5', format='png')


def plot_multi_dia(dias: Iterable[Dia], rows: int, cols: int, file: str):
    def extract(_data, key: str) -> float:
        return vars(_data).get(key)

    def extract_x(_data) -> float:
        return extract(_data, 'x')

    def extract_y(_data) -> float:
        return extract(_data, 'y')

    _dias = list(dias)
    if len(_dias) == 1:
        plot_dia(_dias[0], file)
    elif len(_dias) > 1:
        fig = plt.figure()
        for i, dia in enumerate(dias):
            fig.add_subplot(rows, cols, i + 1)
            for dr in dia.data:
                xs = list(map(extract_x, dr.data))
                ys = list(map(extract_y, dr.data))
                plt.plot(xs, ys, linewidth=0.5, label=dr.name)
                plt.legend()
                if dia.title:
                    plt.title(dia.title)
        fig.savefig(fname=file, dpi=300, papertype='a5', format='png')


def tryout():
    import math

    def data_a(x: float) -> XY:
        y = x ** 2
        return XY(x, y)

    def data_b(x: float) -> XY:
        y = 100 * math.sin(x)
        return XY(x, y)

    xs = np.arange(-5, 5, 0.1)
    data_a = DataRow(map(data_a, xs), "A")
    data_b = DataRow(map(data_b, xs), "B")

    data = [data_a, data_b]
    dia = Dia(data, "Some Testdata")

    file = co.work_file('tryout_001.png')
    plot_dia(dia, file)
    print("wrote to file:{}".format(file))


def tryout_multi():
    import math

    def data_a(x: float) -> XY:
        y = x ** 2
        return XY(x, y)

    def data_b(x: float) -> XY:
        y = 100 * math.sin(x)
        return XY(x, y)

    xs = np.arange(-5, 5, 0.1)
    data_a = DataRow(map(data_a, xs), "A")
    data_b = DataRow(map(data_b, xs), "B")

    dia_a = Dia([data_a], "Some Test Data")
    dia_b = Dia([data_b])

    file = co.work_file('tryout_multi_001.png')
    plot_multi_dia([dia_a, dia_b], 2, 1, file)
    print("wrote to file:{}".format(file))


tryout()
tryout_multi()
