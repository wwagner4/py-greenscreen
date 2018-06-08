from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gs import common as co


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


class Axis:
    def __init__(self, title: str = "", lim: Tuple = None):
        self.title = title
        self.lim = lim

    def __str__(self):
        return "Axis[title:{} lim:{}]".format(self.title, self.lim)

    __repr__ = __str__


class Dia:
    def __init__(self, data: Iterable[DataRow], title: str = "", xaxis: Axis = Axis(), yaxis: Axis = Axis()):
        self.title = title
        self.data = data
        self.xaxis = xaxis
        self.yaxis = yaxis

    def __str__(self):
        return "Dia[title:{} data:{}]".format(self.title, self.title)

    __repr__ = __str__


def plot_dia(dia: Dia, file: str, img_size: Tuple = None):
    fig = _create_figure(img_size)
    fig.add_subplot(111)
    _plot_dia(dia)
    fig.savefig(fname=file, format='png')


def plot_multi_dia(dias: Iterable[Dia], rows: int, cols: int, file: str, img_size: Tuple = None):
    _dias = list(dias)
    if len(_dias) == 1:
        plot_dia(_dias[0], file)
    elif len(_dias) > 1:
        fig = _create_figure(img_size)
        for i, dia in enumerate(dias):
            fig.add_subplot(rows, cols, i + 1)
            _plot_dia(dia)
        plt.tight_layout()
        fig.savefig(fname=file, format='png')


def _create_figure(img_size: Tuple) -> plt.Figure:
    if isinstance(img_size, Tuple):
        base = 12.0
        ratio = img_size[1] / img_size[0]
        x = base
        y = base * ratio
        dpi = img_size[0] / base
        return plt.figure(figsize=(x, y), dpi=dpi)
    return plt.figure()

def _plot_dia(dia: Dia):
    def extract(_data, key: str) -> float:
        return vars(_data).get(key)

    def extract_x(_data) -> float:
        return extract(_data, 'x')

    def extract_y(_data) -> float:
        return extract(_data, 'y')

    for dr in dia.data:
        xs = list(map(extract_x, dr.data))
        ys = list(map(extract_y, dr.data))
        plt.plot(xs, ys, linewidth=1.0, label=dr.name)
        plt.legend()
        if dia.xaxis.title:
            plt.xlabel(dia.xaxis.title)
        if isinstance(dia.xaxis.lim, Tuple):
            plt.xlim(dia.xaxis.lim)

        if dia.yaxis.title:
            plt.ylabel(dia.yaxis.title)
        if isinstance(dia.yaxis.lim, Tuple):
            plt.ylim(dia.yaxis.lim)

        if dia.title:
            plt.title(dia.title)


def _tryout():
    import math

    def data_a(x: float) -> XY:
        y = x ** 2
        return XY(x, y)

    def data_b(x: float) -> XY:
        y = 10 * x * math.sin(x)
        return XY(x, y)

    xs = np.arange(0, 15, 0.1)
    data_a = DataRow(map(data_a, xs), "power 2")
    data_b = DataRow(map(data_b, xs), "sinus")

    data = [data_a, data_b]
    dia = Dia(data, "Some Testdata",
              xaxis=Axis(title="x axis", lim=(2, 20)),
              yaxis=Axis(lim=(-500, 500)))

    file = co.work_file('tryout_plot_002.png')
    plot_dia(dia, file)
    print("wrote to file:{}".format(file))


def _tryout_multi():
    import math

    def data_a(x: float) -> XY:
        y = x ** 2
        return XY(x, y)

    def data_b(x: float) -> XY:
        y = x * math.sin(x)
        return XY(x, y)

    def data_c(x: float) -> XY:
        y = x * math.sin(x * 1.1)
        return XY(x, y)

    xs = np.arange(-5, 5, 0.1)
    data_a = DataRow(map(data_a, xs), "A")
    data_b = DataRow(map(data_b, xs), "B")
    data_c = DataRow(map(data_c, xs), "C")

    dia_a = Dia([data_a], "Some Test Data",
                xaxis=Axis("xxx aaa", lim=(-10, 10)),
                yaxis=Axis(lim=(0, 40)))
    dia_b = Dia([data_b, data_c], 'Sinus', yaxis=Axis("amplitude"))

    file = co.work_file('tryout_plot_multi_002.png')
    plot_multi_dia([dia_a, dia_b], 1, 2, file, img_size=(3000, 3000))
    print("wrote to file:{}".format(file))


if __name__ == "__main__":
    _tryout()
    _tryout_multi()
