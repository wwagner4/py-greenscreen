from typing import Iterable, Tuple

import matplotlib.pyplot as plt


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
