import numpy as np

import gs.common as co
from gs.plot import *


def tryout(work_dir: str):
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

    file = co.work_file(work_dir, 'tryout_plot_002.png', _dir="plot-tryout")
    plot_dia(dia, file)
    print("wrote to file:{}".format(file))


def tryout_multi(work_dir: str):
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

    file = co.work_file(work_dir=work_dir, name='tryout_plot_multi_002.png', _dir="plot_tryout")
    plot_multi_dia([dia_a, dia_b], 1, 2, file, img_size=(3000, 3000))
    print("wrote to file:{}".format(file))
