from typing import List

import matplotlib.pyplot as plt
import gs.common as co


class Data:
    def __init__(self, grp: str, x: float, y: float):
        self.grp = grp
        self.x = x
        self.y = y

    def __str__(self):
        return "Data[grp:{} x:{} y:{}]".format(self.grp, self.x, self.y)

    __repr__ = __str__


class Plotter:

    def _group(self, data_list: List[Data]) -> List[List[Data]]:
        grps = set(map(lambda x: x.grp, data_list))
        return [[dat for dat in data_list if dat.grp == grp] for grp in grps]

    def plot(self, data: List[Data], file: str):
        def extract(_data, key: str) -> float:
            return vars(_data).get(key)

        def extract_x(_data) -> float:
            return extract(_data, 'x')

        def extract_y(_data) -> float:
            return extract(_data, 'y')

        fig = plt.figure()
        fig.add_subplot(111)
        for grp in self._group(data):
            xs = list(map(extract_x, grp))
            ys = list(map(extract_y, grp))
            label = grp[0].grp
            plt.plot(xs, ys, linewidth=2, label=label)
            plt.legend()

        fig.savefig(fname=file, dpi=300, papertype='a7', format='png')
        print("saved img to {}".format(file))


def tryout01():
    data_a = [
        Data('a', 0.0, 0.1),
        Data('a', 1.0, 0.2),
        Data('a', 2.0, 0.5),
        Data('b', 1.0, 0.7),
        Data('b', 4.0, 0.6),
        Data('b', 5.0, 0.8)
    ]
    file = co.work_file('tryout01.png')
    Plotter().plot(data_a, file)


tryout01()
