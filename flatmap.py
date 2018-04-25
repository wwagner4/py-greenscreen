import itertools as it


def flatmap(f, list_of_list):
    return it.chain.from_iterable(map(f, list_of_list))


e2 = [
    [1, 2, 3],
    [4, 5]
]

e3 = [
    range(0, 3),
    range(6, 9)
]


def generator():
    for i in range(0, 3):
        yield range(i, 5)


for x in generator():
    print("gen {}".format(x))

flattened = flatmap(
    lambda _list: [elem + 100 for elem in _list], generator())

print("flattened {}".format(flattened))

for ep in flattened:
    print("flattened elem '{}'".format(ep))
