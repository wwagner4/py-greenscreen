from itertools import chain

episodes = [
    {"id": 1, "topics": [1, 2, 3]},
    {"id": 2, "topics": [4, 5, 6]}
]


def flatmap(f, items):
    return chain.from_iterable(map(f, items))


flattened_episodes = flatmap(
    lambda _episode: [{"id": _episode["id"], "topic": topic} for topic in _episode["topics"]], episodes)

for ep in flattened_episodes:
    print(ep)
