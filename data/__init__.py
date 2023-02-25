from random import shuffle

from tqdm import tqdm as _tqdm

__all__ = [
    "END",
    "encode",
    "decode",
    "Poetries",
    "DICTIONARY_SIZE"
]

END = "\n"

with open("data/poetries.txt", encoding="utf-8") as _poetries:
    _data = _poetries.read().rstrip("\n")

encode, decode = {" ": 0}, [" "]
for _code, _char in enumerate(sorted(set(_data))):
    decode.append(_char)
    encode[_char] = _code + 1

DICTIONARY_SIZE = len(decode)

_poetries = [
    [encode[_char] for _char in list(_poetry) + [END]] for _poetry in
    _tqdm(_data.split("\n"), "Loading dataset", leave=False, unit="poetries")
]

del _char, _code, _data, _tqdm


class Poetries:
    def __init__(self, batch: int):
        self.batch = batch
        self.poetries = _poetries

    def __len__(self):
        return len(self.poetries)

    def __iter__(self):
        poetries = []
        shuffle(self.poetries)
        for item, poetry in enumerate(self.poetries):
            poetries.append(poetry)
            if len(poetries) == self.batch or item + 1 == len(self):
                length = max(map(len, poetries))
                yield [[0] * (length - len(poem)) + poem for poem in poetries]
                poetries.clear()
