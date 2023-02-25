from random import random

import torch

import data
import config as cfg
from model import RNN


__all__ = ["Poet"]


class Poet:
    def __init__(self, pt: int):
        self.rnn = RNN(
            cfg.num_layers,
            cfg.hidden_dim,
            cfg.embedding_dim,
            data.DICTIONARY_SIZE
        )
        self.rnn.eval()
        self.rnn.load_dict(torch.load(f"model/parameters/{pt}.pt"))

    def renewal(self, head: str) -> str:
        poetry = list(head)
        y, h = self.rnn(torch.tensor([[data.encode[char] for char in head]]))
        while len(poetry) <= cfg.max_len:
            y = y[-1:, :]
            x = torch.argmax(y[0])
            char = data.decode[x.item()]
            if char == data.END:
                break
            poetry.append(char)
            y, h = self.rnn(x.reshape((1, -1)), h)
        return "".join(poetry)

    def acrostic(self, head: str) -> str:
        poetry, h = [], None
        punctuations = "，；。！？"
        if len(head) % 2 or random() < 0.5:
            punctuations = punctuations[2:]
        for char in head:
            poetry.append(char)
            y, h = self.rnn(torch.tensor([[data.encode[char]]]), h)
            while poetry[-1] not in punctuations:
                x = torch.argmax(y[0])
                poetry.append(data.decode[x.item()])
                y, h = self.rnn(x.reshape((1, -1)), h)
            poetry.append("\n")
        return "".join(poetry)
