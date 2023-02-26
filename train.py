import os
import math
from time import time

import torch

import data
import config as cfg
from model import RNN


DATA = data.Poetries(cfg.batch)
LOSS = torch.nn.CrossEntropyLoss().cuda(0)
MODEL = RNN(
    cfg.num_layers, cfg.hidden_dim, cfg.embedding_dim, data.DICTIONARY_SIZE
).cuda(0)
MODEL_DIR = os.path.join("model", "parameters")
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), cfg.lr)
SCHEDULER = torch.optim.lr_scheduler.LambdaLR(
    OPTIMIZER, lambda e: math.cos(0.5 * epoch * math.pi / cfg.epoch)
)
if "parameters" not in os.listdir("model"):
    os.mkdir(MODEL_DIR)


def log(head: str = "床前明月光"):
    MODEL.eval()
    result = list(head)
    y, h = MODEL(torch.tensor([[data.encode[char] for char in head]]).cuda(0))
    while len(result) <= cfg.max_len:
        y = y[-1:, :]
        x = torch.argmax(y[0])
        result.append(data.decode[x.item()])
        if result[-1] == data.END:
            break
        y, h = MODEL(x.reshape((1, -1)), h)
    MODEL.train()
    with open("log.txt", "a", encoding="utf-8") as output:
        output.write(f"Epoch: {epoch + 1}\t")
        output.write("".join(result))
    with open(os.path.join(MODEL_DIR, "checkpoint.txt"), "w") as output:
        output.write(str(epoch))


def progress():
    n = (epoch - EPOCH) * len(DATA) + min((step + 1) * cfg.batch, len(DATA))
    t = round(((cfg.epoch - EPOCH) * len(DATA) - n) / (n / (time() - TIME)))
    n = 100 * (n + EPOCH * len(DATA)) / (cfg.epoch * len(DATA))
    print("\rEpoch: {}/{} [{}>{}]{:.2f}% loss:{:.2f} eta {}:{:02}:{:02}".format(
        epoch + 1, cfg.epoch, "-" * (round(n) >> 1),
        "." * (50 - (round(n) >> 1)), n, loss.item(),
        t // 3600, (t % 3600) // 60, t % 60
    ), end=" " * 4)


if "last.pt" in os.listdir(MODEL_DIR):
    EPOCH = int(open(os.path.join(MODEL_DIR, "checkpoint.txt")).read())
    SCHEDULER.last_epoch = EPOCH
    OPTIMIZER.step()
    SCHEDULER.step()
    OPTIMIZER.zero_grad()
    MODEL.load_state_dict(torch.load(os.path.join(MODEL_DIR, "last.pt")))
else:
    EPOCH = -1

TIME = time()

for epoch in range(EPOCH + 1, cfg.epoch):
    for step, poetries in enumerate(DATA):
        poetries = torch.tensor(poetries).cuda(0)
        loss = LOSS(
            MODEL(poetries[:, :-1])[0],
            poetries[:, 1:].flatten()
        )
        progress()
        loss.backward()
        OPTIMIZER.step()
        OPTIMIZER.zero_grad()
    SCHEDULER.step()
    torch.save(
        MODEL.state_dict(),
        os.path.join(MODEL_DIR, "last.pt")
    )
    log()
torch.save(MODEL.state_dict(), os.path.join(MODEL_DIR, "final.pt"))
