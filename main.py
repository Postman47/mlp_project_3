import torch
from nn import MLP
import numpy as np
import random

from dataset import WeatherDataset
from torch.utils.data import DataLoader

import logging
from absl import app

def main(argv):
    del argv
    seed = 0
    torch.random.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    learning_rate = 1e-4
    batch_size = 64
    epochs = 5

    train_dataset = WeatherDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    model = MLP(dim_in=51, dim_out=17, batch_norm=True)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    size = len(train_dataset)
    for epoch in range(epochs): 
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):

            pred = model(X).unsqueeze(1)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                logging.info(f"loss: {loss:0.5e}  [{current:>5d}/{size:>5d}]")

if __name__ == '__main__':
    app.run(main)