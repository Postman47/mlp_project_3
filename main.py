import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import WeatherDataset
from nn import MLP

import logging
from absl import app
from datetime import datetime
import numpy as np
import random

def main(argv):
    del argv
    seed = 0
    torch.random.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    learning_rate = 1e-4
    batch_size = 64
    epochs = 5
    logging_freq = 100
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/weather_trainer_{}'.format(timestamp))

    train_dataset, test_dataset = torch.utils.data.random_split(WeatherDataset(), [0.03, 0.97])
    test_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = MLP(dim_in=51, dim_out=17, hidden_depth=5, batch_norm=True).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    size = len(train_dataset)
    best_vloss = 0.0
    for epoch in range(epochs): 
        logging.info('EPOCH {}:'.format(epoch + 1))

        model.train()
        running_loss, last_loss = 0., 0.
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            pred = model(X.to(device)).unsqueeze(1)
            loss = loss_fn(pred, y.to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch % logging_freq == logging_freq - 1:
                last_loss = running_loss / 100
                current = batch*batch_size + len(X)
                logging.info(f'batch {batch + 1} loss: {last_loss:>7f}  [{current:>5d}/{size:>5d}]')
                tb_x = epoch * len(train_dataloader) + batch + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for batch, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs.to(device)).unsqueeze(1)
                vloss = loss_fn(voutputs, vlabels.to(device))
                running_vloss += vloss

        avg_vloss = running_vloss / (batch + 1)
        logging.info(f'batch {batch + 1} loss: {loss:>7f} valid_loss: {avg_vloss}')
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : last_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    app.run(main)