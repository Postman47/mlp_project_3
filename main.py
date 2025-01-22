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
from customMSELoss import CustomMSELoss

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

    train_dataset, test_dataset = torch.utils.data.random_split(WeatherDataset(), [0.80, 0.20])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = MLP(dim_in=1224, dim_out=3, hidden_depth=5, batch_norm=True).to(device)

    regression_loss_fn = CustomMSELoss()
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    size = len(train_dataset)
    best_vloss = 0.0
    for epoch in range(epochs): 
        logging.info('EPOCH {}:'.format(epoch + 1))

        model.train()
        running_loss, last_loss = 0., 0.
        for batch, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            regression_pred, classification_pred = model(X.to(device))
            regression_loss = regression_loss_fn(regression_pred, y.to(device)[:, :, 0])
            classification_loss = classification_loss_fn(classification_pred, y.to(device)[:, :, 1:].squeeze(1))
            loss = regression_loss + classification_loss

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
        regression_correct = 0
        classification_correct = 0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for batch, vdata in enumerate(test_dataloader):
                vinputs, vlabels = vdata
                regression_voutputs, classification_voutputs = model(vinputs.to(device))
                regression_vloss = regression_loss_fn(regression_voutputs, vlabels.to(device)[:, :, 0])
                classification_vloss = classification_loss_fn(classification_voutputs, vlabels.to(device)[:, :, 1:].squeeze(1))
                vloss = regression_vloss + classification_vloss
                running_vloss += vloss

                regression_correct += torch.sum(torch.abs(regression_voutputs - vlabels.to(device)[:, :, 0]) <= 0.025).item()
                classification_predictions = torch.argmax(classification_voutputs, dim=1)
                classification_correct += torch.sum(classification_predictions == torch.argmax(vlabels.to(device)[:, :, 1:].squeeze(1), dim=1)).item()
                total_samples += vlabels.size(0)

        avg_vloss = running_vloss / (batch + 1)
        regresion_acc = (regression_correct / total_samples) * 100
        classification_acc = (classification_correct / total_samples) * 100
        logging.info(f'batch {batch + 1} loss: {loss:>7f} valid_loss: {avg_vloss}')
        logging.info(f'batch {batch + 1} regression accuracy: {regresion_acc:>4}%')
        logging.info(f'batch {batch + 1} classification accuracy: {classification_acc:>4}%')
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