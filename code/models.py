import os

from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np

from utils import save_dict
from dataset import *

# @TODO: custom teacher forcing module, drop-in for GRU, LSTM, transformer(?): TeacherForcing(nn.GRU(), teachingRatio=.5)

class XRayEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1), # 64
            nn.Conv2d(8, 16, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1), # 16
            nn.Conv2d(16, 32, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1), # 4
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1), # 1
            nn.Flatten()
        )

    def forward(self, images):
        return self.image_net(images)


class XRayDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input, context):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, context[None])
        output = self.out(output)
        output = self.softmax(output)
        return output, hidden


class XRayBaseModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=64):
        super().__init__()

        self.encoder = XRayEncoder()
        self.decoder = XRayDecoder(hidden_size, vocabulary_size)
        
    def forward(self, text, images):
        context = self.encoder(images)
        output, _ = self.decoder(text, context)
        return output


def train(model_name, model, vocabulary, train_dataset, validation_dataset, learning_rate=0.01, epochs=30):
    os.makedirs(os.path.join("results", model_name), exist_ok=True)

    token2id, id2token = map_token_and_id_fn(vocabulary) # @TODO: take token2id as arguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    result = {}

    # prepare dataloaders
    collate_fn = lambda input: report_collate_fn(token2id("[PAD]"), input)

    train_dl = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    validation_dl = DataLoader(validation_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    mean_train_losses = []
    mean_validation_losses = []

    for t in range(epochs):
        logging.info(f"Epoch {t}.")

        model.train()

        train_losses = []

        for xrays, reports, report_lengths in tqdm(train_dl):
            xrays = xrays.to(device)
            reports = reports.to(device)
            report_lengths = report_lengths.to(device)

            y_pred = model(reports, xrays)

            # Since a LM predicts the next token, we need shift the tokens. Tokens "!   !" should be ignored.
            # y_true: !Start!   <hello>   <sailor>  <!>      <[End]>
            # y_est:  <hello>   <sailor>  <!>       <[END]>  !misc!
            y_pred_align = y_pred[:,:-1,:]
            y_true_align = reports[:,1:]

            loss = criterion(y_pred_align.flatten(end_dim=1), y_true_align.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())


        with torch.no_grad():
            model.eval()

            validation_losses = []

            for xrays, reports, report_lengths in validation_dl:
                xrays = xrays.to(device)
                reports = reports.to(device)
                report_lengths = report_lengths.to(device)

                y_pred = model(reports, xrays)
                loss = criterion(y_pred.flatten(end_dim=1), reports.flatten())

                validation_losses.append(loss.detach().cpu().numpy())


        mean_train_loss = np.mean(train_losses)
        mean_train_losses.append(mean_train_loss)

        mean_validation_loss = np.mean(validation_losses)
        mean_validation_losses.append(mean_validation_loss)

        print(f"Epoch {t+1} train loss: {mean_train_loss:.3f}, validation loss: {mean_validation_loss:.3f}")
    

    result["train_losses"] = mean_train_losses
    result["validation_losses"] = mean_validation_losses

    save_dict(result, os.path.join("results", model_name, "result.pkl"))

    torch.save(model.state_dict(), os.path.join("results", model_name, "model.pt"))