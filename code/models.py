import os
import math

from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights, vit_b_16, ViT_B_16_Weights


import numpy as np

from utils import save_dict
from dataset import *

#
# Model 0
#

class XRayPlaygroundEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(8, 16, kernel_size=4, stride=4),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=1),
            nn.LeakyReLU(0.1), # 64x7x7 -> 128x3x3

            nn.Conv2d(64, 128, kernel_size=3, stride=3),
            nn.LeakyReLU(0.1), # 64x7x7 -> 128x1x1

            nn.Flatten(),
        )

    def forward(self, images):
        return self.image_net(images)

    def preprocess(self, images):
        return images.unsqueeze(1)


class XRayPlaygroundModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=128):
        super().__init__()

        self.encoder = XRayPlaygroundEncoder()
        self.decoder = XRayDecoder(hidden_size, vocabulary_size)
        
    def forward(self, text, images):
        context = self.encoder(images)
        output, _ = self.decoder(text, context)
        return output



#
# Model 1
#

class XRayEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(8, 16, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),

            nn.Flatten(),
        )

    def forward(self, images):
        return self.image_net(images)
    
    def preprocess(self, images):
        return images.unsqueeze(1)


class XRayDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, context):
        x = self.embedding(input)
        x = F.relu(x)
        x, h = self.gru(x, context[None])
        x = self.out(x)
        return x, h


class XRayBaseModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=3200):
        super().__init__()

        self.encoder = XRayEncoder()
        self.decoder = XRayDecoder(hidden_size, vocabulary_size)
        
    def forward(self, text, images):
        context = self.encoder(images)
        x, _ = self.decoder(text, context)
        return x

#
# Model 2, @TODO: add decoder
#

class XRayVGG19Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = VGG19_Weights.DEFAULT
        self.preprocess_fn = weights.transforms()

        self.encoder = vgg19(weights=VGG19_Weights(weights))

    def forward(self, images):
        return self.encoder.features[:35](images) # last conv layer, similiar to the paper.
    
    def preprocess(self, images):
        expanded = images.unsqueeze(1).expand(-1, 3, -1, -1)
        return self.preprocess_fn(expanded)


#
# Model 3
#

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class XRayViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT
        self.preprocess_fn = weights.transforms()

        self.encoder = vit_b_16(weights=ViT_B_16_Weights(weights))

    def forward(self, images):
        # Reshape and permute the input tensor
        images = self.encoder._process_input(images)
        n = images.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.encoder.class_token.expand(n, -1, -1)
        images = torch.cat([batch_class_token, images], dim=1)

        images = self.encoder.encoder(images)

        return images
    
    def preprocess(self, images):
        expanded = images.unsqueeze(1).expand(-1, 3, -1, -1)
        return self.preprocess_fn(expanded)


class XRayViTDecoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=768):
        super().__init__()

        self.positional_encoding = PositionalEncoding(hidden_size, dropout=0.1, max_len=5000)
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input, context):
        x = self.embedding(input)
        x = self.positional_encoding(x)
        x = self.decoder_layer(x, context, memory_is_causal=True)
        return self.linear(x)


class XRayViTModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=768):
        super().__init__()

        self.encoder = XRayViTEncoder()
        self.decoder = XRayViTDecoder(vocabulary_size, hidden_size)
        
    def forward(self, text, images):
        context = self.encoder(images)
        output = self.decoder(text, context)
        return output


def train(model_name, model, vocabulary, train_dataset, validation_dataset, 
          epochs, lr, batch_size, weight_decay):
    
    os.makedirs(os.path.join("results", model_name), exist_ok=True)

    token2id, _ = map_token_and_id_fn(vocabulary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    result = {}

    # prepare dataloaders
    collate_fn = lambda input: report_collate_fn(token2id("[PAD]"), input)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_dl = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # hyperparameters
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=token2id("[PAD]"))

    mean_train_losses = []
    mean_validation_losses = []

    save_model_every = 10

    for t in range(epochs):
        model.train()

        train_losses = []

        for xrays, reports, report_lengths in tqdm(train_dl):
            xrays = xrays.to(device)
            reports = reports.to(device)
            report_lengths = report_lengths.to(device)

            y_pred = model(reports, xrays)

            # Since a LM predicts the next token, we need shift the tokens. Tokens "!   !" should be ignored.
            # y_est:  <hello>   <sailor>  <!>       <[END]>  !misc!
            # y_true: !Start!   <hello>   <sailor>  <!>      <[End]>
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

        if t % save_model_every == 0:
            torch.save(model.state_dict(), os.path.join("results", model_name, f"model_{t}.pt"))
    

    result["train_losses"] = mean_train_losses
    result["validation_losses"] = mean_validation_losses

    save_dict(result, os.path.join("results", model_name, "result.pkl"))

    torch.save(model.state_dict(), os.path.join("results", model_name, "model.pt"))

