import os
import math
import copy

from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Normalize

import pickle 

import numpy as np

from utils import save_dict
from dataset import *
from nlp import *
import patched_transformer as pm

#
# Toy models
#

# Model 0

class PlaygroundEncoder(nn.Module):
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


class PlaygroundDecoder(nn.Module):
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


class PlaygroundModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=128):
        super().__init__()

        self.encoder = PlaygroundEncoder()
        self.decoder = PlaygroundDecoder(hidden_size, vocabulary_size)
        
    def forward(self, text, images):
        context = self.encoder(images)
        output, _ = self.decoder(text, context)
        return output
    
    def preprocess(self, images):
        return images.unsqueeze(1)
    
    def cached_emitter(self, image):
        with torch.no_grad():
            self.eval()
            context = self.encoder(image[None])

        def emitter(tokens):
            with torch.no_grad():
                self.eval()
                out, _ = self.decoder(tokens[None], context)
                return F.log_softmax(out[0,-1,:], dim=-1)
    
        return emitter


#
# CheXNet Models

def load_CheXNet():
    """
        Reference:
        https://github.com/jrzech/reproduce-chexnet
    """
    checkpoint = torch.load(os.path.join("models", "chexnet", "chexnet_jrzech.ckt"), map_location=torch.device('cpu'))
    return checkpoint["model"]


class CheXNetEncoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.chexnet = load_CheXNet()
        self.avgpool = nn.AvgPool2d((7, 7))

    def forward(self, xrays):
        xrays = xrays.unsqueeze(1).expand(-1, 3, -1, -1)
        xrays = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(xrays)

        x = self.chexnet.features(xrays)
        x = self.avgpool(x)

        return x[...,0,0]
    

class CheXNetDecoder1(nn.Module):
    def __init__(self, word_embeddings, hidden_size=512, freeze_embeddings=False):
        super().__init__()

        self.max_pool = nn.MaxPool1d(2, stride=2)

        self.vocab_size = word_embeddings.shape[0]
        self.embedding_size = word_embeddings.shape[1]

        self.embed = nn.Embedding.from_pretrained(torch.tensor(word_embeddings, dtype=torch.float32), freeze=freeze_embeddings) # fine-tune word embeddings as some of them where not in GloVe
        self.lstm = nn.LSTM(self.embedding_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, token_ids, context):
        context = self.max_pool(context)

        c0 = torch.zeros(1, token_ids.shape[0], self.lstm.hidden_size, device=token_ids.device)
        h0 = context[None]

        token_ids = self.embed(token_ids)
        token_ids, (hn, cn) = self.lstm(token_ids, (h0, c0))
        token_ids = self.dropout(token_ids)
        
        z = self.linear2(token_ids)
        z = F.relu(z)
        z = self.linear3(z)

        return z


class CheXNet1(nn.Module):
    def __init__(self, word_embeddings, hidden_size=512):
        super().__init__()

        self.encoder = CheXNetEncoder1() # the encoder is pretrained (and we won't fine-tune it), so use the context directly for speed-ups
        self.decoder = CheXNetDecoder1(word_embeddings, hidden_size)
        
    def forward(self, token_ids, context):
        output = self.decoder(token_ids, context)
        return output
    
    def preprocess(self, images):
        return process_to_fixed_context(self.encoder, images)
    
    def cached_emitter(self, context):
        def emitter(tokens):
            with torch.no_grad():
                self.eval()
                out = self.forward(tokens[None], context[None]) # add batch dim
                return F.log_softmax(out[0,-1,:], dim=-1)
        return emitter

#
# Model 3
#


class TransformerDecoder(nn.Module):
    def __init__(self, pretrained_embeddings, context_dim=1024, n_layers=3, n_heads=4, hidden_dim=400, freeze_embeddings=True):
        super().__init__()

        assert n_layers >= 1

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.context_dim = context_dim
        self.vocab_size = pretrained_embeddings.shape[0]
        self.embedding_dim = pretrained_embeddings.shape[1]

        self.positional_encoding = PositionalEncoding(self.embedding_dim, dropout=0.1, max_len=5000)
        self.linear_dim_expand = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings, dtype=torch.float32), freeze=freeze_embeddings)
        self.decoder_layer1 = MyTransformerDecoderLayer(qdim=self.hidden_dim, kdim=self.context_dim, vdim=self.context_dim, n_heads=n_heads, dropout=0.2, batch_first=True)

        self.context_dropout = nn.Dropout(0.4)

        if self.n_layers > 1:
            self.decoder_layerN_type = MyTransformerDecoderLayer(qdim=self.hidden_dim, kdim=self.context_dim, vdim=self.context_dim, n_heads=n_heads, batch_first=True)
            self.decoder_layerN = MyTransFormerDecoder(self.decoder_layerN_type, n_layers=self.n_layers-1)

        self.linear_vocab_dist = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, token_ids, context):
        mask = generate_square_subsequent_mask(token_ids.size(1)).to(token_ids.device)

        context = self.context_dropout(context)

        x = self.embedding(token_ids)
        x = self.positional_encoding(x)
        #x = self.linear_dim_expand(x)
        x = self.decoder_layer1(x, context, target_attn_mask=mask)

        if self.n_layers > 1:
            x = self.decoder_layerN(x, context, target_attn_mask=mask)

        return self.linear_vocab_dist(x)



class CheXNetEncoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.chexnet = load_CheXNet()

    def forward(self, xrays):
        xrays = xrays.unsqueeze(1).expand(-1, 3, -1, -1)
        xrays = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(xrays)
        x = self.chexnet.features(xrays)
        x = x.flatten(start_dim=-2) # (batch_size, 1024, 7, 7) -> (batch_size, 1024, 49)
        x = x.permute(0, 2, 1)      # (batch_size, 1024, 49)   -> (batch_size, 49, 1024)
        return x
    

class CheXTransformerNet(nn.Module):
    def __init__(self, pretrained_embeddings):
        super().__init__()

        self.encoder = CheXNetEncoder2()
        self.decoder = TransformerDecoder(pretrained_embeddings)
        
    def forward(self, text, context):
        output = self.decoder(text, context)
        return output
    
    def preprocess(self, images):
        return process_to_fixed_context(self.encoder, images)
    
    def cached_emitter(self, context):
        def emitter(tokens):
            with torch.no_grad():
                self.eval()
                out = self.forward(tokens[None], context[None]) # add batch dim
                return F.log_softmax(out[0,-1,:], dim=-1)
        return emitter
    

#
# Model 3
#

class XRayViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ViT_B_16_Weights.DEFAULT
        self.preprocess_fn = weights.transforms()

        self.encoder = vit_b_16(weights=ViT_B_16_Weights(weights))

    def forward(self, images):
        images = images.unsqueeze(1).expand(-1, 3, -1, -1)
        images = self.preprocess_fn(images)

        # Reshape and permute the input tensor
        images = self.encoder._process_input(images)
        n = images.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.encoder.class_token.expand(n, -1, -1)
        images = torch.cat([batch_class_token, images], dim=1)

        images = self.encoder.encoder(images)

        return images
    

class XRayViTDecoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, n_transformer_layers, pretrained_embeddings=None):
        super().__init__()

        assert n_transformer_layers >= 0
        
        self.n_transformer_layers = n_transformer_layers
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=0.1, max_len=5000)

        if pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings, dtype=torch.float32), freeze=False)
        else:
            self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        
        self.decoder_layer1 = pm.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        
        if n_transformer_layers > 1:
            self.decoder_layerN_type = pm.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.decoder_layerN = pm.TransformerDecoder(self.decoder_layerN_type, n_transformer_layers - 1)

        self.linear = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, input, context):
        x = self.embedding(input)
        x = self.positional_encoding(x)
        x = self.decoder_layer1(x, context, tgt_is_causal=True)
        #x = self.decoder_layer1(x, context, tgt_mask=generate_square_subsequent_mask(input.size(1)))

        if self.n_transformer_layers > 1:
            x = self.decoder_layerN(x, context)

        return self.linear(x)


class XRayViTModel(nn.Module):
    def __init__(self, vocabulary_size, hidden_size=768, n_transformer_layers=5):
        super().__init__()

        self.encoder = XRayViTEncoder()
        self.decoder = XRayViTDecoder(vocabulary_size, hidden_size, n_transformer_layers=n_transformer_layers)
        
    def forward(self, text, images):
        context = self.encoder(images)
        output = self.decoder(text, context)
        return output
    
    def preprocess(self, images):
        return images
    
    def cached_emitter(self, image):
    # persistent state to speed-up sampling
        with torch.no_grad():
            self.eval()
            context = self.encoder(image[None]).detach()

        # define emitter
        def emitter(tokens):
            with torch.no_grad():
                self.eval()
                out = self.decoder(tokens[None], context) # add batch dim
            return F.log_softmax(out[0,-1,:], dim=-1)
        
        return emitter
    

#
# retired
#

class TransformerDecoderOld(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim=960, context_size=1024, n_transformer_layers=1, freeze_embeddings=False):
        super().__init__()

        assert n_transformer_layers >= 1

        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.n_transformer_layers = n_transformer_layers
        self.vocab_size = pretrained_embeddings.shape[0]
        self.embedding_dim = pretrained_embeddings.shape[1]

        self.context_to_hidden = nn.Linear(context_size, hidden_dim)
        self.embedding_to_hidden = nn.Linear(self.embedding_dim, hidden_dim)

        self.positional_encoding = PositionalEncoding(self.embedding_dim, dropout=0.1, max_len=5000)
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings, dtype=torch.float32), freeze=freeze_embeddings)
        self.decoder_layer1 = pm.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        
        if self.n_transformer_layers > 1:
            self.decoder_layerN_type = pm.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
            self.decoder_layerN = pm.TransformerDecoder(self.decoder_layerN_type, n_transformer_layers - 1)

        self.linear_vocab_dist = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, token_ids, context):
        context = self.context_to_hidden(context)

        mask = generate_square_subsequent_mask(token_ids.size(1)).to(token_ids.device)

        x = self.embedding(token_ids)
        x = self.positional_encoding(x)
        x = self.embedding_to_hidden(x)

        #x = self.decoder_layer1(x, context, tgt_is_causal=True)
        x = self.decoder_layer1(x, context, tgt_mask=mask)

        if self.n_transformer_layers > 1:
            x = self.decoder_layerN(x, context, tgt_mask=mask)

        return self.linear_vocab_dist(x)



class CheXNetEncoder2Old(nn.Module):
    def __init__(self):
        super().__init__()
        self.chexnet = load_CheXNet()

    def forward(self, xrays):
        xrays = xrays.unsqueeze(1).expand(-1, 3, -1, -1)
        xrays = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(xrays)
        x = self.chexnet.features(xrays)
        x = x.flatten(start_dim=-2) # (batch_size, 1024, 7, 7) -> (batch_size, 1024, 49)
        x = x.permute(0, 2, 1)      # (batch_size, 1024, 49) -> (batch_size, 49, 1024)
        return x
    

class CheXTransformerNetOld(nn.Module):
    def __init__(self, pretrained_embeddings, hidden_dim=640, n_transformer_layers=8):
        super().__init__()

        self.encoder = CheXNetEncoder2Old()
        self.decoder = TransformerDecoderOld(pretrained_embeddings, hidden_dim=hidden_dim, n_transformer_layers=n_transformer_layers)
        
    def forward(self, text, context):
        output = self.decoder(text, context)
        return output
    
    def preprocess(self, images):
        return process_to_fixed_context(self.encoder, images)
    
    def cached_emitter(self, context):
        def emitter(tokens):
            with torch.no_grad():
                self.eval()
                out = self.forward(tokens[None], context[None]) # add batch dim
                return F.log_softmax(out[0,-1,:], dim=-1)
        return emitter
    
   
#
# shared
#

def masked_loss(y_pred, y_true, ignore_index, loss_weights=None):
    mask = (y_true != ignore_index).float()
    loss = F.cross_entropy(y_pred, y_true, reduction="none", weight=loss_weights)
    loss = loss * mask
    return loss.sum() / mask.sum()


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class MyTransFormerDecoder(nn.Module):
    def __init__(self, decoder_layer, n_layers):
        super().__init__()
        
        self.layers = _get_clones(decoder_layer, n_layers)

    def forward(self, input, context, target_attn_mask=None, context_attn_mask=None):
        for layer in self.layers:
            input = layer(input, context, target_attn_mask, context_attn_mask)
        return input


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, qdim, kdim, vdim, n_heads, dropout=0.1, layer_norm_eps=1e-5, dim_feedforward=2048, batch_first=True):
        super().__init__()

        # Get set in forward()
        self.masked_attn_weights = None
        self.multihead_attn_weights = None

        # Self-attention
        self.masked_self_attn = nn.MultiheadAttention(embed_dim=qdim, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(qdim, eps=layer_norm_eps)

        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=qdim, num_heads=n_heads, dropout=dropout, kdim=kdim, vdim=vdim, batch_first=batch_first)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(qdim, eps=layer_norm_eps)

        # FFN
        self.linear1 = nn.Linear(qdim, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, qdim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(qdim, eps=layer_norm_eps)

    def forward(self, input, memory, target_attn_mask=None, memory_attn_mask=None):
        # Self-attention
        x1, self.masked_attn_weights = self.masked_self_attn(input, input, input, attn_mask=target_attn_mask)
        x1 = self.dropout1(x1)
        x2 = self.norm1(x1 + input)

        # Multihead attention
        x3, self.multihead_attn_weights = self.multihead_attn(x2, memory, memory, attn_mask=memory_attn_mask)
        x3 = self.dropout2(x3)
        x4 = self.norm2(x3 + x2)

        # FFN
        x5 = self.activation(self.linear1(x4))
        x5 = self.dropout3(x5)

        x6 = self.linear2(x5)
        x6 = self.dropout4(x6)

        x7 = self.norm3(x6 + x4)

        return x7


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


def process_to_fixed_context(encoder, images, batch_size=16):
    with torch.no_grad():
        context_size = encoder(images[:1]).shape[1:]
        result = torch.zeros((images.shape[0], *context_size))

        for at in range(0, images.shape[0], batch_size):
            logging.info(f"processing images: {at}/{images.shape[0]}")
            end = min(at + batch_size, images.shape[0])
            processed = encoder(images[at:end])
            result[at:end] = processed.cpu()

        return result


#
# Embeddings
#

def load_pubmed_embeddings_IU_xray_pretrained():
    with open('embeddings/IU_xray_pretrained.pickle', 'rb') as handle:
        embeddings = pickle.load(handle)

    embeddings["<START>"] = embeddings.pop("startseq")
    embeddings["<END>"]   = embeddings.pop("endseq")
    embeddings["<UNK>"]   = embeddings.pop("UNK")

    return embeddings, 400


def load_bundled_glove_embeddings(embedding_name="glove-wiki-gigaword-300"):
    import gensim.downloader
    glove_vectors = gensim.downloader.load(embedding_name)
    return glove_vectors, len(next(iter(glove_vectors)))


def prepare_word_embeddings(token2id, vectors, embed_dim):
    word_embeddings = np.zeros((len(token2id), embed_dim))
    for token, id in token2id.items():
        if token in vectors:
            word_embeddings[id] = vectors[token]
    return word_embeddings


#
# Dataset
#

def get_dataloader(dataset, token2id, shuffle=True, batch_size=16):
    collate_fn = lambda input: report_collate_fn(token2id["<PAD>"], input)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


class XRayDataset(Dataset):
    def __init__(self, images, reports, token2id):
        self.images = images
        self.reports = reports
        self.token2id = token2id

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        report = self.reports.iloc[idx]
        report = ["<START>"] + report + ["<END>"]

        report_length = len(report)
        report_ids = [self.token2id[token] for token in report]

        return image, report_ids, report_length


def report_collate_fn(pad_id, input):
    images, reports, report_lengths = zip(*input)

    report_max_length = max(report_lengths)
    padded_reports = [report + [pad_id] * (report_max_length - length) for report, length in zip(reports, report_lengths)]

    t_images = torch.stack(list(images), dim=0)
    t_reports = torch.tensor(padded_reports)
    t_report_lengths = torch.tensor(report_lengths)

    return t_images, t_reports, t_report_lengths


#
# Training and evaluation
#

def make_model_dirs(model_name):
    ensure_dir(os.path.join("results", model_name))
    ensure_dir(os.path.join("models", model_name))


def train(model_name, model, vocabulary, train_dataset, validation_dataset, 
          epochs, batch_size, optimizer, loss_weights=None, disable_tqdm=True, checkpoint_save_freq=200, bleu_eval_freq=5, bleu_max_samples=-1, examples_to_show=3, device=None):
    
    make_model_dirs(model_name)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)

    if loss_weights is not None:
        loss_weights = loss_weights.to(device)

    results = {}

    token2id, id2token = map_token_and_id(vocabulary)

    train_dl = get_dataloader(train_dataset, token2id, batch_size=batch_size, shuffle=True)
    validation_dl = get_dataloader(validation_dataset, token2id, batch_size=batch_size, shuffle=False)

    results["train_losses"] = []
    results["validation_losses"] = []
    results["validation_bleu"] = []

    for t in range(epochs):
        train_loss = train_one_epoch(model, train_dl, token2id, optimizer, device, loss_weights, disable_tqdm=disable_tqdm)
        results["train_losses"].append(train_loss)

        validation_loss = evaluate(model, validation_dl, token2id, device, loss_weights, disable_tqdm=disable_tqdm)
        results["validation_losses"].append(validation_loss)

        logging.info(f"Epoch {t+1} train loss: {train_loss:.3f}, validation loss: {validation_loss:.3f}")

        if (t+1) % checkpoint_save_freq == 0:
            torch.save(model.state_dict(), os.path.join("models", model_name, f"model_{t+1}.pt"))
            logging.info(f"Saved model at epoch {t+1}")

        if (t+1) % bleu_eval_freq == 0:
            references, candidates = prepare_for_evaluation(model, validation_dataset, token2id, id2token, device=device, early_exit=bleu_max_samples)
            
            # get some feedback during training
            for i in range(min(examples_to_show, len(references))):
                example_truth = references[i][0]
                example_prediction = candidates[i]

                logging.info("True and predicted report:")
                logging.info(f"true: {tokens_to_text(example_truth)}")
                logging.info(f"pred: {tokens_to_text(example_prediction)}")
                logging.info("")


            bleu = bleu_score(references, candidates)

            results["validation_bleu"].append(bleu)
            logging.info(f"Epoch {t+1} BLEU: {bleu}")
    
    
    torch.save(model.state_dict(), os.path.join("models", model_name, "model.pt"))

    save_dict(results, os.path.join("results", model_name, "train_result.pkl"))


def train_one_epoch(model, train_dataloader, token2id, optimizer, device, loss_weights=None, disable_tqdm=True):
    train_losses = []

    criterion = nn.CrossEntropyLoss(ignore_index=token2id["<PAD>"], weight=loss_weights)

    model.train()
    for xrays, reports, _ in tqdm(train_dataloader, disable=disable_tqdm):
        xrays = xrays.to(device)
        reports = reports.to(device)

        y_pred = model(reports, xrays)

        # Since a LM predicts the next token, we need shift the tokens. Tokens "!   !" should be ignored.
        # y_pred: <hello>   <sailor>  <!>       <[END]>  !misc!
        # y_true: !Start!   <hello>   <sailor>  <!>      <[End]>
        y_pred_align = y_pred[:,:-1,:]
        y_true_align = reports[:,1:]

        loss = criterion(y_pred_align.flatten(end_dim=1), y_true_align.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().cpu().numpy())

    return np.mean(train_losses)


def evaluate(model, test_dataloader, token2id, device, loss_weights=None, disable_tqdm=True):
    criterion = nn.CrossEntropyLoss(ignore_index=token2id["<PAD>"], weight=loss_weights)

    with torch.no_grad():
        model.eval()

        test_losses = []

        for xrays, reports, _ in tqdm(test_dataloader, disable=disable_tqdm):
            xrays = xrays.to(device)
            reports = reports.to(device)

            y_pred = model(reports, xrays)

            y_pred_align = y_pred[:,:-1,:]
            y_true_align = reports[:,1:]

            loss = criterion(y_pred_align.flatten(end_dim=1), y_true_align.flatten())
            test_losses.append(loss.detach().cpu().numpy())

    return np.mean(test_losses)
