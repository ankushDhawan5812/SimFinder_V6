import torch
import torch.nn as nn
import numpy as np
import arspb

class TransformerEncoder(torch.nn.Module):
    def __init__(self, state_embed_dimension):
        super().__init__()
        self.training = False
        self.embed_dimension = state_embed_dimension
        self.num_hidden_layers = 2
        self.num_attention_heads = 1
        self.hidden_size = 16
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dimension, nhead=self.num_attention_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_hidden_layers)
        self.neural_net = nn.Sequential(*[nn.Linear(self.embed_dimension, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, 1)])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        #x is a list of numpy arrays
        x_transformed = [torch.tensor(elt) for elt in x]
        x_transformed_more = torch.stack(x_transformed)
        x_transformed_even_more = torch.unsqueeze(x_transformed_more, 1)
        x_transformed_even_even_more = x_transformed_even_more.float()
        y = self.transformer_encoder(x_transformed_even_even_more)
        z = torch.mean(y, dim=0)
        z = self.neural_net(z)

        return self.sigmoid(z) * 5



