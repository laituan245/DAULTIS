import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, input_size, encoder_hidden_sizes,
                 latent_size, decoder_hidden_sizes):
        assert(len(encoder_hidden_sizes) > 0)
        assert(len(decoder_hidden_sizes) > 0)

        super(VAE, self).__init__()

        # Initialize Encoder Layers
        self.encoder_layers = []
        prev_size = input_size
        for hidden_size in encoder_hidden_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.encoder_layers.append(nn.ReLU(True))
            prev_size = hidden_size
        self.encoder_layer_module = nn.ModuleList(self.encoder_layers)

        # Initialize Decoder Layers
        self.decoder_layers = []
        prev_size = latent_size
        for hidden_size in decoder_hidden_sizes:
            self.decoder_layers.append(nn.Linear(prev_size, hidden_size))
            self.decoder_layers.append(nn.ReLU(True))
            prev_size = hidden_size
        self.decoder_layers.append(nn.Linear(prev_size, input_size))
        self.decoder_layers.append(nn.Sigmoid())
        self.decoder_layer_module = nn.ModuleList(self.decoder_layers)

        # Layers for calculating the mean and logvar of z
        self.hidden2mean = nn.Linear(encoder_hidden_sizes[-1], latent_size)
        self.hidden2logvar = nn.Linear(encoder_hidden_sizes[-1], latent_size)

    def encode(self, x):
        batch_size = x.size()[0]
        out = x
        for layer in self.encoder_layer_module:
            out = layer(out)
        out = out.view(batch_size, -1)
        return self.hidden2mean(out), self.hidden2logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        batch_size = z.size()[0]
        out = z
        for layer in self.decoder_layer_module:
            out = layer(out)
        return out.view(batch_size, -1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class EncoderFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super(EncoderFC, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU(True))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, latent_size))
        self.layers.append(nn.Tanh())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        batch_size = x.size()[0]
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(batch_size, -1)

class DecoderFC(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size):
        super(DecoderFC, self).__init__()
        self.layers = []

        prev_size = latent_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU(True))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        batch_size = x.size()[0]
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(batch_size, -1)

class DiscriminatorFC(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DiscriminatorFC, self).__init__()
        self.layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU(True))
            prev_size = hidden_size

        self.layers.append(nn.Linear(prev_size, output_size))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(-1, 1)
