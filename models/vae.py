from typing import *
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
import contextlib

import torch
from torch import nn, optim
from torch.nn import functional as F

import torchvision
from IPython import display
from PIL import Image
import torchvision.transforms as transforms

from tqdm.auto import tqdm


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        r"""
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        """
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)  # 64x64 --> 32x32
        self.conv2 = nn.Conv2d(32, 32 * 2, 4, stride=2, padding=1)  # 32x32 --> 16x16
        self.conv3 = nn.Conv2d(32 * 2, 32 * 4, 4, stride=2, padding=1)  # 16x16 --> 8x8
        self.conv4 = nn.Conv2d(32 * 4, 32 * 8, 4, stride=2, padding=1)  # 8x8 --> 4x4
        self.conv5 = nn.Conv2d(32 * 8, 32 * 16, 4, stride=2, padding=1)  # 4x4 --> 2x2
        self.conv6 = nn.Conv2d(
            32 * 16, hidden_dim, 4, stride=2, padding=1
        )  # 2x2 --> 1x1
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        x = self.nonlinearity(self.conv6(x).flatten(1))
        x = self.fc(x)
        return x

    def extra_repr(self):
        return f"normalize={self.normalize}"


class Decoder(nn.Module):
    def __init__(self, z_dim):
        r"""
        latent_dim (int): Dimension of latent space
        """
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(
            z_dim, 32 * 16, 4, stride=2, padding=1
        )  # 1x1 --> 2x2
        self.conv2 = nn.ConvTranspose2d(
            32 * 16, 32 * 8, 4, stride=2, padding=1
        )  # 2x2 --> 4x4
        self.conv3 = nn.ConvTranspose2d(
            32 * 8, 32 * 4, 4, stride=2, padding=1
        )  # 4x4 --> 8x8
        self.conv4 = nn.ConvTranspose2d(
            32 * 4, 32 * 2, 4, stride=2, padding=1
        )  # 8x8 --> 16x16
        self.conv5 = nn.ConvTranspose2d(
            32 * 2, 32, 4, stride=2, padding=1
        )  # 16x16 --> 32x32
        self.conv6 = nn.ConvTranspose2d(
            32, 3, 4, stride=2, padding=1
        )  # 32x32 --> 64x64
        self.nonlinearity = nn.ReLU()

    def forward(self, z):
        z = z[..., None, None]  # make it convolution-friendly

        x = self.nonlinearity(self.conv1(z))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        return self.conv6(x)


class LatentLayer(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        r"""
        latent_dim (int): Dimension of latent space
        """
        super(LatentLayer, self).__init__()

        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        return self.mu(x), self.logvar(x)


class VAE(nn.Module):
    def __init__(self, z_dims, hidden_dim, normalize=False):
        super().__init__()
        self.z_dims = z_dims

        # FIXME: Create two encoder layers
        self.encoder = Encoder(hidden_dim)

        self.latent_layer = LatentLayer(hidden_dim, z_dims)

        # FIXME: Create the decoder layers
        self.decoder = Decoder(z_dims)

        self.normalize = normalize

    def forward(self, x):
        # FIXME: Implement the VAE forward function
        encoded_x = self.encoder(x)

        mu, logvar = self.latent_layer(encoded_x)
        # logvar = log(sigma**2) = 2*log(sigma) => sigma = exp(logvar/2)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logvar * 0.5) * eps
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        output = self.decoder(z)
        return output, z, mu, logvar
