import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class CVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):
        super(CVAE, self).__init__()
        HIDDEN = 64
        self.features = features
        # encoder
        # self.gru = nn.GRU(input_size=input_size, hidden_size=HIDDEN, batch_first=True)  # not used for now

        self.encoder = nn.Sequential(
            nn.Embedding(
                num_embeddings=input_size, embedding_dim=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=2 * features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=features + extra_decoder_input, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
            nn.ReLU(),
            nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def encode(self, x):
        #         x, _ = self.gru(x)
        x = self.encoder(x)
        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return mu

    def forward(self, x, xp):
        # encoding
        #         x, _ = self.gru(x)
        x = self.encoder(x)

        mu = x[:, :self.features]
        log_var = x[:, self.features:]

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        dec_input = torch.cat([z, xp], -1)
        # decoding
        reconstruction = self.decoder(dec_input)
        return reconstruction, mu, log_var