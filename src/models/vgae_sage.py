import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GraphSAGEVAE, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv_mu = SAGEConv(hidden_channels, latent_channels)
        self.conv_logvar = SAGEConv(hidden_channels, latent_channels)

        self.feature_decoder = nn.Linear(latent_channels, in_channels)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_features(self, z):
        return self.feature_decoder(z)

    def decode_edges(self, z):
        return torch.matmul(z, z.t())

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)

        x_recon = self.decode_features(z)
        adj_recon = self.decode_edges(z)

        return x_recon, adj_recon, mu, logvar