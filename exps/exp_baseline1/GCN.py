import torch
from torch_geometric.nn import DenseGCNConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = DenseGCNConv(in_channels, 2 * out_channels)
        self.conv2 = DenseGCNConv(2 * out_channels, out_channels)

    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        return self.conv2(x, adj)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = DenseGCNConv(in_channels, 2 * out_channels)
        self.conv_mu = DenseGCNConv(2 * out_channels, out_channels)
        self.conv_logstd = DenseGCNConv(2 * out_channels, out_channels)

    def forward(self, x, adj):
        x = self.conv1(x, adj).relu()
        return self.conv_mu(x, adj), self.conv_logstd(x, adj)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = DenseGCNConv(in_channels, out_channels)

    def forward(self, x, adj):
        return self.conv(x, adj)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = DenseGCNConv(in_channels, out_channels)
        self.conv_logstd = DenseGCNConv(in_channels, out_channels)

    def forward(self, x, adj):
        return self.conv_mu(x, adj), self.conv_logstd(x, adj)
