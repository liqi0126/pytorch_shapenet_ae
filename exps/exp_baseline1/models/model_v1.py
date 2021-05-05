import torch
from torch import nn
import torch.nn.functional as F
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from emd.emd import earth_mover_distance


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn6 = nn.BatchNorm1d(128)

    """
        Input: B x N x 3
        Output: B x F
    """
    def forward(self, pcs):
        net = pcs.permute(0, 2, 1)

        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=-1)[0]

        net = torch.relu(self.bn5(self.fc1(net)))
        net = torch.relu(self.bn6(self.fc2(net)))
        
        return net


class Sampler(nn.Module):

    def __init__(self, feature_size, hidden_size, probabilistic=True):
        super(Sampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2mu = nn.Linear(hidden_size, feature_size)
        self.mlp2var = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        encode = torch.relu(self.mlp1(x))
        mu = self.mlp2mu(encode)

        if self.probabilistic:
            logvar = self.mlp2var(encode)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class SampleDecoder(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(SampleDecoder, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, input_feature):
        output = torch.relu(self.mlp1(input_feature))
        output = torch.relu(self.mlp2(output))
        return output


class FCDecoder(nn.Module):

    def __init__(self, num_point=2048):
        super(FCDecoder, self).__init__()
        print('Using FCDecoder-NoBN!')

        self.mlp1 = nn.Linear(128, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, num_point*3)

    def forward(self, feat):
        batch_size = feat.shape[0]

        net = feat
        net = torch.relu(self.mlp1(net))
        net = torch.relu(self.mlp2(net))
        net = self.mlp3(net).view(batch_size, -1, 3)

        return net


class FCUpconvDecoder(nn.Module):

    def __init__(self, num_point=2048):
        super(FCUpconvDecoder, self).__init__()
        print('Using FCUpconvDecoder-NoBN!')

        self.mlp1 = nn.Linear(128, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 1024*3)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, 1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, 3)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 1, 1)

    def forward(self, feat):
        batch_size = feat.shape[0]

        fc_net = feat
        fc_net = torch.relu(self.mlp1(fc_net))
        fc_net = torch.relu(self.mlp2(fc_net))
        fc_net = self.mlp3(fc_net).view(batch_size, -1, 3)

        upconv_net = feat.view(batch_size, -1, 1, 1)
        upconv_net = torch.relu(self.deconv1(upconv_net))
        upconv_net = torch.relu(self.deconv2(upconv_net))
        upconv_net = torch.relu(self.deconv3(upconv_net))
        upconv_net = torch.relu(self.deconv4(upconv_net))
        upconv_net = self.deconv5(upconv_net).view(batch_size, 3, -1).permute(0, 2, 1)
        
        net = torch.cat([fc_net, upconv_net], dim=1)

        return net


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.encoder = PointNet()
        
        self.sample_encoder = Sampler(128, 256, probabilistic=conf.probabilistic)
        self.sample_decoder = SampleDecoder(128, 256)

        if conf.decoder_type == 'fc':
            self.decoder = FCDecoder(num_point=conf.num_point)

        elif conf.decoder_type == 'fc_upconv':
            self.decoder = FCUpconvDecoder(num_point=conf.num_point)

        else:
            raise ValueError('ERROR: unknown decoder_type %s!' % decoder_type)

    """
        Input: B x N x 3
        Output: B x N x 3, B x F
    """
    def forward(self, input_pcs):
        feats = self.encoder(input_pcs)
        feats = self.sample_encoder(feats)
        ret_list = dict()
        if self.conf.probabilistic:
            feats, obj_kldiv_loss = torch.chunk(feats, 2, 1)
            ret_list['kldiv_loss'] = -obj_kldiv_loss.sum(dim=1)
        feats = self.sample_decoder(feats)
        output_pcs = self.decoder(feats)
        return output_pcs, feats, ret_list
    
    def get_loss(self, pc1, pc2):
        if self.conf.loss_type == 'cd':
            dist1, dist2 = chamfer_distance(pc1, pc2, transpose=False)
            loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

        elif self.conf.loss_type == 'emd':
            loss_per_data = earth_mover_distance(pc1, pc2, transpose=False) / min(pc1.shape[1], pc2.shape[1])

        else:
            raise ValueError('ERROR: unknown loss_type %s!' % loss_type)

        return loss_per_data
    
