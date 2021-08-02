import torch
from torch import nn
import torch.nn.functional as F
import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
from cd.chamfer import chamfer_distance
from emd.emd import earth_mover_distance

from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from .pointnet import PointNet

class PointNet2(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 256],
                use_xyz=True,
            )
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)

        return self.fc_layer2(bottleneck_feats)


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

    def __init__(self, num_point=2048, dim=3):
        super(FCDecoder, self).__init__()
        print('Using FCDecoder-NoBN!')
        self.dim = dim
        self.mlp1 = nn.Linear(128, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, num_point*dim)

    def forward(self, feat):
        batch_size = feat.shape[0]

        net = feat
        net = torch.relu(self.mlp1(net))
        net = torch.relu(self.mlp2(net))
        net = self.mlp3(net).view(batch_size, -1, self.dim)

        return net


class FCUpconvDecoder(nn.Module):

    def __init__(self, num_point=2048, dim=3):
        super(FCUpconvDecoder, self).__init__()
        print('Using FCUpconvDecoder-NoBN!')
        self.dim = dim

        self.mlp1 = nn.Linear(128, 1024)
        self.mlp2 = nn.Linear(1024, 1024)
        self.mlp3 = nn.Linear(1024, 1024*dim)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, 1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 5, 3)
        self.deconv5 = nn.ConvTranspose2d(128, dim, 1, 1)

    def forward(self, feat):
        batch_size = feat.shape[0]

        fc_net = feat
        fc_net = torch.relu(self.mlp1(fc_net))
        fc_net = torch.relu(self.mlp2(fc_net))
        fc_net = self.mlp3(fc_net).view(batch_size, -1, self.dim)

        upconv_net = feat.view(batch_size, -1, 1, 1)
        upconv_net = torch.relu(self.deconv1(upconv_net))
        upconv_net = torch.relu(self.deconv2(upconv_net))
        upconv_net = torch.relu(self.deconv3(upconv_net))
        upconv_net = torch.relu(self.deconv4(upconv_net))
        upconv_net = self.deconv5(upconv_net).view(batch_size, self.dim, -1).permute(0, 2, 1)
        net = torch.cat([fc_net, upconv_net], dim=1)

        return net


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.conf = conf

        self.encoder = PointNet2({'feat_dim': 128})

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
        feats = self.encoder(input_pcs.repeat(1, 1, 2))
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

from .loss import mIoULoss
from sapien_const import OBJ_NUM

class CasualNetwork(nn.Module):
    def __init__(self, conf):
        super(CasualNetwork, self).__init__()
        self.conf = conf
        self.feat_dim = 128
        self.pn_encoder = PointNet(k=128, normal_channel=False)
        self.encoder = PointNet2({'feat_dim': self.feat_dim})
        input_dim = 2 * OBJ_NUM + 128
        # input_dim = 2 * OBJ_NUM

        self.relation_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            nn.Linear(4, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.full_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            nn.Linear(4, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.src_sample_encoder = Sampler(128, 256, probabilistic=conf.probabilistic)
        self.src_sample_decoder = SampleDecoder(128, 256)

        self.dst_sample_encoder = Sampler(128, 256, probabilistic=conf.probabilistic)
        self.dst_sample_decoder = SampleDecoder(128, 256)

        if conf.decoder_type == 'fc':
            self.src_decoder = FCDecoder(num_point=conf.num_point, dim=1)
            self.dst_decoder = FCDecoder(num_point=conf.num_point, dim=1)
        elif conf.decoder_type == 'fc_upconv':
            self.src_decoder = FCUpconvDecoder(num_point=conf.num_point, dim=1)
            self.dst_decoder = FCUpconvDecoder(num_point=conf.num_point, dim=1)
        else:
            raise ValueError('ERROR: unknown decoder_type %s!' % conf.decoder_type)

        self.relation_loss_fn = nn.BCELoss(reduce=False)
        self.full_loss_fn = nn.BCELoss(reduce=False)
        self.iou_loss_fn = mIoULoss(reduce=False)

    """
        Input: B x N x 3
        Output: B x N x 3, B x F
    """

    def forward(self, src_idx, dst_idx, src_pcs, dst_pcs):
        src_feats = self.encoder(src_pcs.repeat(1, 1, 2))
        dst_feats = self.encoder(dst_pcs.repeat(1, 1, 2))
        feats = src_feats + dst_feats
        src_feats = self.src_sample_encoder(feats)
        src_feats = self.src_sample_decoder(src_feats)
        src_pred = torch.sigmoid(self.src_decoder(src_feats))
        dst_feats = self.dst_sample_encoder(feats)
        dst_feats = self.dst_sample_decoder(dst_feats)
        dst_pred = torch.sigmoid(self.dst_decoder(dst_feats))

        src_pn_feats = self.pn_encoder(src_pcs.permute(0, 2, 1))
        dst_pn_feats = self.pn_encoder(dst_pcs.permute(0, 2, 1))
        pn_feats = src_pn_feats + dst_pn_feats
        feats = torch.cat([src_idx, dst_idx, pn_feats], -1)
        # feats = torch.cat([src_idx, dst_idx], -1)
        relation = self.relation_net(feats).squeeze()
        full = self.full_net(feats).squeeze()
        return relation, full, src_pred, dst_pred

    def get_loss(self, relation, full, src_pred, src_gt, tgt_pred, tgt_gt):
        gt_relation = ((src_gt.sum(-1) != 0) & (tgt_gt.sum(-1) != 0))
        gt_full = (tgt_gt.sum(-1) == tgt_gt.shape[1])
        relation_loss = self.relation_loss_fn(relation, gt_relation.float())
        accuracy = ((relation >= 0.5) == gt_relation).float().mean()

        full_loss = self.full_loss_fn(full, gt_full.float())
        full_loss[~gt_relation] = 0

        tgt_oh_pred = tgt_pred[gt_relation & ~gt_full] > 0.5
        tgt_oh_gt = tgt_gt[gt_relation & ~gt_full] > 0.5
        tgt_iou = ((tgt_oh_pred > 0.5) & (tgt_oh_gt > 0.5)).float().sum() / (((tgt_oh_pred > 0.5) | (tgt_oh_gt > 0.5)).float().sum() + 1e-8)
        src_oh_pred = src_pred[gt_relation] > 0.5
        src_oh_gt = src_gt[gt_relation] > 0.5
        src_iou = ((src_oh_pred > 0.5) & (src_oh_gt > 0.5)).float().sum() / (((src_oh_pred > 0.5) | (src_oh_gt > 0.5)).float().sum() + 1e-8)
        tgt_iou_loss = self.iou_loss_fn(tgt_pred, tgt_gt)
        tgt_iou_loss[~gt_relation | gt_full] = 0
        src_iou_loss = self.iou_loss_fn(src_pred, src_gt)
        src_iou_loss[~gt_relation] = 0
        total_loss = relation_loss + full_loss + tgt_iou_loss + src_iou_loss
        return total_loss.mean(), accuracy, src_iou, tgt_iou

    def build_prior(self, indices, pcs):
        with torch.no_grad():
            pc_feats = self.encoder(pcs.repeat(1, 1, 2))
            pn_feats = self.pn_encoder(pcs.permute(0, 2, 1))
            prior_graph = torch.zeros(pcs.shape[0], pcs.shape[0]).to(pcs.device)
            for i in range(pcs.shape[0]):
                for j in range(pcs.shape[0]):
                    pn_feat = pn_feats[i] + pn_feats[j]
                    feats = torch.cat([indices[i], indices[j], pn_feat], -1)
                    relation = self.relation_net(feats.unsqueeze(0)).squeeze()
                    prior_graph[i, j] = relation
        return pc_feats, prior_graph
