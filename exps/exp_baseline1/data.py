import os
import torch
import torch.utils.data as data
import numpy as np
from geometry_utils import load_pts


class PartNetShapeDataset(data.Dataset):

    def __init__(self, data_dir, data_features, num_point=2048):
        self.data_dir = data_dir
        self.num_point = num_point
        self.data_features = data_features

        # load data
        with open(os.path.join(data_dir, 'data_list.txt'), 'r') as fin:
            self.data = [item.rstrip().split('.')[0] for item in fin.readlines()]

    def __str__(self):
        strout = '[PartNetPartDataset %d] data_dir: %s, num_point: %d' % \
                (len(self), self.data_dir, self.num_point)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_id = self.data[index]

        data_feats = ()
        for feat in self.data_features:
            if feat == 'pc':
                pc = load_pts(os.path.join(self.data_dir, shape_id+'.txt'))
                pc = pc[:self.num_point]
                pc = torch.from_numpy(pc).float().unsqueeze(0)
                data_feats = data_feats + (pc,)
           
            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats

