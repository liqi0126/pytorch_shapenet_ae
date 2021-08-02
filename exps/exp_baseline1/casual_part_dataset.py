import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sapien_const import NO_CASUAL, SELF_CASUAL, BINARY_CASUAL, OBJ_NUM


class CasualPartDataset(Dataset):
    def __init__(self, no_casual_num=3, self_casual_num=3, binary_casual_num=3):
        self.no_casual_num = no_casual_num
        self.self_casual_num = self_casual_num
        self.binary_casual_num = binary_casual_num
        self.reset()

    def reset(self):
        self.sapien_indices= []
        self.obj_groups = []

        # for i in range(np.random.randint(1, self.no_casual_num+1)):
        for i in range(self.no_casual_num):
            OBJ = random.choice(NO_CASUAL)
            self.sapien_indices.append(random.choice(OBJ.sapien_id))
            self.obj_groups.append(OBJ)

        # for i in range(np.random.randint(1, self.self_casual_num+1)):
        for i in range(self.self_casual_num):
            OBJ, _, _ = random.choice(SELF_CASUAL)
            self.sapien_indices.append(random.choice(OBJ.sapien_id))
            self.obj_groups.append(OBJ)

        # for i in range(np.random.randint(1, self.self_casual_num+1)):
        for i in range(self.binary_casual_num):
            SRC, _, DST, _ = random.choice(BINARY_CASUAL)
            self.sapien_indices.append(random.choice(SRC.sapien_id))
            self.sapien_indices.append(random.choice(DST.sapien_id))
            self.obj_groups.append(SRC)
            self.obj_groups.append(DST)

        self.obj_num = len(self.obj_groups)

    def __len__(self):
        return self.obj_num * self.obj_num

    def check_relation(self, obj_i, obj_j):
        for OBJ, _, _ in SELF_CASUAL:
            if obj_i == OBJ and obj_j == OBJ:
                return True

        for SRC, _, TGT, _ in BINARY_CASUAL:
            if obj_i == SRC and obj_j == TGT:
                return True
        return False

    def load_data(self, idx):
        df = pd.read_csv(f"part_data/{idx}.xyz")
        pc = df[['x', 'y', 'z']].to_numpy()
        src = df[['src']].to_numpy()
        tgt = df[['tgt']].to_numpy()
        pc = torch.from_numpy(pc).float()
        src = torch.from_numpy(src).float().squeeze()
        tgt = torch.from_numpy(tgt).float().squeeze()
        return pc, src, tgt

    def __getitem__(self, idx):
        i = idx // self.obj_num
        j = idx % self.obj_num

        obj_i = self.obj_groups[i]
        obj_j = self.obj_groups[j]
        idx_i = self.sapien_indices[i]
        idx_j = self.sapien_indices[j]

        pc_i, src_i, tgt_i = self.load_data(idx_i)
        pc_j, key_j, tgt_j = self.load_data(idx_j)
        if self.check_relation(obj_i, obj_j):
            key_i = src_i
            key_j = tgt_j
        else:
            key_i = torch.zeros(pc_i.shape[0])
            key_j = torch.zeros(pc_j.shape[0])

        onehot_i = torch.zeros(OBJ_NUM); onehot_i[obj_i.idx] = 1
        onehot_j = torch.zeros(OBJ_NUM); onehot_j[obj_j.idx] = 1
        return onehot_i, onehot_j, pc_i, pc_j, key_i, key_j

    def relation_graph(self):
        graph = torch.zeros(self.obj_num, self.obj_num, dtype=torch.bool)
        for i in range(self.obj_num):
            for j in range(self.obj_num):
                graph[i, j] = self.check_relation(self.obj_groups[i], self.obj_groups[j])
        return graph

    def get_scene(self):
        pcs, keys = [], []
        for idx in self.sapien_indices:
            pc, key = self.load_data(idx)
            pcs.append(pc)
            keys.append(key)
        return pcs, keys, self.relation_graph()


class CasualRelationDataset(CasualPartDataset):
    def __init__(self, no_casual_num=3, self_casual_num=3, binary_casual_num=3):
        self.no_casual_num = no_casual_num
        self.self_casual_num = self_casual_num
        self.binary_casual_num = binary_casual_num

    def __len__(self):
        return 1

    def check_relation(self, obj_i, obj_j):
        for OBJ, _, _ in SELF_CASUAL:
            if obj_i == OBJ and obj_j == OBJ:
                return True

        for SRC, _, TGT, _ in BINARY_CASUAL:
            if obj_i == SRC and obj_j == TGT:
                return True
        return False

    def __getitem__(self, _):
        self.reset()
        indices, pcs, keys = [], [], []
        for OBJ, idx in zip(self.obj_groups, self.sapien_indices):
            onehot = torch.zeros(OBJ_NUM); onehot[OBJ.idx] = 1
            pc, src, tgt = self.load_data(idx)
            indices.append(onehot)
            pcs.append(pc)
        return torch.stack(indices), torch.stack(pcs), self.relation_graph()


if __name__ == '__main__':
    dataset = CasualPartDataset()
    pc_i, pc_j, key_i, key_j = dataset.__getitem__(0)
