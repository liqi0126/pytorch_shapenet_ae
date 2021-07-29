import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sapien_const import NO_CASUAL, SELF_CASUAL, BINARY_CASUAL


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

    def __getitem__(self, idx):
        i = idx // self.obj_num
        j = idx % self.obj_num

        obj_i = self.obj_groups[i]
        obj_j = self.obj_groups[j]
        idx_i = self.sapien_indices[i]
        idx_j = self.sapien_indices[j]

        if self.check_relation(obj_i, obj_j):
            df_i = pd.read_csv(f"part_data/{idx_i}_src.xyz")
            df_j = pd.read_csv(f"part_data/{idx_j}_dst.xyz")
        else:
            df_i = pd.read_csv(f"part_data/{idx_i}.xyz")
            df_j = pd.read_csv(f"part_data/{idx_j}.xyz")
        pc_i = df_i[['x', 'y', 'z']].to_numpy()
        pc_j = df_j[['x', 'y', 'z']].to_numpy()
        key_i = df_i[['key']].to_numpy()
        key_j = df_j[['key']].to_numpy()
        pc_i = torch.from_numpy(pc_i).float().unsqueeze(0)
        pc_j = torch.from_numpy(pc_j).float().unsqueeze(0)
        key_i = torch.from_numpy(key_i).float().squeeze().unsqueeze(0)
        key_j = torch.from_numpy(key_j).float().squeeze().unsqueeze(0)

        return pc_i, pc_j, key_i, key_j


if __name__ == '__main__':
    dataset = CasualPartDataset()
    for i in range(dataset.__len__()):
        pc_i, pc_j, key_i, key_j = dataset.__getitem__(i)
        if (key_i != 0).any():
            import ipdb; ipdb.set_trace()
