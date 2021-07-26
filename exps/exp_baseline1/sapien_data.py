import torch.utils.data as data
from sapien_const import NO_CASUAL, SELF_CASUAL, BINARY_CASUAL
import numpy as np
from pyntcloud import PyntCloud

import torch

class PartNetSapienDataset(data.Dataset):
    def __init__(self, train):
        self.sapien_indices = []
        self.obj_indices = []
        self.OBJ_SET = set()
        self.train = train

        for OBJ in NO_CASUAL:
            self.OBJ_SET.add(OBJ)

        for OBJ, _, _ in SELF_CASUAL:
            self.OBJ_SET.add(OBJ)

        for SRC, _, DST, _ in BINARY_CASUAL:
            self.OBJ_SET.add(SRC)
            self.OBJ_SET.add(DST)

        for OBJ in self.OBJ_SET:
            slice = int(len(OBJ.sapien_id) * 0.8)
            if train:
                self.sapien_indices += OBJ.sapien_id[:slice]
            else:
                self.sapien_indices += OBJ.sapien_id[slice:]

            self.obj_indices += [OBJ.idx] * len(OBJ.sapien_id)

    def __len__(self):
        return len(self.sapien_indices)

    def __getitem__(self, index):
        data_feats = ()

        sapien_id = self.sapien_indices[index]
        pcd = PyntCloud.from_file(f"data/{sapien_id}.xyz")
        pc = pcd.xyz
        pc = torch.from_numpy(pc).float().unsqueeze(0)
        obj_idx = self.obj_indices[index]
        data_feats += (pc,)
        data_feats += (obj_idx, )
        return data_feats


if __name__ == '__main__':
    import sapien.core as sapien

    engine = sapien.Engine(0, 0.001, 0.005)
    renderer = None
    controller = None

    config = sapien.SceneConfig()
    config.gravity = [0, 0, 0]
    config.solver_iterations = 15
    config.solver_velocity_iterations = 2
    config.enable_pcm = False

    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 200)

    SAPIEN_PATH = '/public/MARS/datasets/partnet_mobility_v0' 
    dataset = PartNetSapienDataset(scene, SAPIEN_PATH)
    pc = dataset.__getitem__(0)
    print(pc.shape)
