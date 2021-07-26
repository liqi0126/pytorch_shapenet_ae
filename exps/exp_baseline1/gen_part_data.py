import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from sapien_const import NO_CASUAL, SELF_CASUAL, BINARY_CASUAL
from sapien_utils import add_model, get_pc, pc_normalize
import sapien.core as sapien


def main():
    sapien_path = '/Users/liqi17thu/data/partnet_mobility_v0'

    engine = sapien.Engine(0, 0.001, 0.005)

    config = sapien.SceneConfig()
    config.gravity = [0, 0, 0]
    config.solver_iterations = 15
    config.solver_velocity_iterations = 2
    config.enable_pcm = False

    scene = engine.create_scene(config=config)
    scene.set_timestep(1 / 200)

    OBJ_SET = set()

    for OBJ in tqdm(NO_CASUAL):
        for idx in OBJ.sapien_id:
            if os.path.exists(f"part_data/{idx}.xyz"):
                continue
            model = add_model(scene, idx, sapien_path)
            pc = get_pc(model, npoints=2048)
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": np.zeros(2048, dtype=int)})
            df.to_csv(f"part_data/{idx}.xyz")

    for OBJ, TRIGGER, TARGET in tqdm(SELF_CASUAL):
        TRIGGER_NAME = []
        for name, _, _ in TRIGGER:
            TRIGGER_NAME.append(name)
        for idx in OBJ.sapien_id:
            if os.path.exists(f"part_data/{idx}_dst.xyz"):
                continue

            model = add_model(scene, idx, sapien_path)

            pc = get_pc(model, npoints=2048)
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": np.zeros(2048, dtype=int)})
            df.to_csv(f"part_data/{idx}.xyz")

            pc_bg = get_pc(model, keys=TRIGGER_NAME, include=False, npoints=1024)
            pc_src = get_pc(model, keys=TRIGGER_NAME, include=True, npoints=1024)
            pc = np.r_[pc_bg, pc_src]
            pc = pc_normalize(pc)
            key = np.r_[np.zeros(1024, dtype=int), np.ones(1024, dtype=int)]
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
            df.to_csv(f"part_data/{idx}_src.xyz")

            if TARGET == 'all':
                pc = get_pc(model, npoints=2048)
                key = np.ones(2048, dtype=int)
            else:
                pc_bg = get_pc(model, keys=TARGET, include=False, npoints=1024)
                pc_dst = get_pc(model, keys=TARGET, include=True, npoints=1024)
                pc = np.r_[pc_bg, pc_dst]
                key = np.r_[np.zeros(1024, dtype=int), np.ones(1024, dtype=int)]
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
            df.to_csv(f"part_data/{idx}_dst.xyz")

        OBJ_SET.add(OBJ)

    for SRC, _, DST, _ in tqdm(BINARY_CASUAL):
        TRIGGER_NAME = []
        for name, _, _ in SRC.trigger:
            TRIGGER_NAME.append(name)
        for idx in SRC.sapien_id:
            if os.path.exists(f"part_data/{idx}_src.xyz"):
                continue

            model = add_model(scene, idx, sapien_path)

            pc = get_pc(model, npoints=2048)
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": np.zeros(2048, dtype=int)})
            df.to_csv(f"part_data/{idx}.xyz")

            pc_bg = get_pc(model, keys=TRIGGER_NAME, include=False, npoints=1024)
            pc_src = get_pc(model, keys=TRIGGER_NAME, include=True, npoints=1024)
            pc = np.r_[pc_bg, pc_src]
            pc = pc_normalize(pc)
            key = np.r_[np.zeros(1024, dtype=int), np.ones(1024, dtype=int)]
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
            df.to_csv(f"part_data/{idx}_src.xyz")

        for idx in DST.sapien_id:
            if os.path.exists(f"part_data/{idx}_dst.xyz"):
                continue
            model = add_model(scene, idx, sapien_path)

            pc = get_pc(model, npoints=2048)
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": np.zeros(2048, dtype=int)})
            df.to_csv(f"part_data/{idx}.xyz")

            if DST.responser == 'all':
                pc = get_pc(model, npoints=2048)
                key = np.ones(2048, dtype=int)
            else:
                pc_bg = get_pc(model, keys=DST.responser, include=False, npoints=1024)
                pc_dst = get_pc(model, keys=DST.responser, include=True, npoints=1024)
                pc = np.r_[pc_bg, pc_dst]
                key = np.r_[np.zeros(1024, dtype=int), np.ones(1024, dtype=int)]
            pc = pc_normalize(pc)
            df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
            df.to_csv(f"part_data/{idx}_dst.xyz")


if __name__ == '__main__':
    main()