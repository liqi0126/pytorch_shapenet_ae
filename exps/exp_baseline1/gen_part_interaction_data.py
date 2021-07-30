import os
import pandas as pd
import sapien.core as sapien

from tqdm import tqdm

from sapien_const import SELF_CASUAL, BINARY_CASUAL
from interaction_env import gen_interaction_pc

from multiprocessing import Pool, Process

Engine = sapien.Engine(0, 0.001, 0.005)


def gen_data(engine, sapien_path, renderer, controller, idx, TRIGGER_NAME):
    pc, key = gen_interaction_pc(Engine, sapien_path, renderer, controller, idx, TRIGGER_NAME)
    df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
    df.to_csv(f"part_data/{idx}_src.xyz")


def main():
    sapien_path = '/public/MARS/datasets/partnet_mobility_v0'
    pool = Pool(8)

    renderer = None
    controller = None

    for OBJ, TRIGGER, TARGET in tqdm(SELF_CASUAL):
        TRIGGER_NAME = []
        for name, _, _ in TRIGGER:
            TRIGGER_NAME.append(name)

        for idx in OBJ.sapien_id:
            if not os.path.exists(f"part_data/{idx}_src.xyz"):
                res = pool.apply_async(gen_data, args=(None, sapien_path, renderer, controller, idx, TRIGGER_NAME))
                res.get()

    for SRC, _, DST, _ in tqdm(BINARY_CASUAL):
        TRIGGER_NAME = []
        for name, _, _ in SRC.trigger:
            TRIGGER_NAME.append(name)
        for idx in SRC.sapien_id:
            if not os.path.exists(f"part_data/{idx}_src.xyz"):
                res = pool.apply_async(gen_data, args=(None, sapien_path, renderer, controller, idx, TRIGGER_NAME))
                res.get()

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
