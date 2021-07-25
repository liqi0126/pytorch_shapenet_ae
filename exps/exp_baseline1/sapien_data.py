import torch.utils.data as data
from sapien_const import NO_CASUAL, SELF_CASUAL, BINARY_CASUAL
import numpy as np
import trimesh

from sapien.core import Pose


def add_model(scene, idx, sapien_path, pose=Pose(), scale=1, fix_base=False, name='', stiffness=0, damping=0):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.scale = scale
    loader.fix_root_link = fix_base
    model = loader.load(f'{sapien_path}/{idx}/mobility.urdf')
    model.set_root_pose(pose)
    model.set_name(name)

    for j in model.get_joints():
        if j.get_dof():
            j.set_drive_property(stiffness=stiffness, damping=damping)

    return model


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def get_global_mesh(obj, count=1024, keys=None):
    final_vs = []
    final_fs = []
    vid = 0
    for l in obj.get_links():
        if keys is not None and l.get_name() not in keys:
            continue
        vs = []
        for s in l.get_collision_shapes():
            v = np.array(s.geometry.vertices, dtype=np.float32)
            f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
            vscale = s.geometry.scale
            v[:, 0] *= vscale[0]
            v[:, 1] *= vscale[1]
            v[:, 2] *= vscale[2]
            ones = np.ones((v.shape[0], 1), dtype=np.float32)
            v_ones = np.concatenate([v, ones], axis=1)
            # transmat = s.get_local_pose().to_transformation_matrix()
            # v = (v_ones @ transmat.T)[:, :3]
            vs.append(v)
            final_fs.append(f + vid)
            vid += v.shape[0]
        if len(vs) > 0:
            vs = np.concatenate(vs, axis=0)
            ones = np.ones((vs.shape[0], 1), dtype=np.float32)
            vs_ones = np.concatenate([vs, ones], axis=1)
            # transmat = l.get_pose().to_transformation_matrix()
            # vs = (vs_ones @ transmat.T)[:, :3]
            final_vs.append(vs)
    if len(final_vs) == 0:
        return None
    final_vs = np.concatenate(final_vs, axis=0)
    final_fs = np.concatenate(final_fs, axis=0)
    mesh = trimesh.Trimesh(vertices=final_vs, faces=final_fs)
    samples, _ = trimesh.sample.sample_surface_even(mesh, count)
    # samples, _ = trimesh.sample.sample_surface(mesh, count)
    return samples


def get_pc(model, npoints):
    if model is not None:
        pc = get_global_mesh(model, int(npoints * 1.5))
        pc = farthest_point_sample(pc, npoints)
        pc = pc_normalize(pc)
    else:
        pc = np.zeros((npoints, 3))
    return pc


class PartNetShapeDataset(data.Dataset):
    def __init__(self, scene, sapien_path):
        self.scene = scene
        self.sapien_path = sapien_path
        self.OBJ_SET = set()
        self.sapien_indices = []

        for OBJ in NO_CASUAL:
            self.OBJ_SET.add(OBJ)

        for OBJ, _, _ in SELF_CASUAL:
            self.OBJ_SET.add(OBJ)

        for SRC, _, DST, _ in BINARY_CASUAL:
            self.OBJ_SET.add(SRC)
            self.OBJ_SET.add(DST)

        for OBJ in self.OBJ_SET:
            self.sapien_indices += OBJ.sapien_id

    def __len__(self):
        return len(self.sapien_indices)

    def __getitem__(self, index):
        sapien_id = self.sapien_indices[index]
        model = add_model(self.scene, sapien_id, self.sapien_path)
        pc = get_global_mesh(model, 2048)
        return pc


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

    SAPIEN_PATH = '/Users/liqi17thu/data/partnet_mobility_v0'
    dataset = PartNetShapeDataset(scene, SAPIEN_PATH)
    pc = dataset.__getitem__(0)
    print(pc.shape)
