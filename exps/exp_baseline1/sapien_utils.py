# -*- coding: utf-8 -*-

import numpy as np
import trimesh

from sapien.core import Pose
import xml.etree.ElementTree as ET


def add_model(scene, idx, sapien_path, pose=Pose(), scale=1, fix_base=False, name='', stiffness=0, damping=0):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = False
    loader.scale = scale
    loader.fix_root_link = fix_base
    model = loader.load(f'{sapien_path}/{idx}/mobility.urdf')
    model.set_root_pose(pose)
    model.set_name(name)

    root = ET.parse(f'{sapien_path}/{idx}/mobility.urdf').getroot()
    for link in root.findall('link'):
        link_name = link.get('name')
        visual = link.find('visual')
        if visual is None:
            continue
        name = visual.get('name').split('-')[0]
        for model_link in model.get_links():
            if model_link.get_name() == link_name:
                model_link.set_name(name)

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

def farthest_point_sample(point, npoint, indices=False):
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
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    if indices:
        return point, centroids
    return point



def get_global_mesh_key(obj, keys=None):
    final_vs = []
    final_fs = []
    key_fs = []
    vid = 0
    for l in obj.get_links():
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
            if keys is not None and l.get_name() in keys:
                key_fs.append(np.ones(f.shape[0], dtype=bool))
            else:
                key_fs.append(np.zeros(f.shape[0], dtype=bool))
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
    key_fs = np.concatenate(key_fs, axis=0)
    mesh = trimesh.Trimesh(vertices=final_vs, faces=final_fs)
    return mesh, key_fs


def get_global_mesh_link(obj, keys=None, include=True):
    final_vs = []
    final_fs = []
    final_links = []
    vid = 0
    for i, l in enumerate(obj.get_links()):
        if keys is not None:
            if include and l.get_name() not in keys:
                continue
            elif not include and l.get_name() in keys:
                continue
        vs = []
        for s in l.get_collision_shapes():
            v = np.array(s.geometry.vertices, dtype=np.float32)
            f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
            vscale = s.geometry.scale
            v[:, 0] *= vscale[0]; v[:, 1] *= vscale[1]; v[:, 2] *= vscale[2];
            ones = np.ones((v.shape[0], 1), dtype=np.float32)
            v_ones = np.concatenate([v, ones], axis=1)
            # transmat = s.pose.to_transformation_matrix()
            # v = (v_ones @ transmat.T)[:, :3]
            vs.append(v)
            final_fs.append(f + vid)
            final_links.append(np.ones(f.shape[0], dtype=int) * i)
            vid += v.shape[0]
        if len(vs) > 0:
            vs = np.concatenate(vs, axis=0)
            ones = np.ones((vs.shape[0], 1), dtype=np.float32)
            vs_ones = np.concatenate([vs, ones], axis=1)
            # transmat = l.get_pose().to_transformation_matrix()
            # vs = (vs_ones @ transmat.T)[:, :3]
            final_vs.append(vs)
    if len(final_vs) == 0:
        return None, None
    final_vs = np.concatenate(final_vs, axis=0)
    final_fs = np.concatenate(final_fs, axis=0)
    final_links = np.concatenate(final_links, axis=0)
    return trimesh.Trimesh(vertices=final_vs, faces=final_fs), final_links


def get_global_mesh(obj, keys=None, include=True):
    final_vs = []
    final_fs = []
    vid = 0
    for l in obj.get_links():
        if keys is not None:
            if include and l.get_name() not in keys:
                continue
            elif not include and l.get_name() in keys:
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
    return trimesh.Trimesh(vertices=final_vs, faces=final_fs)


def get_pc_with_key(model, keys, npoints):
    if model is not None:
        mesh, key_fs = get_global_mesh_key(model, keys)
        pc, fs = trimesh.sample.sample_surface_even(mesh, 2 * npoints)
        key = key_fs[fs]
        pc, indices = farthest_point_sample(pc, npoints, indices=True)
        key = key[indices]
        pc = pc_normalize(pc)
        return pc, key
    else:
        pc = np.zeros((npoints, 3))
        return pc, np.zeros_like(pc)


def get_pc(model, npoints, keys=None, include=True, links=False):
    if model is not None:
        mesh, face_links = get_global_mesh_link(model, keys, include)
        if mesh is None:
            return None
        pc, fs = trimesh.sample.sample_surface_even(mesh, int(2 * npoints))
        pc_links = face_links[fs]
        if links:
            pc, indices = farthest_point_sample(pc, npoints, indices=True)
            pc_links = pc_links[indices]
            return pc, pc_links
        return farthest_point_sample(pc, npoints)
    else:
        return np.zeros((npoints, 3))


def visualize_key(pc, key):
    import pptk
    base_body_color = np.array([[1., 1., 1., 1.]])
    key_color = np.array([[1., 15/255., 0., 1.]])

    color = np.zeros((pc.shape[0], 4))
    color[key] = key_color
    color[~key] = base_body_color

    v = pptk.viewer(pc, color)
    v.set(point_size=0.005)
    v.set(phi=6.89285231)
    v.set(r=2.82503915)
    v.set(theta=0.91104609)
    v.set(lookat=[0.00296851, 0.01331535, -0.47486299])
    return v


if __name__ == '__main__':
    from sapien_const import Switch, get_model
    import sapien.core as sapien
    import pptk
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

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = 0.5

    keys = ['switch', 'handle', 'button']
    for i, idx in enumerate(Switch.sapien_id):
        model = get_model(loader, idx)
        pc, key = get_pc_with_key(model, keys, 2048)
        visualize_key(pc, key)
        import ipdb; ipdb.set_trace()
