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
        mesh, key_fs = get_global_mesh(model, keys)
        pc, fs = trimesh.sample.sample_surface(mesh, npoints)
        # pc, fs = trimesh.sample.sample_surface_even(mesh, npoints)
        # pc = farthest_point_sample(pc, npoints)
        pc = pc_normalize(pc)
        return pc, key_fs[fs]
    else:
        pc = np.zeros((npoints, 3))
        return pc, np.zeros_like(pc)


def get_pc(model, keys=None, include=True, npoints=2048):
    if model is not None:
        mesh = get_global_mesh(model, keys, include)
        if mesh is None:
            return None
        pc, fs = trimesh.sample.sample_surface_even(mesh, int(1.5 * npoints))
        pc = farthest_point_sample(pc, npoints)
        # pc = pc_normalize(pc)
    else:
        pc = np.zeros((npoints, 3))
    return pc