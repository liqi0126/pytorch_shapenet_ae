import math
import numpy as np
import trimesh
import pandas as pd

import sapien.core as sapien
from sapien_utils import  add_model, pc_normalize, get_pc
# MODIFY THIS
SAPIEN_PATH = '/Users/liqi17thu/data/partnet_mobility_v0'


class InteractionEnv(object):
    def __init__(self, engine, sapien_path, sapien_idx, keys, renderer=None, controller=None, npoints=2048):
        self.engine = engine
        self.sapien_path = sapien_path
        self.sapien_idx = sapien_idx
        self.keys = keys
        self.npoints = npoints
        self.renderer = renderer
        self.controller = controller
        if renderer is not None:
            assert controller is not None
            self.renderer = renderer
            self.controller = controller
            self.engine.set_renderer(self.renderer)

        self.config = sapien.SceneConfig()
        self.config.gravity = [0, 0, 0]
        self.config.solver_iterations = 15
        self.config.solver_velocity_iterations = 2
        self.config.enable_pcm = False
        self.scene = self.engine.create_scene(config=self.config)
        self.scene.set_timestep(1 / 200)

        self.window_shown = False
        if self.renderer is not None:
            self.scene.set_ambient_light([1, 1, 1])
            self.scene.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])

            self.controller.set_camera_position(-3, 0, 16)
            self.controller.set_camera_rotation(0, -1.3)

        self.sphere_force = trimesh.sample.sample_surface_sphere(500)

        self.model = add_model(self.scene, sapien_idx, SAPIEN_PATH, scale=1, fix_base=True, stiffness=0, damping=0)
        self.model_qlimits = self.model.get_qlimits()[:, 0]
        self.model_qlimits[np.isneginf(self.model_qlimits)] = -10
        self.model_qlimits[np.isinf(self.model_qlimits)] = 10
        self.model.set_qpos(self.model_qlimits)

        pc_key, pc_links_key = get_pc(self.model, npoints//2, keys, include=True, links=True)
        pc_bg, pc_links_bg = get_pc(self.model, npoints//2, keys, include=False, links=True)
        self.pc = np.r_[pc_key, pc_bg]
        self.pc_links = np.r_[pc_links_key, pc_links_bg]

        self.key_links = []
        self.key_joints = []
        self.key_joint_indices = []
        for link in self.model.get_links():
            if link.get_name() in self.keys:
                self.key_links.append(link)

        joint_idx = 0
        for joint in self.model.get_joints():
            if joint.get_child_link().get_name() in self.keys:
                self.key_joints.append(joint)
                self.key_joint_indices.append(joint_idx)
            # if joint.type != sapien.ArticulationJointType.UNDEFINED and joint.type != sapien.ArticulationJointType.FIX:
            if joint.type != 'fixed' and joint.type != 'unknown':
                joint_idx += 1

        self.reset()


    def reset(self):
        self.scene = self.engine.create_scene(config=self.config)
        self.scene.set_timestep(1 / 200)

        self.model = add_model(self.scene, self.sapien_idx, SAPIEN_PATH, scale=1, fix_base=True, stiffness=0, damping=0)
        self.qpos = self.model.get_qpos()
        self.model.set_qpos(self.model_qlimits)
        while (self.qpos != self.model.get_qpos()).any():
            self.qpos = self.model.get_qpos()
            self.scene.step()
        return None


    def render(self, mode='human'):
        if self.controller:
            self.scene.update_render()
            self.controller.show_window()
            self.controller.set_current_scene(self.scene)
            self.controller.render()


    def step(self, action):
        idx = action[0]
        force = action[1]
        link = self.model.get_links()[self.pc_links[idx]]
        # if link.get_name() not in self.keys:
        #     return 0, 0, True, {}
        link.add_force_at_point(force=force, point=self.pc[idx])
        self.scene.step()
        obs = 0
        done = False
        if (self.qpos[self.key_joint_indices] != self.model.get_qpos()[self.key_joint_indices]).any():
            obs = 1
            done = True
        return obs, 0, done, {}

    def get_material(self, static_friction, dynamic_friction, restitution):
        return self.engine.create_physical_material(static_friction, dynamic_friction, restitution)

    def close(self):
        pass


def gen_interaction_pc(engine, sapien_path, renderer, controller, sapien_idx, keys):
    env = InteractionEnv(engine, sapien_path, sapien_idx, keys, renderer, controller)

    key = np.zeros(env.pc.shape[0], dtype=int)
    for i in range(env.pc.shape[0]):
        env.reset()
        for force in env.sphere_force:
            obs, reward, done, info = env.step((i, force))
            if obs == 1:
                key[i] = 1
            if done:
                break
    pc = pc_normalize(env.pc)
    return pc, key


def main():
    engine = sapien.Engine(0, 0.001, 0.005)
    renderer = None
    controller = None

    sapien_idx = 101579
    keys = ['switch']
    pc, key = gen_interaction_pc(engine, SAPIEN_PATH, renderer, controller, sapien_idx, keys)
    df = pd.DataFrame({"x": pc[:, 0], "y": pc[:, 1], "z": pc[:, 2], "key": key})
    df.to_csv(f"{sapien_idx}.xyz")


if __name__ == '__main__':
    main()

