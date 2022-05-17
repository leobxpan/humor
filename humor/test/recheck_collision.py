import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import torch

import numpy as np
import trimesh

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import shutil

from utils import gen_obj_from_motion_seq, write_to_obj
#from gen_seq_in_scene import check_if_valid, filter_floor_cols, is_person_in_scene
from body_model.body_model import BodyModel

def is_person_in_scene(human_verts, scene_verts, debug_vis=False):
    scene_2d = scene_verts[:, :2]
    human_2d = human_verts[:, :2]

    scene_min = scene_2d.min(axis=0)
    scene_max = scene_2d.max(axis=0)

    if np.all(human_2d >= scene_min) and np.all(human_2d <= scene_max):
        return True
    else:
        return False

scene_root = "/orion/group/Mp3d_Gibson_scenes"
motion_root = "/orion/group/Mp3d_Gibson_motions"
SMPLH_PATH = "/orion/u/bxpan/exoskeleton/humor/body_models/smplh"
MALE_BM_PATH = os.path.join(SMPLH_PATH, "male/model.npz")
FEMALE_BM_PATH = os.path.join(SMPLH_PATH, "female/model.npz")

floor_buffer = 0.1

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

one_floor_scenes = []
all_scenes = os.listdir(motion_root)
for scene in all_scenes:
    scene_dir = os.path.join(motion_root, scene)
    if not os.path.isdir(scene_dir): continue
    with open(os.path.join(scene_root, scene, "floors.txt"), "r") as f:
        floor_heights = sorted(list(map(float, f.readlines())))
    if len(floor_heights) == 1 and np.abs(floor_heights[0]) <= 0.02:
        one_floor_scenes.append(scene)
    
for scene in os.listdir(motion_root):
    scene_dir = os.path.join(motion_root, scene)
    if not os.path.isdir(scene_dir): continue
    scene_mesh = trimesh.exchange.load.load(os.path.join(scene_root, scene, "watertight", "mesh_z_up.obj"))
    scene_simple = scene_mesh.simplify_quadratic_decimation(1e5)
    scene_simple.export(os.path.join(scene_root, scene, "watertight", "simple_1e5.obj"))

    for motion in os.listdir(scene_dir):
        motion_dir = os.path.join(scene_dir, motion)

        if not os.path.isfile(os.path.join(motion_dir, "motion_seq.npz")): continue

        motion_seq = np.load(os.path.join(motion_dir, "motion_seq.npz"))
        new_trans = motion_seq['trans']

        orig_motion_name = motion + "_motion_seq.npz"
        orig_motion_seq = np.load(os.path.join("/orion/group/Exoskeleton_humor_motions", orig_motion_name))
        orig_trans = orig_motion_seq['trans']

        floor_point = new_trans[0] - orig_trans[0]
        translated_trans = orig_trans + floor_point
        translated_joints = orig_motion_seq['joints'] + floor_point
        floor_height = floor_point[2]

        male_bm = BodyModel(bm_path=MALE_BM_PATH, num_betas=16, batch_size=motion_seq['trans'].shape[0]).to(device)
        female_bm = BodyModel(bm_path=FEMALE_BM_PATH, num_betas=16, batch_size=motion_seq['trans'].shape[0]).to(device)
        bm_world = male_bm if motion_seq['gender'][0] == 'male' else female_bm
        verts, faces = gen_obj_from_motion_seq(motion_seq["root_orient"], motion_seq["pose_body"], motion_seq["trans"], motion_seq["betas"], bm_world, device)
        person_out_of_scene_flag = False
        seq_end_idx = orig_trans.shape[0]

        for i in range(verts.shape[0]):
            #if scene not in one_floor_scenes: 
            #    in_penetration = check_if_valid(verts[i], scene_simple)
            #    valid_in_penetration = filter_floor_cols(verts[i], in_penetration, floor_height + floor_buffer)

            #    if valid_in_penetration.sum() >= 10:
            #        seq_end_idx = i
            #        break
            #    else:
            #        if not is_person_in_scene(verts[i], scene_simple.vertices):
            #            person_out_of_scene_flag = True
            #            seq_end_idx = i
            #            break
            #else:
            if not is_person_in_scene(verts[i], scene_simple.vertices, debug_vis=True):
                person_out_of_scene_flag = True
                seq_end_idx = i
                break

        if seq_end_idx < 30:
            print("less than 30 frames at ", motion_dir)
            #shutil.rmtree(motion_dir)
            continue
        else:
            if seq_end_idx == motion_seq['trans'].shape[0]:
                continue
            else:
                if person_out_of_scene_flag and seq_end_idx < motion_seq['trans'].shape[0]:
                    print("out of scene at", motion_dir)
                #os.remove(os.path.join(motion_dir))
                #np.savez(os.path.join(motion_dir, "motion_seq.npz"), fps=30,
                #    gender=orig_motion_seq['gender'],
                #    trans=translated_trans[:seq_end_idx],
                #    root_orient=orig_motion_seq['root_orient'][:seq_end_idx],
                #    pose_body=orig_motion_seq['pose_body'][:seq_end_idx],
                #    betas=orig_motion_seq['betas'],
                #    joints=translated_joints[:seq_end_idx],
                #    joints_vel=orig_motion_seq['joints_vel'][:seq_end_idx],
                #    trans_vel=orig_motion_seq['trans_vel'][:seq_end_idx],
                #    root_orient_vel=orig_motion_seq['root_orient_vel'][:seq_end_idx])








