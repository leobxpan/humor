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
from libmesh.inside_mesh import check_mesh_contains
from body_model.body_model import BodyModel

def check_if_valid(vertices, scene):
    # vertices: bs(1) X N X 3
    in_penetration = check_mesh_contains(scene, vertices)
    return in_penetration

def filter_floor_cols(human_verts, in_penetration, max_floor_height):
    human_verts_above_floor = human_verts[:, 2] > max_floor_height
    valid_in_penetration = np.logical_and(human_verts_above_floor, in_penetration)

    return valid_in_penetration

def is_person_in_scene(human_verts, scene_verts, debug_vis=False):
    scene_2d = scene_verts[:, :2]
    human_2d = human_verts[:, :2]

    scene_min = scene_2d.min(axis=0)
    scene_max = scene_2d.max(axis=0)

    if np.all(human_2d >= scene_min) and np.all(human_2d <= scene_max):
        return True
    else:
        return False

def process_penetration(in_penetration, vertices, joints, counts_thresh=100, debug=False, dataset_type="map_10", device="cpu"):
    if isinstance(in_penetration, np.ndarray):
        in_penetration = torch.from_numpy(in_penetration).to(device)
        vertices = torch.from_numpy(vertices)
        joints = torch.from_numpy(joints)
    verts_in_penetration = vertices[in_penetration, :]
    verts_dist_to_joints = torch.cdist(verts_in_penetration, joints)
    penetration_joints = torch.argmin(verts_dist_to_joints, dim=1)
    unique_penetration_joints, counts = torch.unique(penetration_joints, return_counts=True)

    unique_penetration_joints = map_joints(unique_penetration_joints, dataset_type)
    unique_joints = torch.unique(unique_penetration_joints, return_counts=False)
    new_counts = torch.zeros_like(unique_joints)
    for i in range(unique_joints.shape[0]):
        new_counts[i] = counts[(unique_penetration_joints == unique_joints[i]).nonzero(as_tuple=True)[0]].sum()
    #print("counts:", counts)
    joints_mask = new_counts >= counts_thresh
    unique_joints = torch.masked_select(unique_joints, joints_mask)

    if debug:
        if unique_joints.nelement() == 0:
            unique_joints = torch.tensor([100]).to(vertices.device)
            verts_mask = torch.tensor([]).to(vertices.device)
        else:
            verts_mask = torch.tensor([1 if penetration_joint in unique_joints else 0 for penetration_joint in map_joints(penetration_joints, dataset_type)], device=verts_in_penetration.device).bool()
        return unique_joints, verts_in_penetration[verts_mask]
    else:
        if unique_joints.nelement() == 0:
            unique_joints = torch.tensor([100]).to(vertices.device)
        return unique_joints

def map_joints(penetration_label, dataset_type="map_10"):
    """
    Cluster penetration label to only a few important joints.
    """

    if dataset_type == "nomap_22":
        return penetration_label
    elif dataset_type == "map_10":
        mapping_dict = {
            0 : 0,
            1 : 0,
            2 : 0,
            3 : 0,
            6 : 0,
            9 : 0,
            13: 0,
            14: 0,
            16: 0,
            17: 0,
            18: 1,
            19: 2,
            20: 3,
            21: 4,
            12: 5,
            15: 5,
            4 : 6,
            5 : 7,
            7 : 8,
            10: 8,
            8 : 9,
            11: 9,
            100: 100,
        }

        joint_ind_to_name_dict = {
            0 : "torso",
            1 : "leftForeArm",
            2 : "rightForeArm",
            3 : "leftHand",
            4 : "rightHand",
            5 : "head",
            6 : "leftLeg",
            7 : "rightLeg",
            8 : "leftFoot",
            9 : "rightFoot",
            100: "noCollision"
        }

        mapped_penetration_label = [mapping_dict[orig_label.item()] for orig_label in penetration_label]
        mapped_penetration_label = torch.tensor(mapped_penetration_label, device=penetration_label.device)
        #mapped_penetration_label = torch.unique(mapped_penetration_label, sorted=True)

        return mapped_penetration_label
    else:
        raise ValueError('Unsupported dataset type: {}'.format(dataset_type))

scene_root = "/orion/group/Mp3d_Gibson_scenes"
motion_root = "/orion/group/Mp3d_Gibson_motions"
SMPLH_PATH = "/orion/u/bxpan/exoskeleton/humor/body_models/smplh"
MALE_BM_PATH = os.path.join(SMPLH_PATH, "male/model.npz")
FEMALE_BM_PATH = os.path.join(SMPLH_PATH, "female/model.npz")

floor_buffer = 0.1

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

one_floor_scenes = []
all_scenes = os.listdir(motion_root)
all_scenes = ["Albertville"]
for scene in all_scenes:
    scene_dir = os.path.join(motion_root, scene)
    if not os.path.isdir(scene_dir): continue
    with open(os.path.join(scene_root, scene, "floors.txt"), "r") as f:
        floor_heights = sorted(list(map(float, f.readlines())))
    if len(floor_heights) == 1 and np.abs(floor_heights[0]) <= 0.02:
        one_floor_scenes.append(scene)
    
for scene in os.listdir(motion_root):
    if scene != "Albertville": continue
    scene_dir = os.path.join(motion_root, scene)
    if not os.path.isdir(scene_dir) and scene != "pickles": continue
    scene_mesh = trimesh.exchange.load.load(os.path.join(scene_root, scene, "watertight", "mesh_z_up.obj"))
    scene_simple = trimesh.exchange.load.load(os.path.join(scene_root, scene, "watertight", "simple_1e5.obj"))
    #scene_simple.export(os.path.join(scene_root, scene, "watertight", "simple_1e5.obj"))

    for motion in os.listdir(scene_dir):
        if motion != "BioMotionLab_NTroje_rub018_0029_jumping2_poses_64_frames_30_fps_b34243seq0": continue

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

        male_bm = BodyModel(bm_path=MALE_BM_PATH, num_betas=16, batch_size=orig_motion_seq['trans'].shape[0]).to(device)
        female_bm = BodyModel(bm_path=FEMALE_BM_PATH, num_betas=16, batch_size=orig_motion_seq['trans'].shape[0]).to(device)
        bm_world = male_bm if orig_motion_seq['gender'][0] == 'male' else female_bm
        verts, faces = gen_obj_from_motion_seq(orig_motion_seq["root_orient"], orig_motion_seq["pose_body"], orig_motion_seq["trans"], orig_motion_seq["betas"], bm_world, device)
        verts += floor_point
        person_out_of_scene_flag = False
        seq_end_idx = new_trans.shape[0]


        in_penetration = check_if_valid(verts[seq_end_idx], scene_simple)
        valid_in_penetration = filter_floor_cols(verts[seq_end_idx], in_penetration, floor_height + floor_buffer)
        penetration_label = process_penetration(valid_in_penetration, verts[seq_end_idx], \
            translated_joints[seq_end_idx], debug=False, counts_thresh=0, device=device)
        print("simple:", valid_in_penetration.sum(), penetration_label)

        orig_in_penetration = check_if_valid(verts[seq_end_idx], scene_mesh)
        orig_valid_in_penetration = filter_floor_cols(verts[seq_end_idx], orig_in_penetration, floor_height + floor_buffer)
        if orig_valid_in_penetration.sum() > 0:
            orig_penetration_label = process_penetration(orig_valid_in_penetration, verts[seq_end_idx], \
                translated_joints[seq_end_idx], debug=False, counts_thresh=0, device=device)
            print("original:", orig_valid_in_penetration.sum(), orig_penetration_label)
        else:
            print("original:", orig_valid_in_penetration.sum())
        import pdb; pdb.set_trace()
        for i in range(verts.shape[0]):
            #if scene not in one_floor_scenes: 

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








