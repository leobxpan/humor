import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import glob

import pickle

import torch

import numpy as np
import trimesh

import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm

from libmesh.inside_mesh import check_mesh_contains

from utils import gen_obj_from_motion_seq, write_to_obj
from body_model.body_model import BodyModel

SMPLH_PATH = "/orion/u/bxpan/exoskeleton/humor/body_models/smplh"
MALE_BM_PATH = os.path.join(SMPLH_PATH, "male/model.npz")
FEMALE_BM_PATH = os.path.join(SMPLH_PATH, "female/model.npz")

def check_if_valid(vertices, scene):
    # vertices: bs(1) X N X 3
    in_penetration = check_mesh_contains(scene, vertices)
    return in_penetration

def filter_floor_cols(human_verts, in_penetration, max_floor_height):
    human_verts_above_floor = human_verts[:, 2] > max_floor_height
    valid_in_penetration = np.logical_and(human_verts_above_floor, in_penetration)

    return valid_in_penetration

def map_to_world(xy, trav_map_size=2400, trav_map_resolution=0.01):
    """
    Transforms a 2D point in map reference frame into world (simulator) reference frame
    :param xy: 2D location in map reference frame (image)
    :return: 2D location in world reference frame (metric)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((xy - trav_map_size / 2.0) * trav_map_resolution, axis=axis)

def world_to_map(xy, trav_map_size=2400, trav_map_resolution=0.01):
    """
    Transforms a 2D point in world (simulator) reference frame into map reference frame
    :param xy: 2D location in world reference frame (metric)
    :return: 2D location in map reference frame (image)
    """
    axis = 0 if len(xy.shape) == 1 else 1
    return np.flip((np.array(xy) / trav_map_resolution + trav_map_size / 2.0)).astype(int)

def is_person_in_scene(human_verts, scene_verts):
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
motion_root = "/orion/group/Exoskeleton_humor_motions"
all_motions = sorted(os.listdir(motion_root))

output_dir = "/orion/group/Mp3d_Gibson_motions"
obj_output_dir = "test_obj"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# max and min areas to filter scenes
max_area = 1.5e6
min_area = 0

# number of motions sampled for largest and smallest scenes (to linspace from)
max_num_motions = 200
min_num_motions = 30

# parallelization
num_workers = 20
worker_id = 9 

floor_buffer = 0.1                  # 10cm buffer to avoid checking collisions with floor

num_col_verts_thresh = 10           # when more than 10 vertices colliding, say it's a collision sequence

min_seq_len = 30                    # minimum sequence length for it to be considered

addt_col_hor = 10                   # check for 10 time steps after the actual collision for additional collision labels

# debug visualizations
vis_traj = False
save_obj = False

with open(os.path.join(scene_root, "descending_scene_and_areas.pkl"), "rb") as f:
    descending_scenes_and_areas = pickle.load(f)

chosen_scenes = [scene for scene in descending_scenes_and_areas.keys() if descending_scenes_and_areas[scene] > min_area and descending_scenes_and_areas[scene] < max_area]
chosen_scene_areas = [descending_scenes_and_areas[scene] for scene in chosen_scenes]
all_num_motions = [min_num_motions + (area - chosen_scene_areas[-1]) * (max_num_motions - min_num_motions) / (chosen_scene_areas[0] - chosen_scene_areas[-1]) for area in chosen_scene_areas]

existing_scenes = os.listdir(output_dir)
one_floor_scenes = []
for scene in existing_scenes:
    scene_dir = os.path.join(output_dir, scene)
    if not os.path.isdir(scene_dir): continue
    with open(os.path.join(scene_root, scene, "floors.txt"), "r") as f:
        floor_heights = sorted(list(map(float, f.readlines())))
    if len(floor_heights) == 1 and np.abs(floor_heights[0]) <= 0.02:
        one_floor_scenes.append(scene)

remaining_scenes = [scene for scene in chosen_scenes if scene not in one_floor_scenes]
remaining_inds = [chosen_scenes.index(scene) for scene in remaining_scenes]
chosen_scenes = [chosen_scenes[idx] for idx in remaining_inds]
chosen_scene_areas = [chosen_scene_areas[idx] for idx in remaining_inds]
all_num_motions = [all_num_motions[idx] for idx in remaining_inds]

chosen_scenes = np.array(chosen_scenes)
chosen_scene_areas = np.array(chosen_scene_areas)
all_num_motions = np.array(all_num_motions).astype(int)

cur_scene_inds = np.arange(worker_id, len(chosen_scenes), num_workers)          # assign roughly equal amount of work to each worker
jobs = chosen_scenes[cur_scene_inds]
num_motion_jobs = all_num_motions[cur_scene_inds]

for i in range(len(jobs)):
    scene = jobs[i]
    scene_mesh = trimesh.exchange.load.load(os.path.join(scene_root, scene, "watertight", "mesh_z_up.obj"))
    scene_mesh = scene_mesh.simplify_quadratic_decimation(1e5)                  # dowmsample scene mesh to speed up collision check
    scene_mesh.export(os.path.join(scene_root, scene, "watertight", "simple_1e5.obj"))
    scene_output_dir = os.path.join(output_dir, scene)

    if not os.path.isdir(scene_output_dir):
        os.makedirs(scene_output_dir)

    trav_map_paths = sorted(glob.glob(os.path.join(scene_root, scene, "floor_trav_*_v*.png")))
    if not trav_map_paths:
        trav_map_paths = sorted(glob.glob(os.path.join(scene_root, scene, "floor_trav_*.png")))
    trav_maps = [np.array(PIL.Image.open(trav_map_path).convert('L')) for trav_map_path in trav_map_paths]
    
    with open(os.path.join(scene_root, scene, "floors.txt"), "r") as f:
        floor_heights = np.array(sorted(list(map(float, f.readlines()))))
    
    num_motions = num_motion_jobs[i]
    num_checked_motions = 0
    while len(os.listdir(scene_output_dir)) < num_motions:
        num_checked_motions += 1
        if num_checked_motions >= num_motions * 20:
            break

        rand_motion_idx = np.random.randint(0, len(all_motions))
        motion = all_motions[rand_motion_idx]
        motion_seq = np.load(os.path.join(motion_root, motion))
    
        rand_floor_idx = np.random.randint(0, len(trav_maps))
        trav_map = trav_maps[rand_floor_idx]
        all_traversable_pts = np.where(trav_map == 255)

        rand_floor_pt_idx = np.random.randint(0, all_traversable_pts[0].shape)
        rand_floor_pt = (all_traversable_pts[0][rand_floor_pt_idx][0], all_traversable_pts[1][rand_floor_pt_idx][0])
    
        rand_floor_height = floor_heights[rand_floor_idx]
    
        rand_floor_pt_3d = map_to_world(np.array([rand_floor_pt[0], rand_floor_pt[1]]), trav_map_size=trav_map.shape[0])
        rand_floor_pt_3d = np.concatenate((rand_floor_pt_3d, rand_floor_height[np.newaxis]))
    
        joints = motion_seq['joints']
        joints += rand_floor_pt_3d
    
        trans = motion_seq['trans']
        trans += rand_floor_pt_3d
    
        male_bm = BodyModel(bm_path=MALE_BM_PATH, num_betas=16, batch_size=motion_seq['trans'].shape[0]).to(device)
        female_bm = BodyModel(bm_path=FEMALE_BM_PATH, num_betas=16, batch_size=motion_seq['trans'].shape[0]).to(device)
        bm_world = male_bm if motion_seq['gender'][0] == 'male' else female_bm
        verts, faces = gen_obj_from_motion_seq(motion_seq["root_orient"], motion_seq["pose_body"], trans, motion_seq["betas"], bm_world, device)

        # collision checking
        person_out_of_scene_flag = False
        seq_end_idx = motion_seq['trans'].shape[0]                      # step of collision
        
        for i in range(verts.shape[0]):
            in_penetration = check_if_valid(verts[i], scene_mesh)
            valid_in_penetration = filter_floor_cols(verts[i], in_penetration, rand_floor_height + floor_buffer)
            if valid_in_penetration.sum() >= num_col_verts_thresh:
                seq_end_idx = i
                break
            else:
                if not is_person_in_scene(verts[i], scene_mesh.vertices):
                    person_out_of_scene_flag = True
                    seq_end_idx = i
                    break

        if seq_end_idx < min_seq_len:                                   # continue to next motion if sequence is too short
            continue

        motion_output_dir = os.path.join(scene_output_dir, motion[:-15])
        os.makedirs(motion_output_dir, exist_ok=True)

        # save motion sequence
        np.savez(os.path.join(motion_output_dir, "motion_seq.npz"), fps=30,
            gender=motion_seq['gender'],
            trans=trans[:seq_end_idx],
            root_orient=motion_seq['root_orient'][:seq_end_idx],
            pose_body=motion_seq['pose_body'][:seq_end_idx],
            betas=motion_seq['betas'],
            joints=joints[:seq_end_idx],
            joints_vel=motion_seq['joints_vel'][:seq_end_idx],
            trans_vel=motion_seq['trans_vel'][:seq_end_idx],
            root_orient_vel=motion_seq['root_orient_vel'][:seq_end_idx])

        # save collision labels
        if seq_end_idx == motion_seq['trans'].shape[0]:
            penetration_label = torch.tensor([100], device=device)
        else:
            if person_out_of_scene_flag:
                penetration_label = torch.tensor([100], device=device)
            else:
                penetration_label = process_penetration(valid_in_penetration, verts[seq_end_idx], \
                    joints[seq_end_idx], debug=False, counts_thresh=0, device=device)
                
                # collision spot
                col_verts = verts[seq_end_idx][valid_in_penetration]
                np.save(os.path.join(motion_output_dir, "col_verts.npy"), col_verts)
        np.save(os.path.join(motion_output_dir, "penetration_label.npy"), penetration_label.cpu().numpy())

        # additional collision labels
        all_addt_penetration_labels = []
        all_addt_col_verts = []
        addt_time_steps = []
        if seq_end_idx != motion_seq['trans'].shape[0] and not person_out_of_scene_flag:
            for i in range(addt_col_hor):
                if seq_end_idx + 1 + i >= motion_seq['trans'].shape[0]:
                    break
                addt_in_penetration = check_if_valid(verts[seq_end_idx + 1 + i], scene_mesh)
                valid_addt_in_penetration = filter_floor_cols(verts[seq_end_idx + 1 + i], addt_in_penetration, rand_floor_height + floor_buffer)

                if valid_addt_in_penetration.sum() < num_col_verts_thresh:
                    continue
                else:
                    if not is_person_in_scene(verts[seq_end_idx + 1 + i], scene_mesh.vertices):
                        break
                    else:
                        addt_penetration_label = process_penetration(valid_addt_in_penetration, \
                            verts[seq_end_idx + 1 + i], joints[seq_end_idx + 1 + i], debug=False, counts_thresh=0)
                        addt_col_verts = verts[seq_end_idx + 1 + i][valid_addt_in_penetration]

                        all_addt_penetration_labels.append(addt_penetration_label.cpu().numpy())
                        all_addt_col_verts.append(addt_col_verts)
                        addt_time_steps.append(i)

        if addt_time_steps:
            for i in range(len(addt_time_steps)):
                time_step = addt_time_steps[i]
                np.save(os.path.join(motion_output_dir, "addt_penetration_label_%02d.npy"%(time_step)), all_addt_penetration_labels[i])
                np.save(os.path.join(motion_output_dir, "addt_col_verts_%02d.npy"%(time_step)), all_addt_col_verts[i])

        if save_obj:
            motion_mesh_dir = os.path.join(motion_output_dir, "human_meshes")
            os.makedirs(motion_mesh_dir, exist_ok=True)
            for i in range(seq_end_idx):
                mesh_path = os.path.join(motion_mesh_dir, "%05d"%(i) + ".obj")
                write_to_obj(mesh_path, verts[i], faces.data.cpu().numpy())
    
        if vis_traj:
            root_traj_2d = joints[:, 0, :2]
            root_traj_2d = world_to_map(root_traj_2d) 
            traverse_map_with_traj = np.copy(trav_map)
            traverse_map_with_traj = np.repeat(traverse_map_with_traj[:, :, np.newaxis], 3, axis=2)
            traverse_map_with_traj[root_traj_2d[:, 0], root_traj_2d[:, 1], :] = np.array([255, 0, 0])
            im = PIL.Image.fromarray(traverse_map_with_traj)
            im.save(os.path.join(motion_output_dir, "traj.png"))
    
