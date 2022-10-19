import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import pickle
import glob

import numpy as np
from scipy.spatial.transform import Rotation as R

import trimesh

import torch

from tqdm import tqdm

import smplx

from human_body_prior.tools.model_loader import load_vposer
from viz.utils import viz_smplx_seq, create_video

from config.config import MotionFromGazeConfig

import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def to_zup(root_orient, trans, pelvis):
    # transform from y-up to z-up
    rot_mat = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
      ])
    root_orient_mat = R.from_rotvec(root_orient).as_matrix()
    root_orient_mat = rot_mat @ root_orient_mat
    root_orient = R.from_matrix(root_orient_mat).as_rotvec()
    
    trans = rot_mat @ (pelvis + trans) - pelvis

    return root_orient, trans

data_root = "/orion/u/bxpan/exoskeleton/gaze_dataset"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = MotionFromGazeConfig().parse_args()
vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
vposer = vposer.to(device)

body_mesh_model = smplx.create(config.smplx_path,
                               model_type='smplx',
                               gender='neutral', ext='npz',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=1,
                               num_betas=10,
                               num_expression_coeffs=10)

with torch.no_grad():
    for scene in os.listdir(data_root):
        scene_path = os.path.join(data_root, scene)
        if os.path.isdir(scene_path):
            for split in tqdm(os.listdir(scene_path)):
                if split == "scene_obj": continue
                all_pkls = glob.glob(os.path.join(scene_path, split, "smplx_local", "*.pkl"))
                all_pkls = sorted(all_pkls, key=natural_key)
    
                all_poses = []
                all_root_oris = []
                all_root_trans = []
                all_latents = []
                all_jaw_poses = []
                all_betas = []
                all_expressions = []
                all_lhand_poses = []
                all_rhand_poses = []
   
                count = 0
                for i in range(0, len(all_pkls), 10):
                    pkl = all_pkls[i]
                    pose_pkl = pickle.load(open(pkl, 'rb'))

                    root_orient = pose_pkl["orient"].cpu().numpy()
                    trans = pose_pkl["trans"].cpu().numpy()
                    latents = pose_pkl["latent"].cpu().numpy()
                    pose_jaw = pose_pkl["jaw_pose"].cpu().numpy()
                    betas = pose_pkl["beta"].cpu().numpy()
                    expressions = pose_pkl["expression"].cpu().numpy()
                    lhand = pose_pkl["lhand"].cpu().numpy()
                    rhand = pose_pkl["rhand"].cpu().numpy()

                    body_pose = vposer.decode(pose_pkl["latent"].to(device), output_type='aa')

                    poses = np.zeros(165)
                    poses[:3] = root_orient
                    poses[3:66] = body_pose.cpu().numpy().flatten()
                    poses[75:120] = lhand
                    poses[120:] = rhand

                    # transform from y-up to z-up
                    pose = {}
                    pose["body_pose"] = body_pose.cpu().unsqueeze(0)
                    pose["pose_embedding"] = torch.tensor(latents).unsqueeze(0)
                    pose["global_orient"] = torch.zeros(3).unsqueeze(0)
                    pose["transl"] = torch.zeros(3).unsqueeze(0)
                    smplx_output = body_mesh_model(return_verts=True, **pose)

                    pelvis = smplx_output.joints[0, 0].cpu().numpy()
                    root_orient, trans = to_zup(root_orient, trans, pelvis)
                    poses[:3] = root_orient

                    # debugging visualization
                    # pose["global_orient"] = torch.tensor(root_orient).unsqueeze(0).to(torch.float32)
                    # pose["transl"] = torch.tensor(trans).unsqueeze(0).to(torch.float32)
                    # smplx_output = body_mesh_model(return_verts=True, **pose)
                    # body_verts_batch = smplx_output.vertices
                    # smplx_faces = body_mesh_model.faces
                    # out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)
                    # out_mesh.export(pkl.replace(".pkl", "_rot.obj"))

                    all_poses.append(poses)
                    all_root_oris.append(root_orient)
                    all_root_trans.append(trans)
                    all_latents.append(latents)
                    all_jaw_poses.append(pose_jaw)
                    all_betas.append(betas)
                    all_expressions.append(expressions)
                    all_lhand_poses.append(lhand)
                    all_rhand_poses.append(rhand)

                all_poses = np.stack(all_poses)
                all_root_oris = np.stack(all_root_oris)
                all_root_trans = np.stack(all_root_trans)
                all_latents = np.stack(all_latents)
                all_jaw_poses = np.stack(all_jaw_poses)
                all_betas = np.stack(all_betas)
                all_expressions = np.stack(all_expressions)
                all_lhand_poses = np.stack(all_lhand_poses)
                all_rhand_poses = np.stack(all_rhand_poses)

                output_path = os.path.join(scene_path, split, "poses.npz")
                np.savez(output_path, fps=30,
                    poses=all_poses,
                    root_orient=all_root_oris,
                    trans=all_root_trans,
                    latents=all_latents,
                    pose_jaw=all_jaw_poses,
                    betas=all_betas,
                    expressions=all_expressions,
                    lhand=all_lhand_poses,
                    rhand=all_rhand_poses)
