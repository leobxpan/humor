# post process the saved humor rollout sequences to get the vposer latents
# this script does not change the coordinate frame, so the processed data is still in the humor frame

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pickle

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import trimesh

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.rotation_tools import matrot2aa

import smplx

from body_model.body_model import BodyModel
from body_model.utils import SMPLX_PATH

def batch_to_yup(root_orient, trans, pelvis):
    # transform from z-up to y-up
    rot_mat = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
      ])
    root_orient_mat = R.from_rotvec(root_orient).as_matrix()
    root_orient_mat = rot_mat @ root_orient_mat
    root_orient = R.from_matrix(root_orient_mat).as_rotvec()
 
    trans = (rot_mat @ (pelvis + trans).T).T - pelvis

    return root_orient, trans

vposer, _ = load_vposer('../vposer_v1_0', vp_model='snapshot')

neutral_bm_path = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")
neutral_bm = BodyModel(bm_path=neutral_bm_path, num_betas=16, batch_size=1)

body_mesh_model = smplx.create(SMPLX_PATH,
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
                               batch_size=150,
                               num_betas=10,
                               num_expression_coeffs=10)

humor_seqs_path = "./rollout_humor_scripts/all_scenes_stride_1/humor_gen_seqs/test_seqs.pkl"
with open(humor_seqs_path, 'rb') as f:
    humor_seqs = pickle.load(f)

for scene_seq, scene_seq_dict in humor_seqs.items():
    for start_idx, subseq_list in scene_seq_dict.items():
        for i in range(len(subseq_list)):
            seq = subseq_list[i]            # this is the actual dict for a seq

            pose_body = matrot2aa(seq["pose_body"].reshape((-1, 3, 3))).view(150, -1, 3)
            latents = vposer.encode(pose_body.reshape((150, -1))).rsample()
            seq["latents"] = latents.cpu()
            seq["root_orient"] = matrot2aa(seq["root_orient"].view(-1, 3, 3)).view(150, 3)
            seq["trans"] = seq["trans"].squeeze(0)

            # transform to y-up
            pose = {}
            pose["body_pose"] = pose_body.cpu()
            pose["pose_embedding"] = latents.detach()
            pose["global_orient"] = torch.zeros(150, 3)
            pose["transl"] = torch.zeros(150, 3)
            smplx_output = body_mesh_model(return_verts=True, **pose)

            pelvis = smplx_output.joints[:, 0].detach().cpu().numpy()
            root_orient, trans = batch_to_yup(seq["root_orient"], seq["trans"].cpu().numpy(), pelvis)

            seq["root_orient"] = root_orient
            seq["trans"] = trans

            # debug visualizations
            # # before transformation
            # pose["global_orient"] = seq["root_orient"]
            # pose["transl"] = seq["trans"]
            # smplx_output = body_mesh_model(return_verts=True, **pose)
            # body_verts_batch = smplx_output.vertices
            # smplx_faces = body_mesh_model.faces
            # for i in range(10):
            #     out_mesh = trimesh.Trimesh(body_verts_batch[i].detach().cpu().numpy(), smplx_faces, process=False)
            #     out_mesh.export('before_%d.obj'%i)

            # # after transformation
            # pose["global_orient"] = torch.tensor(root_orient).float()
            # pose["transl"] = torch.tensor(trans).float()
            # smplx_output = body_mesh_model(return_verts=True, **pose)
            # body_verts_batch = smplx_output.vertices
            # smplx_faces = body_mesh_model.faces
            # for i in range(10):
            #     out_mesh = trimesh.Trimesh(body_verts_batch[i].detach().cpu().numpy(), smplx_faces, process=False)
            #     out_mesh.export('after_%d.obj'%i)

with open(humor_seqs_path.replace('.pkl', '_yup.pkl'), 'wb') as f:
    pickle.dump(humor_seqs, f)
