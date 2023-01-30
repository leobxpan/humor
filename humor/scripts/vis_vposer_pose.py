import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pickle

import trimesh

import torch
import smplx

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.rotation_tools import matrot2aa

from body_model.body_model import BodyModel
from body_model.utils import SMPLX_PATH

#from dataset.utils import batch_smplx_forward

vposer, _ = load_vposer('../vposer_v1_0', vp_model='snapshot')

with open('../smplh_dict.pkl', 'rb') as f:
    smplh_dict = pickle.load(f)

pose_body = matrot2aa(smplh_dict["pose_body"].reshape((-1, 3, 3)))

# latents = vposer.encode(pose_body.reshape((150, -1))).rsample()
decoded_poses = vposer(pose_body.reshape(150, -1))
decoded_poses = matrot2aa(decoded_poses["pose"].view(-1, 3, 3)).reshape((150, -1))

# body_mesh_model = smplx.create('smplx_models',
#                                model_type='smplx',
#                                gender='neutral', ext='npz',
#                                num_pca_comps=12,
#                                create_global_orient=True,
#                                create_body_pose=True,
#                                create_betas=True,
#                                create_left_hand_pose=True,
#                                create_right_hand_pose=True,
#                                create_expression=True,
#                                create_jaw_pose=True,
#                                create_leye_pose=True,
#                                create_reye_pose=True,
#                                create_transl=True,
#                                batch_size=150,
#                                num_betas=10,
#                                num_expression_coeffs=10)
neutral_bm_path = os.path.join(SMPLX_PATH, 'SMPLX_NEUTRAL.npz')
neutral_bm = BodyModel(bm_path=neutral_bm_path, num_betas=16, batch_size=150)

root_orient = matrot2aa(smplh_dict["root_orient"].view(-1, 3, 3)).view(150, 3)

#smplx_output = batch_smplx_forward(vposer, body_mesh_model, root_orient, smplh_dict["trans"].squeeze(0), latents)
smplx_output = neutral_bm(pose_body=decoded_poses, \
                          pose_hand=torch.zeros(150, 90).float(), \
                          betas=torch.zeros(150, 16).float(), \
                          root_orient=root_orient, \
                          trans=smplh_dict['trans'].squeeze())

for i in range(10):
    trimesh.PointCloud(smplx_output.v[i].detach().cpu().numpy()).export("humor/decoded_future_%d.obj"%i)
