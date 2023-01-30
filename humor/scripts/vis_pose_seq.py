import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch

import trimesh

from body_model.body_model import BodyModel
from body_model.utils import SMPLX_PATH, SMPLH_PATH
from viz.utils import viz_smpl_seq, create_video

neutral_bm_path = os.path.join(SMPLX_PATH, 'SMPLX_NEUTRAL.npz')
#neutral_bm_path = os.path.join(SMPLH_PATH, 'neutral/model.npz')

seq = "/scr/bxpan/gimo_processed/seminar_room0_0219/2022-02-18-223709/poses_3475_frames_30_fps.npz"
#seq = "/orion/u/bxpan/exoskeleton/gaze_dataset/bedroom0122/2022-01-21-194925/poses.npz"
#seq = "/scr-ssd/bxpan/gimo_processed/bedroom0122/2022-01-21-194925/bedroom0122_2022-01-21-194925_poses_838_frames_30_fps.npz"
seq = np.load(seq, allow_pickle=True)

#neutral_bm = BodyModel(bm_path=neutral_bm_path, num_betas=16, batch_size=seq['poses'].shape[0])
neutral_bm = BodyModel(bm_path=neutral_bm_path, num_betas=16, batch_size=seq['pose_body'].shape[0])

# body_pred = neutral_bm(pose_body=torch.from_numpy(seq['poses'][:, 3:66]).float(), \
#                        pose_hand=torch.from_numpy(seq['poses'][:, 75:]).float(), \
#                        betas=torch.zeros(seq['poses'].shape[0], 16).float(), \
#                        root_orient=torch.from_numpy(seq['root_orient']).float(), \
#                        trans=torch.from_numpy(seq['trans']).float())

body_pred = neutral_bm(pose_body=torch.from_numpy(seq['pose_body']).float(), \
                       pose_hand=torch.zeros(seq['pose_body'].shape[0], 90).float(), \
                       betas=torch.zeros(seq['pose_body'].shape[0], 16).float(), \
                       root_orient=torch.from_numpy(seq['root_orient']).float(), \
                       trans=torch.from_numpy(seq['trans']).float())

idx = 2176
trimesh.PointCloud(body_pred.v[idx].detach().cpu().numpy()).export('hist.obj')
import pdb; pdb.set_trace()
viz_smpl_seq(body_pred, use_offscreen=True)
create_video('./render_out/' + '/frame_%08d.' + '%s' % ('png'), 'test_processed.mp4', 30)
