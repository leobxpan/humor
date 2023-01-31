# post process the saved humor rollout sequences to get the vposer latents
# this script does not change the coordinate frame, so the processed data is still in the humor frame

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pickle

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.rotation_tools import matrot2aa

vposer, _ = load_vposer('../vposer_v1_0', vp_model='snapshot')

humor_seqs_path = "./rollout_humor_scripts/all_scenes_stride_1/humor_gen_seqs/train.pkl"
with open(humor_seqs_path, 'rb') as f:
    humor_seqs = pickle.load(f)

for scene_seq, scene_seq_dict in humor_seqs.items():
    for start_idx, subseq_list in scene_seq_dict.items():
        for i in range(len(subseq_list)):
            seq = subseq_list[i]            # this is the actual dict for a seq

            pose_body = matrot2aa(seq["pose_body"]).reshape((-1, 3, 3))
            latents = vposer.encode(pose_body.reshape((150, -1))).rsample()
            seq["latents"] = latents.cpu()
            seq["root_orient"] = matrot2aa(seq["root_orient"].view(-1, 3, 3)).view(150, 3)

with open(humor_seqs_path, 'wb') as f:
    pickle.dump(humor_seqs, f)
