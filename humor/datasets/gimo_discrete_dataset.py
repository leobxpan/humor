
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import glob, time, copy
import pickle

import numpy as np
import torch
from torchvision import transforms, utils

from datasets.amass_utils import TRAIN_DATASETS, TEST_DATASETS, VAL_DATASETS, SPLITS, SPLIT_BY, ROT_REPS
from datasets.amass_utils import CONTACT_INDS, NUM_BODY_JOINTS
from datasets.amass_utils import RETURN_CONFIGS
from datasets.amass_utils import data_name_list, data_dim

from utils.logging import Logger
from torch.utils.data import Dataset, DataLoader
from body_model.utils import SMPL_JOINTS
from utils.torch import copy2cpu as c2c
from utils.transforms import batch_rodrigues, rot6d_to_rotmat, compute_world2aligned_joints_mat, matrot2axisangle

ALL_SCENES = os.listdir("/scr/bxpan/gimo_processed/")

class GimoDiscreteDataset(Dataset):
    '''
    AMASS Motion Capture data subsampled to unified framerate.
    '''
    def __init__(self, split='train', 
                       data_paths=None, 
                       splits_path=None,
                       train_frac=0.8, val_frac=0.1,
                       sample_num_frames=10,
                       step_frames_in=1,
                       step_frames_out=1,
                       frames_out_step_size=1,
                       data_rot_rep='aa',
                       data_return_config='smpl+joints+contacts',
                       return_global=False,
                       only_global=False,
                       data_noise_std=0.0,
                       deterministic_train=False,
                       custom_split=None
                 ):
        '''
        - split : [train, val, test]
        - split_by : method to split the data [single, sequence, subject, dataset]
        - data_paths : list of data roots (value will depend on split_by type), can be regex
        - train_frac, val_frac : fraction of data to use for training/validation split, respectively (only for sequence or subject split)
        - sample_num_frames : the number of frames returned for each sequence, i.e. the number of input/output pairs
        - step_frames_in : the number of input frames (past observed frames) to return for each step of the sampled sequence. These will be contiguous at the data sampling rate. Each returned sequence will be zero padded at the front as needed.
        - step_frames_out : the number of out frames (future predicted frames) to return for each step of the sampled sequence. sampled at a rate of frames_out_step_size. No zero-padding is done for the end, only sequences with the available future frames are returned.
        - frames_out_step_size : in addition to immediate future step, how many steps between returned future frames. e.g. step_frames_out=3 and frames_out_step_size=4 will return the the 1st, 5th, and 9th future frames.
        - splits_path : directory to split files if want to use fixed splits
        - data_rot_rep : the rotation representation for the INPUT data. Output data is always given as a rotation matrix.
        - return_global : if true, returns the global motion trajectory (trans, orient, joints, joints_vel) in addition to the relative per-step data for each sequence.
        - only_global : if true, only returns the global sequence and not per step in/out
        - data_noise_std : standard deviation for gaussian noise to add to INPUT data
        - deterministic_train : if True, does not randomly choose sequences for training split
        - custom_split : a list of datasets to use as the split if split_by is 'custom'
        '''
        super(GimoDiscreteDataset, self).__init__()

        if data_paths is None:
            raise Exception('Must provide data paths')
        self.data_roots = data_paths
        Logger.log('Loading data from' + str(self.data_roots))

        self.splits_path = splits_path

        if split in SPLITS:
            self.split = split
        else:
            print(SPLITS)
            raise Exception('Not a valid split: %s' % (split))

        if data_rot_rep not in ROT_REPS:
            print(ROT_REPS)
            raise Exception('Not a valid rotation representation: %s' % (data_rot_rep))

        # only used if splitting by sequence or subject
        if train_frac + val_frac >= 1.0:
            raise Exception('train_frac and val_frac must add to less than 1')
        self.train_frac = train_frac
        self.val_frac = val_frac

        self.sample_num_frames = sample_num_frames
        self.step_frames_in = step_frames_in
        self.step_frames_out = step_frames_out
        self.frames_out_step_size = frames_out_step_size
        self.rot_rep = data_rot_rep
        self.only_global = only_global
        self.return_global = return_global or self.only_global
        self.noise_std = data_noise_std
        self.return_cfg = RETURN_CONFIGS[data_return_config]
        self.deterministic_train = deterministic_train
        self.custom_split = custom_split

        # based on the number of input and output frames (need self.sample_num_frames inputs and outputs)
        #self.effective_seq_len = self.sample_num_frames + 1 + (self.step_frames_out-1)*self.frames_out_step_size  # number of frames we need to subsample to build a data sequence
        self.effective_seq_len = self.sample_num_frames

        # prepare paths to sequences and their durations (in seconds, to determine sampling probability irrespective of fps)
        # "length" of dataset is how many subsequences we can chop it into
        #  subsequence maps data_idx -> (sequence_path_idx, start_frame, end_frame)
        self.sequence_paths, self.sequence_info, self.subseq_map = self.load_data()
        self.num_seq = len(self.sequence_paths)
        self.data_len = len(self.subseq_map)

        Logger.log('This split contains %d sequences (that meet the duration criteria).' % (self.num_seq))
        Logger.log('The dataset contains %d sub-sequences in total.' % (self.data_len))
    
    def parse_sequence_info(self, npz_file):
        ''' given npz file path, parses sequence duration from file name '''
        name_toks = npz_file[:-4].split('_')
        num_frames = int(name_toks[-4])
        fps = int(name_toks[-2])
        seq_dur = float(num_frames) / float(fps)
        return seq_dur, num_frames, fps

    def load_data(self):
        sequence_paths = []
        sequence_info = []
        subseq_map = {}

        data_root = self.data_roots[0]

        with open(os.path.join(self.splits_path, "%s_predict_dict.pkl"%self.split), 'rb') as f:
            split_pred_dict = pickle.load(f)

        seq_id_subseq_map = {}
        unique_seq_ids = list(set(split_pred_dict.keys()))
        for i in range(len(unique_seq_ids)):
            seq_id = unique_seq_ids[i]
            seq_id_subseq_map[seq_id] = i

            scene, seq = seq_id.split('_')[0], seq_id.split('_')[1]
            # special cases
            if scene == "seminar":
                scene = "_".join(seq_id.split('_')[:3])
                seq = seq_id[19:]
            elif scene == "middle":
                scene = "_".join(seq_id.split('_')[:2])
                seq = seq_id[16:]
            assert scene in ALL_SCENES

            npz_path = glob.glob(os.path.join(data_root, scene, seq, "*.npz"))
            assert len(npz_path) == 1, "more than one .npz files found at {} {}".format(scene, seq)
            sequence_paths.append(npz_path[0])

        subseq_cnt = 0
        for scene_seq, all_pred_inds in split_pred_dict.items():
            seq_id = seq_id_subseq_map[scene_seq]
            scene, seq = scene_seq.split('_')[0], scene_seq.split('_')[1]

            # special cases
            if scene == "seminar":
                scene = "_".join(scene_seq.split('_')[:3])
                seq = scene_seq[19:]
            elif scene == "middle":
                scene = "_".join(scene_seq.split('_')[:2])
                seq = scene_seq[16:]

            assert scene in ALL_SCENES

            for i in range(len(all_pred_inds)):
                pred_inds = all_pred_inds[i]

                npz_path = glob.glob(os.path.join(data_root, scene, seq, "*.npz"))
                assert len(npz_path) == 1, "more than one .npz files found at {} {}".format(scene, seq)

                start_idx = pred_inds[0] - 1
                assert start_idx >= 0, "negative start idx"

                end_idx = pred_inds[-1]
                assert end_idx - start_idx == 60, "incorrect subseq length"

                # start_idx is the input idx, end_idx is the last predicted idx (inclusive)
                subseq_map[subseq_cnt] = (seq_id, start_idx, end_idx)
                subseq_cnt += 1

        sequence_info = []

        return sequence_paths, sequence_info, subseq_map

    def __len__(self):
        return self.data_len

    def zero_pad_front(self, pad_size, array2pad):
        ''' array2pad : T x D1 x D2 x ... '''
        arr_size = list(array2pad.shape)
        arr_size[0] = pad_size
        padding = np.zeros(tuple(arr_size))
        padded_array = np.concatenate([padding, array2pad], axis=0)
        return padded_array

    def __getitem__(self, idx):
        start_t = time.time()
        seq_file_path = None
        sample_frame_inds = None
        sample_nframes = self.effective_seq_len # need num_frames with an input (past) and output (future) and future may be at large step size

        # want to choose deterministically
        # index into prepared subsequences
        seq_idx, start_frame_idx, end_frame_idx = self.subseq_map[idx]
        seq_file_path = self.sequence_paths[seq_idx]
        #sample_frame_inds = np.arange(start_frame_idx, end_frame_idx)
        sample_frame_inds = start_frame_idx

        # read in npz file (only read in what we need based on flags)
        data = np.load(seq_file_path)
        fps = data['fps']
        gender = str(data['gender'])

        # smpl
        world2aligned_rot = data['world2aligned_rot'][sample_frame_inds]
        #world2aligned_rot = np.eye(3)[None, :]
        betas = np.repeat(data['betas'][np.newaxis], self.sample_num_frames, axis=0)
        trans = data['trans'][sample_frame_inds][None, :]
        root_orient = data['root_orient'][sample_frame_inds][None, :]
        pose_body = data['pose_body'][sample_frame_inds][None, :]

        trans_vel = data['trans_vel'][sample_frame_inds][None, :]
        root_orient_vel = data['root_orient_vel'][sample_frame_inds][None, :]
        pose_body_vel = data['pose_body_vel'][sample_frame_inds][None, :]

        # Joints
        joints = data['joints'][sample_frame_inds][None, :]
        verts = data['mojo_verts'][sample_frame_inds][None, :]
        num_verts = verts.shape[1]

        joints_vel = data['joints_vel'][sample_frame_inds][None, :]
        verts_vel = data['mojo_verts_vel'][sample_frame_inds][None, :]
        joints_orient_vel = data['joint_orient_vel_seq'][:, None][sample_frame_inds][None, :]

        contacts = data['contacts'][sample_frame_inds]
        #contacts = contacts[:, CONTACT_INDS] # only need certain joints
        contacts = contacts[CONTACT_INDS][None, :]

        # build return dicts and convert to torch tensors
        meta = {'fps' : fps,
                'path' : '/'.join(seq_file_path.split('/')[-3:]),
                'gender' : gender,
                'betas' : torch.Tensor(betas)
        }

        pose_body_mat = batch_rodrigues(torch.Tensor(pose_body).reshape(-1, 3)).numpy().reshape((sample_nframes, NUM_BODY_JOINTS*9)) # T x 21 x 9
        root_orient_mat = batch_rodrigues(torch.Tensor(root_orient).reshape(-1, 3)).numpy() # T x 3 x 3

        # GLOBAL data is only output of immediate next step
        global_data_dict = dict()
        if self.return_global:
            # collect all outputs transformed to the frame of the initial input
            global_world2aligned_trans = np.zeros((1, 3))
            trans2joint = np.zeros((1, 1, 3))
            #global_world2aligned_rot = world2aligned_rot[0]
            global_world2aligned_rot = world2aligned_rot
            global_world2aligned_trans[0, :2] = -trans[0, :2]
            trans2joint[0, 0, :2] = -(joints[0:1, 0, :] + global_world2aligned_trans)[0, :2]

            #sidx_glob = 0 if self.only_global else 1 # return full vs only outputs
            sidx_glob = 0
            #glob_num_frames = (self.sample_num_frames + 1) if self.only_global else self.sample_num_frames
            glob_num_frames = 1
            eidx_glob = self.sample_num_frames + 1
            #eidx_glob = self.sample_num_frames
            if self.return_cfg['root_orient']:
                # root orient
                global_root_orient = np.matmul(global_world2aligned_rot, root_orient_mat[sidx_glob:eidx_glob].copy()).reshape((glob_num_frames, 9))
                # print(global_root_orient.shape)
                global_data_dict['global_root_orient'] = global_root_orient
            if self.return_cfg['trans']:
                # trans
                global_trans = trans[sidx_glob:eidx_glob].copy() + global_world2aligned_trans
                global_trans = np.matmul(global_world2aligned_rot, global_trans.T).T
                # print(global_trans.shape)
                global_data_dict['global_trans'] = global_trans
            if self.return_cfg['joints']:
                # joints
                global_joints = joints[sidx_glob:eidx_glob].copy() + global_world2aligned_trans.reshape((1,1,3))
                global_joints += trans2joint
                global_joints = np.matmul(global_world2aligned_rot, global_joints.reshape((-1, 3)).T).T.reshape((glob_num_frames, len(SMPL_JOINTS), 3))
                global_joints -= trans2joint
                # print(global_joints.shape)
                global_data_dict['global_joints'] = global_joints
            if self.return_cfg['verts']:
                # verts
                global_verts = verts[sidx_glob:eidx_glob].copy() + global_world2aligned_trans.reshape((1,1,3))
                global_verts += trans2joint
                global_verts = np.matmul(global_world2aligned_rot, global_verts.reshape((-1, 3)).T).T.reshape((glob_num_frames, num_verts, 3))
                global_verts -= trans2joint
                # print(global_verts.shape)
                global_data_dict['global_verts'] = global_verts
            if self.return_cfg['trans_vel']:
                global_trans_vel = trans_vel[sidx_glob:eidx_glob].copy()
                global_trans_vel = np.matmul(global_world2aligned_rot, global_trans_vel.T).T
                global_data_dict['global_trans_vel'] = global_trans_vel
            if self.return_cfg['root_orient_vel']:
                global_root_orient_vel = root_orient_vel[sidx_glob:eidx_glob].copy()
                global_root_orient_vel = np.matmul(global_world2aligned_rot, global_root_orient_vel.T).T
                global_data_dict['global_root_orient_vel'] = global_root_orient_vel
            if self.return_cfg['joints_vel']:
                # joints vel
                global_joints_vel = joints_vel[sidx_glob:eidx_glob].copy()
                global_joints_vel = np.matmul(global_world2aligned_rot, global_joints_vel.reshape((-1, 3)).T).T.reshape((glob_num_frames, len(SMPL_JOINTS), 3))
                # print(global_joints_vel.shape)
                global_data_dict['global_joints_vel'] = global_joints_vel
            if self.return_cfg['verts_vel']:
                # verts vel
                global_verts_vel = verts_vel[sidx_glob:eidx_glob].copy()
                global_verts_vel = np.matmul(global_world2aligned_rot, global_verts_vel.reshape((-1, 3)).T).T.reshape((glob_num_frames, num_verts, 3))
                # print(global_verts_vel.shape)
                global_data_dict['global_verts_vel'] = global_verts_vel
            if self.return_cfg['pose_body']:
                global_data_dict['global_pose_body'] = pose_body_mat[sidx_glob:eidx_glob].copy()
            if self.return_cfg['pose_body_vel']:
                global_data_dict['global_pose_body_vel'] = pose_body_vel[sidx_glob:eidx_glob].copy()
            if self.return_cfg['joints_orient_vel']:
                global_data_dict['global_joints_orient_vel'] = joints_orient_vel[sidx_glob:eidx_glob].copy().reshape((glob_num_frames, 1))
            if self.return_cfg['contacts']:
                global_data_dict['global_contacts'] = contacts[sidx_glob:eidx_glob].copy()

            if self.only_global:
                # only return global data
                data_out = dict()
                for k, v in global_data_dict.items():
                    data_out[k] = torch.Tensor(v)
                # add one to beta
                meta['betas'] = torch.zeros((glob_num_frames, 10))
                #meta['betas'] = meta['betas'][0:1, :].expand((glob_num_frames, meta['betas'].size(1)))
                return data_out, meta

        # set up transformation to canonical frame for all input sequences
        world2aligned_trans = np.zeros((self.sample_num_frames, 3))
        trans2joint = np.zeros((1, 1, 3))
        # align using smpl translation and root orientation
        world2aligned_rot = np.eye(3)[None, ...].repeat(self.sample_num_frames, axis=0)
        #world2aligned_rot = world2aligned_rot[0:self.sample_num_frames].copy()
        #world2aligned_trans[:, :2] = -trans[0:self.sample_num_frames, :2].copy()
        # offset between the translation origin and the root joint (depends on body shape beta)
        trans2joint[0, 0, :2] = -(joints[0, 0, :] + world2aligned_trans[0])[:2]

        # print(world2aligned_trans.shape)
        # print(world2aligned_rot.shape)
        # print(trans2joint.shape)

        if self.step_frames_in > 1:
            pad_size = self.step_frames_in - 1
            # must zero pad the front
            root_orient_mat = self.zero_pad_front(pad_size, root_orient_mat)
            root_orient_vel = self.zero_pad_front(pad_size, root_orient_vel)
            pose_body_mat = self.zero_pad_front(pad_size, pose_body_mat)
            pose_body_vel = self.zero_pad_front(pad_size, pose_body_vel)
            trans = self.zero_pad_front(pad_size, trans)
            trans_vel = self.zero_pad_front(pad_size, trans_vel)
            joints = self.zero_pad_front(pad_size, joints)
            joints_vel = self.zero_pad_front(pad_size, joints_vel)
            joints_orient_vel = self.zero_pad_front(pad_size, joints_orient_vel)
            verts = self.zero_pad_front(pad_size, verts)
            verts_vel = self.zero_pad_front(pad_size, verts_vel)
            contacts = self.zero_pad_front(pad_size, contacts)

        # build concated in/out array T x num_in+num_out x D for each data
        # slice_size = self.step_frames_in + self.step_frames_out
        #slice_size = self.step_frames_in + 1 + (self.step_frames_out-1)*self.frames_out_step_size
        slice_size = self.step_frames_in
        all_data_dict = dict()

        # relative body joint angles
        pose_body_in = pose_body_out = None
        if self.return_cfg['pose_body']:
            pose_body_data = np.zeros((self.sample_num_frames, slice_size, NUM_BODY_JOINTS*9))
            for t in range(self.sample_num_frames):
                pose_body_data[t] = pose_body_mat[t:t+slice_size].copy()
            pose_body_in = pose_body_data[:,:self.step_frames_in]
            pose_body_out = pose_body_data[:,self.step_frames_in::self.frames_out_step_size]
            
            if self.rot_rep == 'aa':
                pose_body_in = matrot2axisangle(pose_body_in.reshape((self.sample_num_frames, self.step_frames_in*NUM_BODY_JOINTS, 9))).reshape((self.sample_num_frames, self.step_frames_in, NUM_BODY_JOINTS*3))
            elif self.rot_rep == '6d':
                pose_body_in = pose_body_in.reshape((self.sample_num_frames, self.step_frames_in, NUM_BODY_JOINTS, 9))[:,:,:,:6].reshape((self.sample_num_frames, self.step_frames_in, NUM_BODY_JOINTS*6))
        all_data_dict['pose_body'] = (pose_body_in, pose_body_out)

        # relative body joint angle velocities
        pose_body_vel_in = pose_body_vel_out = None
        if self.return_cfg['pose_body_vel']:
            pose_body_vel_data = np.zeros((self.sample_num_frames, slice_size, NUM_BODY_JOINTS, 3))
            for t in range(self.sample_num_frames):
                pose_body_vel_data[t] = pose_body_vel[t:t+slice_size].copy()
            pose_body_vel_in = pose_body_vel_data[:,:self.step_frames_in]
            pose_body_vel_out = pose_body_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['pose_body_vel'] = (pose_body_vel_in, pose_body_vel_out)

        # smpl root orientation
        root_orient_in = root_orient_out = None
        if self.return_cfg['root_orient']:
            root_orient_mat_seq_data = np.zeros((self.sample_num_frames, slice_size, 3, 3))
            for t in range(self.sample_num_frames):
                cur_align_rot = world2aligned_rot[t] # transform to frame of the last input step
                alinged_root_orient_slice = np.matmul(cur_align_rot, root_orient_mat[t:t+slice_size].copy())
                root_orient_mat_seq_data[t] = alinged_root_orient_slice

            root_orient_mat_seq_data = root_orient_mat_seq_data.reshape((self.sample_num_frames, slice_size, 9))
            root_orient_in = root_orient_mat_seq_data[:,:self.step_frames_in]
            root_orient_out = root_orient_mat_seq_data[:,self.step_frames_in::self.frames_out_step_size]

            if self.rot_rep == 'aa':
                root_orient_in = matrot2axisangle(root_orient_in.reshape((self.sample_num_frames, self.step_frames_in, 9))).reshape((self.sample_num_frames, self.step_frames_in, 3))
            elif self.rot_rep == '6d':
                root_orient_in = root_orient_in.reshape((self.sample_num_frames, self.step_frames_in, 9))[:,:,:6]
        all_data_dict['root_orient'] = (root_orient_in, root_orient_out)

        # smpl root orientation angular velocity
        root_orient_vel_in = root_orient_vel_out = None
        if self.return_cfg['root_orient_vel']:
            root_orient_vel_data = np.zeros((self.sample_num_frames, slice_size, 3))
            for t in range(self.sample_num_frames):
                cur_align_rot = world2aligned_rot[t] # transform to frame of the last input step
                cur_root_orient_vel_data = root_orient_vel[t:t+slice_size].copy()
                aligned_root_orient_vel = np.matmul(cur_align_rot, cur_root_orient_vel_data.T).T
                root_orient_vel_data[t] = aligned_root_orient_vel

            root_orient_vel_in = root_orient_vel_data[:,:self.step_frames_in]
            root_orient_vel_out = root_orient_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['root_orient_vel'] = (root_orient_vel_in, root_orient_vel_out)

        # smpl root translation
        trans_in = trans_out = None
        if self.return_cfg['trans']:
            trans_data = np.zeros((self.sample_num_frames, slice_size, 3))
            for t in range(self.sample_num_frames):
                cur_trans_data = trans[t:t+slice_size].copy()
                # transform to frame of the last input step
                cur_trans_data = cur_trans_data + world2aligned_trans[t:t+1]
                cur_align_rot = world2aligned_rot[t]
                aligned_cur_trans_data = np.matmul(cur_align_rot, cur_trans_data.T).T
                if self.step_frames_in > 1 and t < (self.step_frames_in - 1):
                    # reset padding to zero
                    aligned_cur_trans_data[:(self.step_frames_in - 1 - t)] = 0.0
                trans_data[t] = aligned_cur_trans_data

            trans_in = trans_data[:,:self.step_frames_in]
            trans_out = trans_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['trans'] = (trans_in, trans_out)

        # smpl_root_velocity
        trans_vel_in = trans_vel_out = None
        if self.return_cfg['trans_vel']:
            trans_vel_data = np.zeros((self.sample_num_frames, slice_size, 3))
            for t in range(self.sample_num_frames):
                cur_align_rot = world2aligned_rot[t] # transform to frame of the last input step
                cur_trans_vel_data = trans_vel[t:t+slice_size].copy()
                aligned_trans_vel = np.matmul(cur_align_rot, cur_trans_vel_data.T).T
                trans_vel_data[t] = aligned_trans_vel

            trans_vel_in = trans_vel_data[:,:self.step_frames_in]
            trans_vel_out = trans_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['trans_vel'] = (trans_vel_in, trans_vel_out)

        # joint positions
        joints_in = joints_out = None
        if self.return_cfg['joints']:
            joints_data = np.zeros((self.sample_num_frames, slice_size, len(SMPL_JOINTS), 3))
            for t in range(self.sample_num_frames):
                cur_joints_data = joints[t:t+slice_size].copy()
                # transform to frame of the last input step
                cur_joints_data = cur_joints_data + world2aligned_trans[t].reshape((1,1,3)) + trans2joint
                cur_align_rot = world2aligned_rot[t] 
                aligned_cur_joints_data = np.matmul(cur_align_rot, cur_joints_data.reshape((-1, 3)).T).T.reshape((slice_size, len(SMPL_JOINTS), 3))
                # back to align with global translation. Note, we do not need to rotate the offset because the global rotation is actually done about the root joint
                aligned_cur_joints_data = aligned_cur_joints_data - trans2joint
                if self.step_frames_in > 1 and t < (self.step_frames_in - 1):
                    # reset padding to zero
                    aligned_cur_joints_data[:(self.step_frames_in - 1 - t)] = 0.0
                joints_data[t] = aligned_cur_joints_data

            joints_in = joints_data[:,:self.step_frames_in]
            joints_out = joints_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['joints'] = (joints_in, joints_out)

        # joints angular velocity around z
        joints_orient_vel_in = joints_orient_vel_out = None
        if self.return_cfg['joints_orient_vel']:
            joints_orient_vel_data = np.zeros((self.sample_num_frames, slice_size, 1))
            for t in range(self.sample_num_frames):
                joints_orient_vel_data[t] = joints_orient_vel[t:t+slice_size].copy().reshape((slice_size, 1))

            joints_orient_vel_in = joints_orient_vel_data[:,:self.step_frames_in]
            joints_orient_vel_out = joints_orient_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['joints_orient_vel'] = (joints_orient_vel_in, joints_orient_vel_out)

        # vertex positions
        verts_in = verts_out = None
        if self.return_cfg['verts']:
            verts_data = np.zeros((self.sample_num_frames, slice_size, num_verts, 3))
            for t in range(self.sample_num_frames):
                cur_verts_data = verts[t:t+slice_size].copy()
                # transform to frame of the last input step
                cur_verts_data = cur_verts_data + world2aligned_trans[t].reshape((1,1,3)) + trans2joint
                cur_align_rot = world2aligned_rot[t] 
                aligned_cur_verts_data = np.matmul(cur_align_rot, cur_verts_data.reshape((-1, 3)).T).T.reshape((slice_size, num_verts, 3))
                # back to align with global translation. Note, we do not need to rotate the offset because the global rotation is actually done about the root joint
                aligned_cur_verts_data = aligned_cur_verts_data - trans2joint
                if self.step_frames_in > 1 and t < (self.step_frames_in - 1):
                    # reset padding to zero
                    aligned_cur_verts_data[:(self.step_frames_in - 1 - t)] = 0.0
                verts_data[t] = aligned_cur_verts_data

            verts_in = verts_data[:,:self.step_frames_in]
            verts_out = verts_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['verts'] = (verts_in, verts_out)

        # joint positional velocities
        joints_vel_in = joints_vel_out = None
        if self.return_cfg['joints_vel']:
            # joint velocities
            joint_vel_data = np.zeros((self.sample_num_frames, slice_size, len(SMPL_JOINTS), 3))
            for t in range(self.sample_num_frames):
                cur_align_rot = world2aligned_rot[t] # transform to frame of the last input step
                cur_joint_vel_data = joints_vel[t:t+slice_size].copy()
                aligned_joint_vel = np.matmul(cur_align_rot, cur_joint_vel_data.reshape((slice_size*len(SMPL_JOINTS), 3)).T).T.reshape((slice_size, len(SMPL_JOINTS), 3))
                joint_vel_data[t] = aligned_joint_vel

            joints_vel_in = joint_vel_data[:,:self.step_frames_in]
            joints_vel_out = joint_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['joints_vel'] = (joints_vel_in, joints_vel_out)

        # vertex velocities
        verts_vel_in = verts_vel_out = None
        if self.return_cfg['verts_vel']:
            # verts velocities
            vert_vel_data = np.zeros((self.sample_num_frames, slice_size, num_verts, 3))
            for t in range(self.sample_num_frames):
                cur_align_rot = world2aligned_rot[t] # transform to frame of the last input step
                cur_vert_vel_data = verts_vel[t:t+slice_size].copy()
                aligned_vert_vel = np.matmul(cur_align_rot, cur_vert_vel_data.reshape((slice_size*num_verts, 3)).T).T.reshape((slice_size, num_verts, 3))
                vert_vel_data[t] = aligned_vert_vel

            verts_vel_in = vert_vel_data[:,:self.step_frames_in]
            verts_vel_out = vert_vel_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['verts_vel'] = (verts_vel_in, verts_vel_out)

        # contacts
        contacts_in = contacts_out = None
        if self.return_cfg['contacts']:
            contacts_data = np.zeros((self.sample_num_frames, slice_size, len(CONTACT_INDS)))
            for t in range(self.sample_num_frames):
                contacts_data[t] = contacts[t:t+slice_size].copy()
            contacts_in = contacts_data[:,:self.step_frames_in]
            contacts_out = contacts_data[:,self.step_frames_in::self.frames_out_step_size]
        all_data_dict['contacts'] = (contacts_in, contacts_out)

        # prepare final output
        data_in = dict()
        data_out = dict()
        for k, v in all_data_dict.items():
            if self.return_cfg[k]:
                # add noise to inputs
                cur_in = v[0]
                cur_out = v[1]

                data_in[k] = torch.Tensor(cur_in)
                data_out[k] = torch.Tensor(cur_out)

        if self.return_global:
            for k, v in global_data_dict.items():
                data_out[k] = torch.Tensor(v)

        data_in["seq_file_path"] = seq_file_path
        data_in["start_frame_idx"] = start_frame_idx
        data_in["end_frame_idx"] = end_frame_idx

        meta["betas"] = torch.zeros(1, 10)
        return data_in, data_out, meta

if __name__=='__main__':
    # dataset
    data_paths = ['./data/amass_30_fps_no_terrain_contacts']
    data_rot_rep = 'mat'
    data_return_config = 'smpl+joints+contacts'
    return_config_dict = RETURN_CONFIGS[data_return_config]
    sample_num_frames = 120
    steps_in = 1
    steps_out = 1
    frames_out_step_size = 1
    slice_size = steps_in + steps_out
    noise_std = 0.0
    dataset = AmassDiscreteDataset( split='train', 
                                    data_paths=data_paths,
                                    step_frames_in=steps_in,
                                    step_frames_out=steps_out,
                                    frames_out_step_size=frames_out_step_size,
                                    split_by='dataset',
                                    train_frac=0.8, val_frac=0.1,
                                    sample_num_frames=sample_num_frames,
                                    data_rot_rep=data_rot_rep,
                                    data_return_config=data_return_config,
                                    return_global=True,
                                    data_noise_std=noise_std,
                                    )

    # create loaders
    batch_size = 1
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                        worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    from viz.utils import viz_smpl_seq
    from body_model.body_model import BodyModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    male_bm_path = os.path.join('./body_models/smplh', 'male/model.npz')
    male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=slice_size).to(device)
    male_bm_world = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=sample_num_frames+1).to(device)
    male_bm_global = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=sample_num_frames, use_vtx_selector=False).to(device)
    female_bm_path = os.path.join('./body_models/smplh', 'female/model.npz')
    female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=slice_size).to(device)
    female_bm_world = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=sample_num_frames+1).to(device)
    female_bm_global = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=sample_num_frames, use_vtx_selector=False).to(device)

    for i, data in enumerate(loader):
        data_in, data_out, meta = data

        print(meta.keys())
        print(data_in.keys())
        print(data_out.keys())

        print(meta['fps'])
        print(meta['gender'])
        print(meta['path'])

        betas = meta['betas'].to(device)

        #
        # Visualize each subsequence
        #

        if data_return_config == 'smpl+joints+contacts':
            root_orient_in = data_in['root_orient']
            print(root_orient_in.size())
            if data_rot_rep == 'mat':
                root_orient_in = matrot2axisangle(root_orient_in.numpy().reshape((batch_size*sample_num_frames, steps_in, 9))).reshape((batch_size, sample_num_frames, steps_in, 3))
                root_orient_in = torch.Tensor(root_orient_in)
            root_orient_out_mat = data_out['root_orient']
            root_orient_out_aa = matrot2axisangle(root_orient_out_mat.numpy().reshape((batch_size*sample_num_frames, steps_out, 9))).reshape((batch_size, sample_num_frames, steps_out, 3))
            root_orient = torch.cat([root_orient_in, torch.Tensor(root_orient_out_aa)], axis=2).to(device)

            pose_body_in = data_in['pose_body']
            print(pose_body_in.size())
            if data_rot_rep == 'mat':
                pose_body_in = matrot2axisangle(pose_body_in.numpy().reshape((batch_size*sample_num_frames, NUM_BODY_JOINTS*steps_in, 9))).reshape((batch_size, sample_num_frames, steps_in, NUM_BODY_JOINTS*3))
                pose_body_in = torch.Tensor(pose_body_in)
            pose_body_out_mat = data_out['pose_body']
            pose_body_out_aa = matrot2axisangle(pose_body_out_mat.numpy().reshape((batch_size*sample_num_frames, NUM_BODY_JOINTS*steps_out, 9))).reshape((batch_size, sample_num_frames, steps_out, NUM_BODY_JOINTS*3))
            pose_body_out_aa = torch.Tensor(pose_body_out_aa)
            pose_body = torch.cat([pose_body_in, pose_body_out_aa], axis=2).to(device)

            # print(pose_body_in.size())

            trans_in = data_in['trans']
            print(trans_in.size())
            trans_out = data_out['trans']
            trans = torch.cat([trans_in, trans_out], axis=2).to(device)

            trans_vel_in = data_in['trans_vel']
            print(trans_vel_in.size())
            trans_vel_out = data_out['trans_vel']
            trans_vel = torch.cat([trans_vel_in, trans_vel_out], axis=2).to(device)

            root_orient_vel_in = data_in['root_orient_vel']
            print(root_orient_vel_in.size())
            root_orient_vel_out = data_out['root_orient_vel']
            root_orient_vel = torch.cat([root_orient_vel_in, root_orient_vel_out], axis=2).to(device)

            joints_in = data_in['joints']
            print(joints_in.size())
            joints_out = data_out['joints']
            joints = torch.cat([joints_in, joints_out], axis=2).to(device)

            joints_vel_in = data_in['joints_vel']
            print(joints_vel_in.size())
            joints_vel_out = data_out['joints_vel']
            joints_vel = torch.cat([joints_vel_in, joints_vel_out], axis=2).to(device)

        if return_config_dict['contacts']:
            contacts_in = data_in['contacts']
            contacts_out = data_out['contacts']
            contacts = torch.cat([contacts_in, contacts_out], axis=2).to(device)
            print(contacts.size())
        
        bm = male_bm if meta['gender'][0] == 'male' else female_bm
        # NOTE: show each in/out slice individually
        for t in range(0, sample_num_frames):
            # print('t: %d' % (t))
            render_body = True
            render_joints = True
            joints_seq = joints_vel_seq = None
            if data_return_config == 'smpl+joints+contacts':
                body = bm(pose_body=pose_body[0, t], pose_hand=None, betas=betas[0, :slice_size], root_orient=root_orient[0, t], trans=trans[0, t])
                joints_seq = joints[0, t]
                joints_vel_seq = trans_vel[0, t].reshape((-1, 1, 3)).repeat((1, 22, 1))
            else:
                body = bm(betas=betas[0, :slice_size])
                render_body = False
                joints_seq = joints[0, t]

                joints_vel_seq = torch.zeros((slice_size, 3)).to(device)
                joints_vel_seq[:, 2] = joints_orient_vel[0, t, :, 0]
                joints_vel_seq = joints_vel_seq.reshape((-1, 1, 3)).repeat((1, 22, 1))

            contacts_seq = None
            if return_config_dict['contacts']:
                contacts_seq = torch.zeros((slice_size, len(SMPL_JOINTS))).to(device)
                contacts_seq[:,CONTACT_INDS] = contacts[0, t]

            
            # viz_smpl_seq(body, imw=1080, imh=1080, fps=steps_in+steps_out, contacts=contacts_seq,
            #     render_body=render_body, render_joints=render_joints, render_skeleton=(data_return_config=='joints' or data_return_config=='joints+verts'), render_ground=True,
            #     joints_seq=joints_seq) #,
                # joints_vel=joints_vel_seq)
                # points_seq=verts[0, t])
                # points_vel=verts_vel[0, t]) #, vtx_list=MOJO_VERTS)
                # joints_seq=None)

        
        #
        # Then directly use returned global
        #
        root_orient = data_out['global_root_orient']
        root_orient = matrot2axisangle(root_orient.numpy().reshape((batch_size, sample_num_frames, 9))).reshape((batch_size, sample_num_frames, 3))
        root_orient = torch.Tensor(root_orient).to(device)

        pose_body = data_out['global_pose_body']
        pose_body = matrot2axisangle(pose_body.numpy().reshape((batch_size*sample_num_frames, NUM_BODY_JOINTS, 9))).reshape((batch_size, sample_num_frames, NUM_BODY_JOINTS*3))
        pose_body = torch.Tensor(pose_body).to(device)

        trans = data_out['global_trans'].to(device)
        joints = data_out['global_joints'].to(device)
        joints_vel = data_out['global_joints_vel'].to(device)
        trans_vel = data_out['global_trans_vel'].to(device)

        viz_contacts = None
        if return_config_dict['contacts']:
            contacts = data_out['global_contacts']
            print(contacts.size())
            viz_contacts = torch.zeros((batch_size, sample_num_frames, len(SMPL_JOINTS))).to(contacts)
            viz_contacts[:,:,CONTACT_INDS] = contacts
            print(viz_contacts.size())
        
        bm_global = male_bm_global if meta['gender'][0] == 'male' else female_bm_global
        body = bm_global(pose_body=pose_body[0], pose_hand=None, betas=betas[0,0].reshape((1, -1)).expand((sample_num_frames, 16)), root_orient=root_orient[0], trans=trans[0])
        viz_smpl_seq(body, imw=1080, imh=1080, fps=30,
            render_body=True, render_joints=True, render_skeleton=False, render_ground=True,
            contacts=viz_contacts[0],
            joints_seq=joints[0])
            # joints_vel=root_orient_vel[0].reshape((-1, 1, 3)).repeat((1, 22, 1)))
            # points_seq=verts[0],
            # joints_vel=joints_vel[0],
            # points_vel=verts_vel[0]) 
            #vtx_list=MOJO_VERTS) 
            # joints_seq=None)
