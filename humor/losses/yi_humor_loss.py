
import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model.body_model import BodyModel
from body_model.utils import SMPLH_PATH, SMPL_JOINTS, VPOSER_PATH, SMPL_PARENTS, KEYPT_VERTS
from utils.transforms import rotation_matrix_to_angle_axis
from datasets.amass_utils import CONTACT_INDS

NUM_BODY_JOINTS = len(SMPL_JOINTS) - 1
BETA_SIZE = 16
CONTACT_THRESH = 0.5

class YiHumorLoss(nn.Module):

    def __init__(self,
                    kl_loss=1.0,
                    kl_loss_anneal_start=0,
                    kl_loss_anneal_end=0,
                    kl_loss_cycle_len=-1, # if > 0 will anneal KL loss cyclicly
                    regr_trans_loss=1.0,
                    regr_trans_vel_loss=1.0,
                    regr_root_orient_loss=1.0,
                    regr_root_orient_vel_loss=1.0,
                    regr_pose_loss=1.0,
                    regr_pose_vel_loss=1.0,
                    regr_joint_loss=1.0,
                    regr_joint_vel_loss=1.0,
                    regr_joint_orient_vel_loss=1.0,
                    regr_vert_loss=1.0,
                    regr_vert_vel_loss=1.0,
                    contacts_loss=0.0, # classification loss on binary contact prediction
                    contacts_vel_loss=0.0, # velocity near 0 at predicted contacts
                    smpl_joint_loss=0.0,
                    smpl_mesh_loss=0.0,
                    smpl_joint_consistency_loss=0.0,
                    smpl_vert_consistency_loss=0.0,
                    smpl_batch_size=480):
        super(YiHumorLoss, self).__init__()
        '''
        All loss inputs are weights for that loss term. If the weight is 0, the loss is not used.

        - regr_*_loss :                 L2 regression losses on various state terms (root trans/orient, body pose, joint positions, and joint velocities)
        - smpl_joint_loss :             L2 between GT joints and joint locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_mesh_loss :              L2 between GT and predicted vertex locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_joint_consistency_loss : L2 between regressed joints and predicted joint locations from SMPL model (ensures consistency between
                                        state joint locations and joint angle predictions)
        - kl_loss :                     divergence between predicted posterior and prior

        - smpl_batch_size : the size of batches that will be given to smpl. if less than this is passed in, will be padded accordingly. however, passed
                            in batches CANNOT be greater than this number.
        '''
        self.kl_loss_weight = kl_loss
        self.kl_loss_anneal_start = kl_loss_anneal_start
        self.kl_loss_anneal_end = kl_loss_anneal_end
        self.use_kl_anneal = self.kl_loss_anneal_end > self.kl_loss_anneal_start

        self.kl_loss_cycle_len = kl_loss_cycle_len
        self.use_kl_cycle = False
        if self.kl_loss_cycle_len > 0:
            self.use_kl_cycle = True
            self.use_kl_anneal = False

        self.contacts_loss_weight = contacts_loss
        self.contacts_vel_loss_weight = contacts_vel_loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # build dict of all possible regression losses based on inputs
        # keys must be the same as we expect from the pred/gt data
        self.regr_loss_weight_dict = {
            'trans' : regr_trans_loss,
            'trans_vel' : regr_trans_vel_loss,
            'root_orient' : regr_root_orient_loss,
            'root_orient_vel' : regr_root_orient_vel_loss,
            'pose_body' : regr_pose_loss,
            'pose_body_vel' : regr_pose_vel_loss,
            'joints' : regr_joint_loss,
            'joints_vel' : regr_joint_vel_loss,
            'joints_orient_vel' : regr_joint_orient_vel_loss,
            'verts' : regr_vert_loss,
            'verts_vel' : regr_vert_vel_loss
        }

        self.smpl_joint_loss_weight = smpl_joint_loss
        self.smpl_mesh_loss_weight = smpl_mesh_loss
        self.smpl_joint_consistency_loss_weight = smpl_joint_consistency_loss
        self.smpl_vert_consistency_loss_weight = smpl_vert_consistency_loss

        self.l2_loss = nn.MSELoss(reduction='none')
        self.regr_loss = nn.MSELoss(reduction='none')

        smpl_losses = [self.smpl_joint_loss_weight, self.smpl_mesh_loss_weight, self.smpl_joint_consistency_loss_weight, self.smpl_vert_consistency_loss_weight]
        self.smpl_batch_size = smpl_batch_size
        self.use_smpl_losses = False
        if sum(smpl_losses) > 0.0:
            self.use_smpl_losses = True
            # need a body model to compute the losses
            male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
            self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=self.smpl_batch_size)
            female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
            self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=self.smpl_batch_size)

    def forward(self, pred_dict, gt_dict, cur_epoch, gender=None, betas=None):
        '''
        Compute the loss.

        All data in the dictionaries should be of size B x D.

        group_regr_losses will be used to aggregate every group_regr_lossth batch idx together into
        the stats_dict. This can be useful when there are multiple output steps and you want to track
        each separately.
        '''
        loss = 0.0
        stats_dict = dict()

        #
        # KL divergence
        #
        if self.kl_loss_weight > 0.0:
            qm, qv = pred_dict['posterior_distrib']
            pm, pv = pred_dict['prior_distrib']
            kl_loss = self.kl_normal(qm, qv, pm, pv)
            kl_stat_loss = kl_loss.mean()
            kl_loss = kl_stat_loss
            # print(kl_loss.size())
            stats_dict['kl_loss'] = kl_stat_loss
            anneal_weight = 1.0
            if self.use_kl_anneal or self.use_kl_cycle:
                anneal_epoch = cur_epoch
                anneal_start = self.kl_loss_anneal_start
                anneal_end = self.kl_loss_anneal_end
                if self.use_kl_cycle:
                    anneal_epoch = cur_epoch % self.kl_loss_cycle_len
                    anneal_start = 0
                    anneal_end = self.kl_loss_cycle_len // 2 # optimize full weight for second half of cycle
                if anneal_epoch >= anneal_start:
                    anneal_weight = (anneal_epoch - anneal_start) / (anneal_end - anneal_start)
                else:
                    anneal_weight = 0.0
                anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight

            import pdb; pdb.set_trace()
            loss = loss + anneal_weight*self.kl_loss_weight*kl_loss

            stats_dict['kl_anneal_weight'] = anneal_weight
            stats_dict['kl_weighted_loss'] = loss

        # 
        # Reconstruction 
        #

        # regression terms
        for cur_key in gt_dict.keys():
            # print(cur_key)
            if cur_key not in self.regr_loss_weight_dict:
                continue
            cur_regr_weight = self.regr_loss_weight_dict[cur_key]
            if cur_regr_weight > 0.0:
                pred_val = pred_dict[cur_key]
                gt_val = gt_dict[cur_key]

                if cur_key == 'root_orient' or cur_key == 'pose_body':
                    # rotations use L2 for matrices
                    cur_regr_loss = self.l2_loss(pred_val, gt_val)
                else:
                    cur_regr_loss = self.regr_loss(pred_val, gt_val)
                agg_cur_regr_loss = cur_regr_loss
                cur_regr_stat_loss = agg_cur_regr_loss.mean()
                agg_cur_regr_loss = cur_regr_stat_loss
                stats_dict[cur_key + '_loss'] = cur_regr_stat_loss
                loss = loss + cur_regr_weight*agg_cur_regr_loss

        if self.kl_loss_weight > 0.0:
            stats_dict['reconstr_weighted_loss'] = loss - stats_dict['kl_weighted_loss']
        
        return loss, stats_dict

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x D
        '''
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
        return new_pad_list

    
    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension
        ​
        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance
        ​
        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def log_normal(self, x, m, v):
        """
        Computes the elem-wise log probability of a Gaussian and then sum over the
        last dim. Basically we're assuming all dims are batch dims except for the
        last dim.    Args:
            x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
            m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
            v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance    Return:
            log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
                each sample. Note that the summation dimension is not kept
        """
        log_prob = -torch.log(torch.sqrt(v)) - math.log(math.sqrt(2*math.pi)) \
                        - ((x - m)**2 / (2*v))
        log_prob = torch.sum(log_prob, dim=-1)
        return log_prob

    