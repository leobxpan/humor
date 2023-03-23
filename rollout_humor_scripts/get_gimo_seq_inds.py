import os, sys
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
sys.path.append(os.path.join(cur_file_path, '../..'))

from collections import defaultdict
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src import config
from dataset import unit_single_scene_dataset

def main(cfg):
    # train_dataset = unit_single_scene_dataset.UnitSingleSceneDataset(cfg, train=True)
    # test_dataset = unit_single_scene_dataset.UnitSingleSceneDataset(cfg, train=False)

    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=cfg['training']['num_workers'], pin_memory=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg['training']['num_workers'], pin_memory=True)

    # train_dict = defaultdict(list)
    # test_dict = defaultdict(list)
    # cnt = 0
    # with torch.no_grad():
    #   for data in tqdm(train_dataloader):
    #       _, _, _, _, _, _, _, _, seq, scene, poses_predict_idx, poses_input_idx, _, _ = data

    #       gimo_input_idx = poses_input_idx[-1].item()
    #       gimo_output_idx_end = poses_predict_idx[-2].item()
    #       gimo_output_inds = np.arange(gimo_input_idx + 1, gimo_output_idx_end + 1, 1).tolist()
    #       if len(gimo_output_inds) != 150: print("not full seq, {}, {}".format(poses_input_idx, poses_predict_idx))

    #       scene_seq_id = scene[0] + "_" + seq[0]
    #       train_dict[scene_seq_id].append(gimo_output_inds)

    #   for data in tqdm(test_dataloader):
    #       _, _, _, _, _, _, _, _, seq, scene, poses_predict_idx, poses_input_idx, _, _ = data

    #       gimo_input_idx = poses_input_idx[-1].item()
    #       gimo_output_idx_end = poses_predict_idx[-2].item()
    #       gimo_output_inds = np.arange(gimo_input_idx + 1, gimo_output_idx_end + 1, 1).tolist()
    #       if len(gimo_output_inds) != 150: print("not full seq, {}, {}".format(poses_input_idx, poses_predict_idx))

    #       scene_seq_id = scene[0] + "_" + seq[0]
    #       test_dict[scene_seq_id].append(gimo_output_inds)

    #   with open("rollout_humor_scripts/train_predict_dict.pkl", "wb") as f:
    #       pickle.dump(train_dict, f)

    #   with open("rollout_humor_scripts/test_predict_dict.pkl", "wb") as f:
    #       pickle.dump(test_dict, f)

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    #load_dir = os.path.join("humor/rollout_humor_scripts", "all_scenes_stride_{}_hist_{}_future_{}".format(cfg['data']['stride'], cfg['data']['input_seq_len'], cfg['data']['output_seq_len']))
    load_dir = os.path.join("humor/rollout_humor_scripts", "all_scenes_stride_1_hist_1_future_60_latents")

    train_poses = np.load(os.path.join(load_dir, "train_poses.npy"))
    train_seqs = np.load(os.path.join(load_dir, "train_seqs.npy"))
    train_scenes = np.load(os.path.join(load_dir, "train_scenes.npy"))
    train_start_end = np.load(os.path.join(load_dir, "train_start_end.npy"))
    assert (len(train_seqs) == len(train_scenes)) and (len(train_seqs) == len(train_start_end)) and (len(train_seqs) == len(train_poses)), "training seq lengths not same"

    test_poses = np.load(os.path.join(load_dir, "test_poses.npy"))
    test_seqs = np.load(os.path.join(load_dir, "test_seqs.npy"))
    test_scenes = np.load(os.path.join(load_dir, "test_scenes.npy"))
    test_start_end = np.load(os.path.join(load_dir, "test_start_end.npy"))
    assert (len(test_seqs) == len(test_scenes)) and (len(test_seqs) == len(test_start_end)) and (len(test_seqs) == len(test_poses)), "testing seq lengths not same"

    fps = cfg['data']['fps']
    input_len = cfg['data']['input_seq_len']
    output_len = cfg['data']['output_seq_len']
    for i in tqdm(range(len(train_seqs))):
        ego_idx = train_poses[i]
        scene = train_scenes[i]
        seq = train_seqs[i]
        start_frame, end_frame = train_start_end[i]

        first_predict_idx = ego_idx + int((input_len - 1) * 30 / fps) + 1
        last_predict_idx = ego_idx + int(input_len * 30 / fps) + int((output_len - 1) * 30 / fps)
        predict_inds = np.arange(first_predict_idx, last_predict_idx + 1, 1).tolist()

        if len(predict_inds) != output_len * 30 / fps:
            print("training seq not complete, ", scene_seq_id)

        scene_seq_id = scene + "_" + seq
        train_dict[scene_seq_id].append(predict_inds)

    for i in tqdm(range(len(test_seqs))):
        ego_idx = test_poses[i]
        scene = test_scenes[i]
        seq = test_seqs[i]
        start_frame, end_frame = test_start_end[i]

        first_predict_idx = ego_idx + int((input_len - 1) * 30 / fps) + 1
        last_predict_idx = ego_idx + int(input_len * 30 / fps) + int((output_len - 1) * 30 / fps)
        predict_inds = np.arange(first_predict_idx, last_predict_idx + 1, 1).tolist()

        if len(predict_inds) != output_len * 30 / fps:
            print("training seq not complete, ", scene_seq_id)

        scene_seq_id = scene + "_" + seq
        test_dict[scene_seq_id].append(predict_inds)

    with open(os.path.join(load_dir, "train_predict_dict.pkl".format(cfg['data']['stride'], cfg['data']['input_seq_len'], cfg['data']['output_seq_len'])), "wb") as f:
        pickle.dump(train_dict, f)

    with open(os.path.join(load_dir, "test_predict_dict.pkl".format(cfg['data']['stride'], cfg['data']['input_seq_len'], cfg['data']['output_seq_len'])), "wb") as f:
        pickle.dump(test_dict, f)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(42)
    cfg_path = sys.argv[1]
    cfg = config.load_config(cfg_path, "configs/default.yaml")

    main(cfg)
