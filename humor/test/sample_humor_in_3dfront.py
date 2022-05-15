import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import argparse
import importlib, time

import glob
import json

import numpy as np

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.config import TestConfig
from utils.logging import Logger, class_name_to_file_name, mkdir, cp_files
from utils.torch import get_device, save_state, load_state
from utils.stats import StatTracker
from utils.transforms import rotation_matrix_to_angle_axis
from body_model.utils import SMPL_JOINTS
from datasets.amass_utils import NUM_KEYPT_VERTS, CONTACT_INDS
from losses.humor_loss import CONTACT_THRESH
from libmesh.inside_mesh import check_mesh_contains

NUM_WORKERS = 0
NUM_VERTS_THRESH = 5
#THREEDFRONT_ROOT = '/orion/u/paschald/Datasets/3D-FRONT_scenes/'
#THREEDFRONT_ROOT = '/orion/u/bxpan/exoskeleton/3dfront_dataset/'
THREEDFRONT_ROOT = '/orion/group/3D-FRONT_scenes/'
ADDT_COL_HOR = 10           # additional buffer to check collision after the actual collision (as a tolerance for collision checking)

def parse_args(argv):
    # create config and parse args
    config = TestConfig(argv)
    known_args, unknown_args = config.parse()
    print('Unrecognized args: ' + str(unknown_args))
    return known_args

def swap_axes(mesh):
    mesh_x = np.array(mesh.vertices[:, 0])
    mesh_y = np.array(mesh.vertices[:, 1])
    mesh_z = np.array(mesh.vertices[:, 2])

    new_mesh_verts = np.stack((-mesh_x, mesh_z, mesh_y), axis=1)
    new_mesh = trimesh.Trimesh(vertices=new_mesh_verts, faces=mesh.faces)

    return new_mesh

def write_to_obj(dest_obj_file, vertices, faces):
    # vertices: N X 3, faces: N X 3
    w_f = open(dest_obj_file, 'w')

    # print("Total vertices:{0}".format(vertices.shape[0]))

    # Write vertices to file
    for idx in range(vertices.shape[0]):
        w_f.write("v "+str(vertices[idx, 0])+" "+str(vertices[idx, 1])+" "+str(vertices[idx, 2])+"\n")

    # Write faces to file 
    for idx in range(faces.shape[0]):
        w_f.write("f "+str(faces[idx, 0]+1)+" "+str(faces[idx, 1]+1)+" "+str(faces[idx, 2]+1)+"\n")

    w_f.close() 

def seq_is_forward_walking(motion_seq, end_idx, check_hor=30, check_thresh=0.5):
    root_ori = motion_seq["root_orient"].reshape(-1, 3, 3)
    root_trans = motion_seq["joints"][0, :, 0, :]

    dot_products = []
    for t in range(end_idx-2, end_idx-2-check_hor, -1):
        root_ori_t = root_ori[t]
        root_ori_t_x = -root_ori_t[:, 0]
        root_ori_t_y = root_ori_t[:, 2]
        root_ori_t_z = root_ori_t[:, 1]
        root_ori_t = torch.stack((root_ori_t_x, root_ori_t_y, root_ori_t_z), axis=1)
        ori = root_ori_t @ torch.FloatTensor([0, 1, 0]).to(root_ori.device)
        ori = ori / torch.norm(ori)

        root_trans_t = root_trans[t, :]
        root_trans_t1 = root_trans[t+1, :]
        vel = root_trans_t1 - root_trans_t
        vel_pred = motion_seq["trans_vel"][0, t, :]
        vel = vel / torch.norm(vel)
        #print("vel:", vel)
        #print("vel_pred:", vel_pred)

        dot_products.append(ori @ vel)
    dot_products = torch.stack(dot_products)

    print(dot_products)
    if torch.all(dot_products > check_thresh):
        return True
    else:
        return False

# def is_person_in_scene(human_verts, scene_bounds):
#     human_verts = human_verts[0][:, :2]
#     
#     scene_bounds_min = scene_bounds[0][:2].repeat(human_verts.shape[0], 1)
#     scene_bounds_max = scene_bounds[1][:2].repeat(human_verts.shape[0], 1)
# 
#     return (human_verts >= scene_bounds_min).all().item() and (human_verts <= scene_bounds_max).all().item()

def is_person_in_scene(human_verts, floor_verts):
    floor_2d = floor_verts[:, :2]
    human_2d = human_verts[:, :2]

    floor_min = floor_2d.min(axis=0) - 0.5
    floor_max = floor_2d.max(axis=0) + 0.5

    resolution = 64

    floor_img = np.zeros((resolution, resolution))
    floor_2d = (floor_2d - floor_min) / (floor_max - floor_min)
    floor_2d = np.rint(floor_2d).astype(int)
    for i in range(floor_2d.shape[0]):
        floor_img[floor_2d[i, 0], floor_2d[i, 1]] = 1

    human_img = np.zeros((resolution, resolution))
    human_2d = (human_2d - floor_min) / (floor_max - floor_min)
    human_2d = np.rint(human_2d).astype(int)
    for i in range(human_2d.shape[0]):
        human_img[human_2d[i, 0], human_2d[i, 1]] = 1

    union = np.logical_or(floor_img, human_img)
    diff = union - floor_img

    if diff.sum() == 0:
        return True
    else:
        return False

def check_if_valid(vertices, scene):
    # vertices: bs(1) X N X 3
    in_penetration = check_mesh_contains(scene, vertices.squeeze(0).cpu().numpy())
    return in_penetration

def reject_outliers(data, m = 2., return_idx=False):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    if return_idx:
        return data[s<m], s<m
    else:
        return data[s<m]

def gen_data_npz(x_pred_dict, meta, actual_t, end_idx, cano_rot_inv, dest_npz_path):
    # x_pred_dict: contains data in the canonical frame of the first frame's pose. 
    # ['trans', 'trans_vel', 'root_orient', 'root_orient_vel', 'pose_body', 'joints', 'joints_vel', 'contacts']
    # cano_rot_inv: 4 X 4, transform the canonical frame of the first pose to the aligned_floor frame. 
    # new_world2aligned_rot: T X 3 X 3 
    # new_cano_rot_mat_arr: T X 4 X 4 
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    new_trans = x_pred_dict['trans'][0, :end_idx][-actual_t:] # T X 3
    new_trans_vel= x_pred_dict['trans_vel'][0, :end_idx][-actual_t:] # T X 3
    new_root_orient = x_pred_dict['root_orient'][0, :end_idx][-actual_t:] # T X 9 
    new_root_orient_vel = x_pred_dict['root_orient_vel'][0, :end_idx][-actual_t:] # T X 3
    new_pose_body = x_pred_dict['pose_body'][0, :end_idx][-actual_t:] # T X 189
    new_joints = x_pred_dict['joints'][0, :end_idx][-actual_t:] # T X 66
    new_joints_vel = x_pred_dict['joints_vel'][0, :end_idx][-actual_t:] # T X 66 

    # J = len(SMPL_JOINTS)
    # V = NUM_KEYPT_VERTS
    # male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    # female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
    # male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=actual_t).to(new_trans.device)
    # female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=actual_t).to(new_trans.device)
    
    # male_bm_path = os.path.join(SMPLX_PATH, 'SMPLX_MALE.npz')
    # female_bm_path = os.path.join(SMPLX_PATH, 'SMPLX_FEMALE.npz')
    # male_bm = BodyModel(bm_path=male_bm_path, num_betas=10, batch_size=actual_t).to(new_trans.device)
    # female_bm = BodyModel(bm_path=female_bm_path, num_betas=10, batch_size=actual_t).to(new_trans.device)

   
    # # Change root translation 
    # curr_pelvis_joint = new_joints[0:1, :3] - new_trans[0:1, :] # 1 X 3 
    # curr_cam_pose_trans = cano_rot_inv[:3, 3][None].to(new_trans.device) # 1 X 3
    # sum_for_mul = curr_pelvis_joint + new_trans # T X 3
    # new_trans = torch.matmul(cano_rot_inv[:3, :3].repeat(actual_t, 1, 1).to(new_trans.device), \
    #     sum_for_mul[:, :, None])[:, :, 0] + curr_cam_pose_trans - curr_pelvis_joint
    # T X 3 

    np.savez(dest_npz_path, fps=30,
        gender=meta['gender'],
        trans=new_trans.data.cpu().numpy(), # T X 3
        root_orient=new_root_orient.data.cpu().numpy(), # T X 3 
        pose_body=new_pose_body.data.cpu().numpy(), # T X 63
        betas=meta['betas'][0, 0].data.cpu().numpy(), # 10
        joints=new_joints.data.cpu().numpy(), # T X 22 X 3
        joints_vel=new_joints_vel.data.cpu().numpy(), # T X 22 X 3
        trans_vel=new_trans_vel.data.cpu().numpy(), # T X 3
        root_orient_vel=new_root_orient_vel.data.cpu().numpy()) # T X 3
        # joint_orient_vel_seq=joint_orient_vel_seq, # T: Based on joints_world2aligned_rot, calculate angular velocity for z-axis rotation. 
        # pose_body_vel=pose_body_vel_seq, # T X 21 X 3

    # Visualize for debug generated data 
    vis_debug = False 
    if vis_debug:
        human_verts, human_faces = get_human_verts(new_root_orient[None], new_pose_body[None], new_trans[None], new_joints[None], \
            meta, male_bm, female_bm)
        # human_verts: BS X T X Nv X 3
        # human_faces: Nf X 3 
        
        debug_fodler = os.path.join("./tmp_check_generated_nav_data", meta['scene_name'][0])
        if not os.path.exists(debug_fodler):
            os.makedirs(debug_fodler)
        for t_idx in range(actual_t):
            dest_mesh_path = os.path.join(debug_fodler, "%05d"%(t_idx)+".obj")
            write_to_obj(dest_mesh_path, human_verts[0, t_idx], human_faces.data.cpu().numpy())

def test(args_obj, config_file):

    # set up output
    args = args_obj.base
    mkdir(args.out)

    # create logging system
    test_log_path = os.path.join(args.out, 'test.log')
    Logger.init(test_log_path)

    # save arguments used
    Logger.log('Base args: ' + str(args))
    Logger.log('Model args: ' + str(args_obj.model))
    Logger.log('Dataset args: ' + str(args_obj.dataset))
    Logger.log('Loss args: ' + str(args_obj.loss))

    # save training script/model/dataset/config used
    test_scripts_path = os.path.join(args.out, 'test_scripts')
    mkdir(test_scripts_path)
    pkg_root = os.path.join(cur_file_path, '..')
    dataset_file = class_name_to_file_name(args.dataset)
    dataset_file_path = os.path.join(pkg_root, 'datasets/' + dataset_file + '.py')
    model_file = class_name_to_file_name(args.model)
    loss_file = class_name_to_file_name(args.loss)
    model_file_path = os.path.join(pkg_root, 'models/' + model_file + '.py')
    train_file_path = os.path.join(pkg_root, 'test/test_humor.py')
    # cp_files(test_scripts_path, [train_file_path, model_file_path, dataset_file_path, config_file])

    # load model class and instantiate
    model_class = importlib.import_module('models.' + model_file)
    Model = getattr(model_class, args.model)
    model = Model(**args_obj.model_dict,
                    model_smpl_batch_size=args.batch_size) # assumes model is HumorModel

    # load loss class and instantiate
    loss_class = importlib.import_module('losses.' + loss_file)
    Loss = getattr(loss_class, args.loss)
    loss_func = Loss(**args_obj.loss_dict,
                      smpl_batch_size=args.batch_size*args_obj.dataset.sample_num_frames) # assumes loss is HumorLoss

    device = get_device(args.gpu)
    model.to(device)
    loss_func.to(device)

    print(model)

    # count params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Logger.log('Num model params: ' + str(params))

    # freeze params in loss
    for param in loss_func.parameters():
        param.requires_grad = False

    # load in pretrained weights if given
    if args.ckpt is not None:
        start_epoch, min_val_loss, min_train_loss = load_state(args.ckpt, model, optimizer=None, map_location=device, ignore_keys=model.ignore_keys)
        Logger.log('Successfully loaded saved weights...')
        Logger.log('Saved checkpoint is from epoch idx %d with min val loss %.6f...' % (start_epoch, min_val_loss))
    else:
        Logger.log('ERROR: No weight specified to load!!')
        # return

    # load dataset class and instantiate training and validation set
    if args.test_on_train:
        Logger.log('WARNING: running evaluation on TRAINING data as requested...should only be used for debugging!')
    elif args.test_on_val:
        Logger.log('WARNING: running evaluation on VALIDATION data as requested...should only be used for debugging!')
    Dataset = getattr(importlib.import_module('datasets.' + dataset_file), args.dataset)
    split = 'test'
    if args.test_on_train:
        split = 'train'
    elif args.test_on_val:
        split = 'val'
    test_dataset = Dataset(split=split, **args_obj.dataset_dict)

    # only select a subset of data
    #subset_indices = np.random.choice(len(test_dataset), size=args.num_batches, replace=False)
    #subset_sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
    # create loaders
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            #sampler=subset_sampler,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    test_dataset.return_global = True
    model.dataset = test_dataset

    if args.eval_full_test:
        Logger.log('Running full test set evaluation...')
        # stats tracker
        tensorboard_path = os.path.join(args.out, 'test_tensorboard')
        mkdir(tensorboard_path)
        stat_tracker = StatTracker(tensorboard_path)

        # testing with same stats as training
        test_start_t = time.time()
        test_dataset.pre_batch()
        model.eval()
        for i, data in enumerate(test_loader):
            batch_start_t = time.time()
            # run model
            #   note we're always using ground truth input so this is only measuring single-step error, just like in training
            loss, stats_dict = model_class.step(model, loss_func, data, test_dataset, device, 0, mode='test', use_gt_p=1.0)

            # collect stats
            batch_elapsed_t = time.time() - batch_start_t
            total_elapsed_t = time.time() - test_start_t
            stats_dict['loss'] = loss
            stats_dict['time_per_batch'] = torch.Tensor([batch_elapsed_t])[0]

            stat_tracker.update(stats_dict, tag='test')

            if i % args.print_every == 0:
                stat_tracker.print(i, len(test_loader),
                                0, 1,
                                total_elapsed_time=total_elapsed_t,
                                tag='test')

            test_dataset.pre_batch()

    metadata = {
        "dataset_type": args.dataset_type,
    }
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    if args.eval_sampling or args.eval_sampling_debug:
        #all_houses = sorted(os.listdir(THREEDFRONT_ROOT))
        all_houses = np.load('/orion/u/bxpan/exoskeleton/all_scenes_new.npy')
        #all_houses = ['LivingDiningRoom-291']
        job_num = len(all_houses) // args.num_workers
        if args.worker_id == args.num_workers - 1:
            jobs = all_houses[args.worker_id*job_num:]
        else:
            jobs = all_houses[args.worker_id*job_num: (args.worker_id+1)*job_num]

        #jobs = ['LivingRoom-71076']

        # region_index_list = house_region_index_dict[house_name]
        # region_name_list = house_region_name_dict[house_name]
        # for i in range(len(region_index_list)):
        #     region_index = region_index_list[i]
        #     region_name = region_name_list[i]

        eval_sampling(model, test_dataset, test_loader, device, jobs,
                            out_dir=args.out if args.eval_sampling else None,
                            num_samples=args.eval_num_samples,
                            samp_len=args.eval_sampling_len,
                            viz_contacts=args.viz_contacts,
                            viz_pred_joints=args.viz_pred_joints,
                            viz_smpl_joints=args.viz_smpl_joints,
                            write_obj=args.write_obj, 
                            save_seq_len=args.seq_len,
                            debug=args.debug,
                            dataset_type=args.dataset_type,
                            num_batches=args.num_batches)

    Logger.log('Finished!')

def eval_sampling(model, test_dataset, test_loader, device, houses, 
                  out_dir=None,
                  num_samples=1,
                  samp_len=10.0,
                  viz_contacts=False,
                  viz_pred_joints=False,
                  viz_smpl_joints=False,
                  write_obj=False,
                  save_seq_len=None,
                  debug=False,
                  dataset_type="nomap_22",
                  num_batches=50):
    Logger.log('Evaluating sampling qualitatively...')
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    eval_qual_samp_len = int(samp_len * 30.0) # at 30 Hz

    for house in houses:
        scene = trimesh.exchange.load.load(os.path.join(THREEDFRONT_ROOT, house, 'watertight', 'scene_objects.off'))
        #scene_with_floor = trimesh.exchange.load.load(os.path.join(THREEDFRONT_ROOT, house, 'floor.off'))
        floor = trimesh.exchange.load.load(os.path.join(THREEDFRONT_ROOT, house, 'watertight', 'floor.off'))
        wall = trimesh.exchange.load.load(os.path.join(THREEDFRONT_ROOT, house, 'watertight', 'wall.off'))

        scene = swap_axes(scene)
        floor = swap_axes(floor)
        wall = swap_axes(wall)
        
        scene_bounds_min = torch.tensor(wall.vertices.min(axis=0)).to(device)
        scene_bounds_max = torch.tensor(wall.vertices.max(axis=0)).to(device)
        scene_bounds = (scene_bounds_min, scene_bounds_max)

        res_out_dir = None
        if out_dir is not None:
            #res_out_dir = os.path.join(out_dir, 'eval_sampling')
            house_out_dir = os.path.join(out_dir, house)
            if not os.path.exists(house_out_dir):
                os.mkdir(house_out_dir)

        J = len(SMPL_JOINTS)
        V = NUM_KEYPT_VERTS
        male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
        female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
        male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)
        female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)

        saved_batch_cnt = 0
        max_batch = num_batches * 50                # give up on this scenes if evaluated over this number of samples
        batch_cnt = 0
        with torch.no_grad():
            test_dataset.pre_batch()
            model.eval()
            for i, data in enumerate(test_loader):
                person_out_of_scene_flag = False
                # get inputs
                batch_in, batch_out, meta = data
                print(meta['path'])
                seq_name_list = [spath[:-4] for spath in meta['path']]
                if house_out_dir is None:
                    batch_res_out_list = [None]*len(seq_name_list)
                else:
                    batch_res_out_list = [os.path.join(house_out_dir, seq_name.replace('/', '_') + '_b' + str(i) + 'seq' + str(sidx)) for sidx, seq_name in enumerate(seq_name_list)]
                    print(batch_res_out_list)
                # continue
                x_past, _, gt_dict, input_dict, global_gt_dict = model.prepare_input(batch_in, device, 
                                                                                    data_out=batch_out,
                                                                                    return_input_dict=True,
                                                                                    return_global_dict=True)

                # roll out predicted motion
                B, T, _, _ = x_past.size()
                x_past = x_past[:,0,:,:] # only need input for first step
                rollout_input_dict = dict()
                for k in input_dict.keys():
                    rollout_input_dict[k] = input_dict[k][:,0,:,:] # only need first step

                min_seq_len = 30

                # sample same trajectory multiple times and save the joints/contacts output
                #for samp_idx in range(num_samples):
                x_pred_dict = model.roll_out(x_past, rollout_input_dict, eval_qual_samp_len, gender=meta['gender'], betas=meta['betas'].to(device))

                # translate the human to the scene
                x_pred_dict = translate_to_scene(x_pred_dict, floor)

                # visualize and save
                #print('Visualizing sample %d/%d!' % (samp_idx+1, num_samples))
                imsize = (1080, 1080)
                cur_res_out_list = batch_res_out_list
                if house_out_dir is not None:
                    #cur_res_out_list = [out_path + '_samp%d' % (samp_idx) for out_path in batch_res_out_list]
                    cur_res_out_list = batch_res_out_list
                    imsize = (720, 720)
                human_verts, human_faces = viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm, cur_res_out_list,
                                imw=imsize[0],
                                imh=imsize[1],
                                show_smpl_joints=viz_smpl_joints,
                                show_pred_joints=viz_pred_joints,
                                show_contacts=viz_contacts
                              )
                human_verts = torch.from_numpy(human_verts).float().to(device).squeeze(0)

                valid_verts_list = []
                end_idx = 0                           # the index till where is the valid sequence
                for f_idx in range(eval_qual_samp_len):
                    scene_in_penetration = check_if_valid(human_verts[f_idx: f_idx + 1], scene)
                    wall_in_penetration = check_if_valid(human_verts[f_idx: f_idx + 1], wall)

                    print('num in penetration. scene: {}, wall: {}'.format(scene_in_penetration.sum(), wall_in_penetration.sum()))
                    if scene_in_penetration.sum() >= NUM_VERTS_THRESH or wall_in_penetration.sum() >= NUM_VERTS_THRESH:
                        end_idx = f_idx
                        break
                    else:
                        # check if person is still within scene bounds (need this due to missing doors / windows)
                        if is_person_in_scene(human_verts[f_idx: f_idx + 1].squeeze(0).cpu().numpy(), floor.vertices):
                            valid_verts_list.append(human_verts[f_idx])
                        else:
                            person_out_of_scene_flag = True
                            end_idx = f_idx
                            break

                    #if in_penetration.sum() >= NUM_VERTS_THRESH:
                    #    # if len(valid_verts_list) >= min_valid_seq_len:
                    #    end_idx = f_idx
                    #    break
                    #else:
                    #    valid_verts_list.append(human_verts[f_idx])
                    #valid_verts_list.append(human_verts[f_idx])
                
                if len(valid_verts_list) == eval_qual_samp_len:
                    end_idx = eval_qual_samp_len

                if save_seq_len:
                    seq_len = save_seq_len
                else:
                    seq_len = end_idx

                if len(valid_verts_list) >= min_seq_len:
                    # filter to have only forward motion
                    if not seq_is_forward_walking(x_pred_dict, end_idx, check_hor=30, check_thresh=0.3):
                        continue

                    # also check for the next ADDT_COL_HOR time steps. Collisions during that period are also counted
                    all_addt_penetration_labels = []
                    if end_idx != eval_qual_samp_len and not person_out_of_scene_flag:
                        for i in range(ADDT_COL_HOR):
                            if end_idx + 1 + i >= eval_qual_samp_len:
                                break
                            addt_scene_in_penetration = check_if_valid(human_verts[end_idx + 1 + i: end_idx + 2 + i], scene)
                            addt_wall_in_penetration = check_if_valid(human_verts[end_idx + 1 + i: end_idx + 2 + i], wall)
                            if addt_scene_in_penetration.sum() < NUM_VERTS_THRESH and addt_wall_in_penetration.sum() < NUM_VERTS_THRESH:
                                continue
                            elif addt_scene_in_penetration.sum() >= NUM_VERTS_THRESH and addt_wall_in_penetration.sum() < NUM_VERTS_THRESH:
                                addt_in_penetration = addt_scene_in_penetration
                            elif addt_scene_in_penetration.sum() < NUM_VERTS_THRESH and addt_wall_in_penetration.sum() >= NUM_VERTS_THRESH:
                                addt_in_penetration = addt_wall_in_penetration
                            else:
                                addt_in_penetration = np.logical_or(addt_scene_in_penetration, addt_wall_in_penetration)

                            addt_penetration_label = process_penetration(addt_in_penetration.squeeze(), human_verts[end_idx + 1 + i], x_pred_dict['joints'][0, end_idx + 1 + i], debug=False, counts_thresh=0)
                            all_addt_penetration_labels.append(addt_penetration_label.cpu().numpy())

                    # If longer than minimum sequence length, save motion sequence and collision supervision
                    #valid_verts_seq = torch.stack(valid_verts_list[-seq_len:]).to(device).squeeze(0)
                    #valid_verts_seq = torch.stack(valid_verts_list[:seq_len]).to(device).squeeze(0)
                    valid_verts_seq = torch.stack(valid_verts_list).to(device).squeeze(0)
                    os.makedirs(cur_res_out_list[0], exist_ok=False)
                    if write_obj:
                        for t_idx in range(seq_len):
                            dest_mesh_path = os.path.join(cur_res_out_list[0], "%05d"%(t_idx) + ".obj")
                            write_to_obj(dest_mesh_path, valid_verts_seq[t_idx].data.cpu().numpy(), human_faces.data.cpu().numpy())

                    #if end_idx == eval_qual_samp_len:
                    #    continue

                    if end_idx == eval_qual_samp_len:
                        # no collision happening in this sequence
                        penetration_label = torch.tensor([100], device=device) 
                    else:
                        if person_out_of_scene_flag:
                            print("person out of scene!")
                            penetration_label = torch.tensor([100], device=device)
                        else:
                            if scene_in_penetration.sum() >= NUM_VERTS_THRESH and wall_in_penetration.sum() < NUM_VERTS_THRESH:
                                in_penetration = scene_in_penetration
                            elif scene_in_penetration.sum() < NUM_VERTS_THRESH and wall_in_penetration.sum() >= NUM_VERTS_THRESH:
                                in_penetration = wall_in_penetration
                            else:
                                in_penetration = np.logical_or(scene_in_penetration, wall_in_penetration)

                            # save collision vertices (on human)
                            col_verts = human_verts[end_idx][in_penetration]
                            col_verts_path = os.path.join(cur_res_out_list[0], 'col_verts.npy')
                            np.save(col_verts_path, col_verts.cpu().numpy())

                            # save all motion info starting from collision step
                            dest_npz_path = os.path.join(cur_res_out_list[0], "motion_col_step.npz")

                            new_trans = x_pred_dict['trans'][0, end_idx:]
                            new_trans_vel= x_pred_dict['trans_vel'][0, end_idx:]
                            new_root_orient = x_pred_dict['root_orient'][0, end_idx:]
                            new_root_orient_vel = x_pred_dict['root_orient_vel'][0, end_idx:]
                            new_pose_body = x_pred_dict['pose_body'][0, end_idx:]
                            new_joints = x_pred_dict['joints'][0, end_idx:]
                            new_joints_vel = x_pred_dict['joints_vel'][0, end_idx:]

                            np.savez(dest_npz_path, fps=30,
                                gender=meta['gender'],
                                trans=new_trans.data.cpu().numpy(), # T X 3
                                root_orient=new_root_orient.data.cpu().numpy(), # T X 3 
                                pose_body=new_pose_body.data.cpu().numpy(), # T X 63
                                betas=meta['betas'][0, 0].data.cpu().numpy(), # 10
                                joints=new_joints.data.cpu().numpy(), # T X 22 X 3
                                joints_vel=new_joints_vel.data.cpu().numpy(), # T X 22 X 3
                                trans_vel=new_trans.data.cpu().numpy(), # T X 3
                                root_orient_vel=new_root_orient_vel.data.cpu().numpy()) # T X 3
        # joint_orient_vel_seq=joint_orient_vel_seq, # T: Based on joints_world2aligned_rot, calculate angular velocity for z-axis rotation. 
                            #if debug:
                            #    penetration_label, verts_in_penetration = process_penetration(in_penetration.squeeze(), human_verts[end_idx], x_pred_dict['joints'][0, end_idx], debug=True, counts_thresh=0)
                            #    verts_path = os.path.join(cur_res_out_list[0], 'verts_in_penetration.npy')
                            #    np.save(verts_path, verts_in_penetration.cpu().numpy())
                            #else:
                            penetration_label = process_penetration(in_penetration.squeeze(), human_verts[end_idx], x_pred_dict['joints'][0, end_idx], debug=False, counts_thresh=0)

                    #penetration_label = process_label(penetration_label, dataset_type)
                    penetration_label_path = os.path.join(cur_res_out_list[0], 'penetration_label.npy')
                    np.save(penetration_label_path, penetration_label.cpu().numpy())

                    if all_addt_penetration_labels:
                        addt_penetration_label_path = os.path.join(cur_res_out_list[0], 'addt_penetration_label.npy')
                        np.save(addt_penetration_label_path, np.array(all_addt_penetration_labels, dtype=object), allow_pickle=True)

                    dest_npz_path = os.path.join(cur_res_out_list[0], "motion_seq.npz")
                    cano_rot_inv = torch.eye(4).to(x_past.device)
                    gen_data_npz(x_pred_dict, meta, seq_len, end_idx, cano_rot_inv, dest_npz_path)

                    saved_batch_cnt += 1
                    if saved_batch_cnt >= num_batches:
                        break

                print("saved:", saved_batch_cnt)
                print("all:", batch_cnt)
                batch_cnt += 1
                if batch_cnt >= max_batch:
                    break

def translate_to_scene(x_pred_dict, floor, return_floor=False):
    """Translate the human location to a random location on the floor."""
    num_floor_vertices = floor.vertices.shape[0]

    rand_vertex_id = np.random.choice(num_floor_vertices)

    rand_vertex_x = floor.vertices[rand_vertex_id][0]
    rand_vertex_y = floor.vertices[rand_vertex_id][1]
    rand_vertex_z = floor.vertices[rand_vertex_id][2]

    x_pred_dict['trans'][..., :] += torch.Tensor([rand_vertex_x, rand_vertex_y, rand_vertex_z]).to(x_pred_dict['trans'].device)
    x_pred_dict['joints'] = x_pred_dict['joints'].view(x_pred_dict['joints'].shape[0], x_pred_dict['joints'].shape[1], -1, 3)
    x_pred_dict['joints'] += torch.Tensor([rand_vertex_x, rand_vertex_y, rand_vertex_z]).to(x_pred_dict['joints'].device)
   
    return x_pred_dict

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

def process_penetration(in_penetration, vertices, joints, counts_thresh=100, debug=False, dataset_type="map_10"):
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

def viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm, out_path_list,
                    imw=720,
                    imh=720,
                    show_pred_joints=False,
                    show_smpl_joints=False,
                    show_contacts=False):
    '''
    Given x_pred_dict from the model rollout and the ground truth dict, runs through SMPL model to visualize
    '''
    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS

    pred_world_root_orient = x_pred_dict['root_orient']
    B, T, _ = pred_world_root_orient.size()
    pred_world_root_orient = rotation_matrix_to_angle_axis(pred_world_root_orient.reshape((B*T, 3, 3))).reshape((B, T, 3))
    pred_world_pose_body = x_pred_dict['pose_body']
    pred_world_pose_body = rotation_matrix_to_angle_axis(pred_world_pose_body.reshape((B*T*(J-1), 3, 3))).reshape((B, T, (J-1)*3))
    pred_world_trans = x_pred_dict['trans']
    pred_world_joints = x_pred_dict['joints'].reshape((B, T, J, 3))

    viz_contacts = [None]*B
    if show_contacts and 'contacts' in x_pred_dict.keys():
        pred_contacts = torch.sigmoid(x_pred_dict['contacts'])
        pred_contacts = (pred_contacts > CONTACT_THRESH).to(torch.float)
        viz_contacts = torch.zeros((B, T, len(SMPL_JOINTS))).to(pred_contacts)
        viz_contacts[:,:,CONTACT_INDS] = pred_contacts
        pred_contacts = viz_contacts

    betas = meta['betas'].to(global_gt_dict[list(global_gt_dict.keys())[0]].device)
    human_verts_list = []
    human_faces = None
    for b in range(B):
        bm_world = male_bm if meta['gender'][b] == 'male' else female_bm
        # pred
        body_pred = bm_world(pose_body=pred_world_pose_body[b], 
                        pose_hand=None,
                        betas=betas[b,0].reshape((1, -1)).expand((T, 16)),
                        root_orient=pred_world_root_orient[b],
                        trans=pred_world_trans[b])

        pred_smpl_joints = body_pred.Jtr[:, :J]
        pred_smpl_verts = body_pred.v
        if human_faces is None:
            human_faces = body_pred.f
        human_verts_list.append(pred_smpl_verts.data.cpu().numpy())
    human_verts = np.asarray(human_verts_list)

    return human_verts, human_faces
        # viz_joints = None
        # if show_smpl_joints:
        #     viz_joints = pred_smpl_joints
        # elif show_pred_joints:
        #     viz_joints = pred_world_joints[b]

        # cur_offscreen = out_path_list[b] is not None
        # from viz.utils import viz_smpl_seq, create_video
        # body_alpha = 0.5 if viz_joints is not None and cur_offscreen else 1.0
        # viz_smpl_seq(body_pred,
        #                 imw=imw, imh=imh, fps=30,
        #                 render_body=True,
        #                 render_joints=viz_joints is not None,
        #                 render_skeleton=viz_joints is not None and cur_offscreen,
        #                 render_ground=True,
        #                 contacts=viz_contacts[b],
        #                 joints_seq=viz_joints,
        #                 body_alpha=body_alpha,
        #                 use_offscreen=cur_offscreen,
        #                 out_path=out_path_list[b],
        #                 wireframe=False,
        #                 RGBA=False,
        #                 follow_camera=True,
        #                 cam_offset=[0.0, 2.2, 0.9],
        #                 joint_color=[ 0.0, 1.0, 0.0 ],
        #                 point_color=[0.0, 0.0, 1.0],
        #                 skel_color=[0.5, 0.5, 0.5],
        #                 joint_rad=0.015,
        #                 point_rad=0.015
        #         )

        # if cur_offscreen:
        #     create_video(out_path_list[b] + '/frame_%08d.' + '%s' % ('png'), out_path_list[b] + '.mp4', 30)


def main(args, config_file):
    test(args, config_file)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]

    main(args, config_file)
