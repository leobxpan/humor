import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from collections import OrderedDict

import json

import time

import trimesh

import pybullet_data
import pybullet as p

import torch

import numpy as np

from body_model.body_model import BodyModel
from geom_utils import gen_obj_from_motion_seq, write_to_obj

def is_person_in_scene(human_verts, scene_verts):
    scene_2d = scene_verts[:, :2]
    human_2d = human_verts[:, :2]

    scene_min = scene_2d.min(axis=0)
    scene_max = scene_2d.max(axis=0)

    if np.all(human_2d >= scene_min) and np.all(human_2d <= scene_max):
        return True
    else:
        return False

def filter_floor_cols(pene, max_floor_height):
    return [vertex for vertex in pene if vertex[2] > max_floor_height]

scene_root = "/orion/group/Mp3d_Gibson_scenes"
motion_root = "/orion/group/Mp3d_Gibson_motions"
SMPLH_PATH = "/orion/u/bxpan/exoskeleton/humor/body_models/smplh"
MALE_BM_PATH = os.path.join(SMPLH_PATH, "male/model.npz")
FEMALE_BM_PATH = os.path.join(SMPLH_PATH, "female/model.npz")

floor_buffer = 0.1

device = torch.device("cpu")

with open('/orion/u/bxpan/exoskeleton/exoskeleton/data_utils/smpl_vert_segmentation.json','r') as fp:
    part_info = OrderedDict(json.load(fp))
num_verts = np.zeros((len(part_info.keys()), 6890)) - 1
for i, (k, v) in enumerate(part_info.items()):
    temp_arr = np.zeros(6890) - 1
    temp_arr[v] = np.arange(len(v)) + 1
    num_verts[i] = temp_arr
num_verts = num_verts.astype(int)

for scene in os.listdir(motion_root):
    #if scene != "Albertville": continue
    #if scene != "Uxmj2M2itWa": continue
    #if scene != "Convoy": continue
    #if scene != "Samuels": continue
    scene_dir = os.path.join(motion_root, scene)
    if os.path.isdir(scene_dir) and scene != "pickles":
        scene_mesh_path = os.path.join(scene_root, scene, "mesh_z_up.obj")
        scene_mesh = trimesh.exchange.load.load(scene_mesh_path)

        scaling  = [1, 1, 1]
        physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        
        collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=scene_mesh_path, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualId = -1
        
        for motion in os.listdir(scene_dir):
            #if motion != "BioMotionLab_NTroje_rub018_0029_jumping2_poses_64_frames_30_fps_b34243seq0": continue
            #if motion != "KIT_317_turn_right02_poses_130_frames_30_fps_b31318seq0": continue
            #if motion != "CMU_79_79_47_poses_492_frames_30_fps_b21554seq0": continue
            #if motion != "CMU_136_136_09_poses_225_frames_30_fps_b10766seq0": continue
            motion_dir = os.path.join(scene_dir, motion)

            motion_seq = np.load(os.path.join(motion_dir, "motion_seq.npz"))
            new_trans = motion_seq['trans']

            orig_motion_name = motion + "_motion_seq.npz"
            orig_motion_seq = np.load(os.path.join("/orion/group/Exoskeleton_humor_motions", orig_motion_name))
            orig_trans = orig_motion_seq['trans']

            floor_point = new_trans[0] - orig_trans[0]
            translated_trans = orig_trans + floor_point
            translated_joints = orig_motion_seq['joints'] + floor_point
            floor_height = floor_point[2]

            male_bm = BodyModel(bm_path=MALE_BM_PATH, num_betas=16, batch_size=orig_motion_seq['trans'].shape[0]).to(device)
            female_bm = BodyModel(bm_path=FEMALE_BM_PATH, num_betas=16, batch_size=orig_motion_seq['trans'].shape[0]).to(device)
            bm_world = male_bm if orig_motion_seq['gender'][0] == 'male' else female_bm
            verts, faces = gen_obj_from_motion_seq(orig_motion_seq["root_orient"], orig_motion_seq["pose_body"], orig_motion_seq["trans"], orig_motion_seq["betas"], bm_world, device)
            faces = faces.cpu().numpy()

            faces_str = {}
            for idx,(k,v) in enumerate(part_info.items()):
                part_str = ""
                for abc in faces:
                    verts_f = num_verts[idx][abc]
                    if -1 in verts_f: continue
                    part_str += f"f {verts_f[0]} {verts_f[1]} {verts_f[2]}\n"
                faces_str[k] = part_str
            verts += floor_point
            orig_seq_end_idx = new_trans.shape[0]
            seq_end_idx = orig_trans.shape[0]

            for i in range(verts.shape[0]):
                vertices = verts[i]

                pene = []
                for idx,(k,v) in enumerate(part_info.items()):

                    verts_str = ""
                    for x, y, z in vertices[v]:
                        verts_str += f"v {x: .3f} {y: .3f} {z: .3f}\n"

                    verts_str += faces_str[k]
                    with open(f"/tmp/temp_obj_%s_%s_%d_%d.obj"%(scene, motion, i, idx), "w") as fp:
                        fp.write(verts_str)

                    #partId = p.createCollisionShape(p.GEOM_MESH, vertices=vertices[v])
                    partId = p.createCollisionShape(p.GEOM_MESH, fileName=f"/tmp/temp_obj_%s_%s_%d_%d.obj"%(scene, motion, i, idx))

                    pts = p.getClosestPoints(bodyA=-1,
                                         bodyB=-1,
                                         distance=0,
                                         collisionShapeA=collisionId,
                                         collisionShapeB=partId)
                    pene += [k[5] for k in pts]

                    p.removeCollisionShape(partId)
                    os.remove(f"/tmp/temp_obj_%s_%s_%d_%d.obj"%(scene, motion, i, idx))

                pene = filter_floor_cols(pene, floor_height + floor_buffer)
                if len(pene) > 5:
                    seq_end_idx = i
                    break
                else:
                    if not is_person_in_scene(vertices, scene_mesh.vertices):
                        seq_end_idx = i
                        break

            if np.abs(seq_end_idx - orig_seq_end_idx) >= 10:
                with open("/orion/u/bxpan/exoskeleton/problematic_collision_motions.txt", 'a') as f:
                    f.write("faulty collision detection at: {}. original end_idx {}; should be {}\n".format(motion_dir, orig_seq_end_idx, seq_end_idx))
                
                # print("faulty collision detection at: {}. original end_idx {}; should be {}".format(motion_dir, orig_seq_end_idx, seq_end_idx))
                # write_to_obj(f"test%d.obj"%(seq_end_idx), vertices, faces)
                # pene = trimesh.Trimesh(vertices=pene)
                # pene.export("pene_scene.obj")

        p.disconnect(physicsClient)
