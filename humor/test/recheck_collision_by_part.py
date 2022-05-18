import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import json

import time

import trimesh

import pybullet_data
import pybullet as p

import torch

import numpy as np

from body_model.body_model import BodyModel
from utils import gen_obj_from_motion_seq

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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

with open('/orion/u/bxpan/exoskeleton/exoskeleton/data_utils/smpl_vert_segmentation.json','r') as fp:
    part_info = json.load(fp)

for scene in os.listdir(motion_root):
    if scene != "Albertville": continue
    scene_dir = os.path.join(motion_root, scene)
    if os.path.isdir(scene_dir) and scene != "pickles":
        time1 = time.time()
        scene_mesh_path = os.path.join(scene_root, scene, "watertight", "mesh_z_up.obj")
        scene_mesh = trimesh.exchange.load.load(scene_mesh_path)

        scaling  = [1, 1, 1]
        physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        
        collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=scene_mesh_path, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualId = -1
        
        boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
        p.changeDynamics(boundaryUid, -1, lateralFriction=1)
        p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                            specularColor=[0.5, 0.5, 0.5])

        time2 = time.time()
        print("scene:", time2 - time1)
        for motion in os.listdir(scene_dir):
            if motion != "BioMotionLab_NTroje_rub018_0029_jumping2_poses_64_frames_30_fps_b34243seq0": continue
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
            verts += floor_point
            person_out_of_scene_flag = False
            orig_seq_end_idx = new_trans.shape[0]
            seq_end_idx = orig_trans.shape[0]

            for i in range(verts.shape[0]):
                vertices = verts[i]
            
                pene = []
                time3 = time.time()
                for k,v in part_info.items():
                    partId = p.createCollisionShape(p.GEOM_MESH, vertices=vertices[v])
                    #partUid= p.createMultiBody(baseCollisionShapeIndex = partId, baseVisualShapeIndex = -1)

                    pts = p.getClosestPoints(bodyA=-1,
                                             bodyB=-1,
                                             distance=0,
                                             collisionShapeA=collisionId,
                                             collisionShapeB=partId)
                    pene += [k[6] for k in pts]
                    #pene += [k[6] for k in p.getClosestPoints(boundaryUid, partUid, distance=0)]

                    p.removeCollisionShape(partId)
                    #p.removeBody(partUid)
                    #p.removeBody(collisionId)
                #print(partUid, collisionId)
                # p.resetSimulation(physicsClient)
                # p.setGravity(0,0,-10)
                # 
                # collisionId = p.createCollisionShape(p.GEOM_MESH, fileName=scene_mesh_path, meshScale=scaling, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
                # visualId = -1
                # 
                # boundaryUid = p.createMultiBody(baseCollisionShapeIndex = collisionId, baseVisualShapeIndex = visualId)
                # p.changeDynamics(boundaryUid, -1, lateralFriction=1)
                # p.changeVisualShape(boundaryUid, -1, rgbaColor=[168 / 255.0, 164 / 255.0, 92 / 255.0, 1.0],
                #                     specularColor=[0.5, 0.5, 0.5])
                time4 = time.time()
                pene = filter_floor_cols(pene, floor_height + floor_buffer)
                if len(pene) >= 10:
                    seq_end_idx = i
                    break
                else:
                    if not is_person_in_scene(vertices, scene_mesh.vertices):
                        person_out_of_scene_flag = True
                        seq_end_idx = i
                        break
                time5 = time.time()
                #print("collision:", time4 - time3)
                #print("within bounds:", time5 - time4)

            #print("collision check:", time3 - time2)
            if seq_end_idx != orig_seq_end_idx:
                print("faulty collision detection at: {}. original end_idx {}; should be {}".format(motion_dir, orig_seq_end_idx, seq_end_idx))

            import pdb; pdb.set_trace()
        p.disconnect(physicsClient)
