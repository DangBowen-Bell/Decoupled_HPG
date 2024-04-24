# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import torch
import numpy as np
import torchgeometry as tgm

import pickle
import json
import pandas as pd

from lib import eulerangles
from misc import constants
from utils import misc_utils, other_utils


def batch2features(in_batch, use_semantics, **kwargs):
    in_batch = in_batch.squeeze(0)
    if torch.is_tensor(in_batch):
        in_batch = in_batch.detach().cpu().numpy()
    x = in_batch[:, 0]
    x_semantics = None
    if use_semantics:
        x_semantics = in_batch[:, 1:]
    return x, x_semantics


def features2batch(x, x_normals=None, x_semantics=None, use_semantics=False, use_sdf_normals=False, nv=None):
    in_batch = x.reshape(-1, nv, 1)
    batch_size = in_batch.shape[0]
    if use_sdf_normals:
        in_batch = torch.cat((in_batch, x_normals.reshape(-1, nv, 3)), dim=-1)
        if use_semantics:
            in_batch = torch.cat((in_batch, x_semantics.reshape(batch_size, nv, -1)), dim=-1)
    elif use_semantics:
        in_batch = torch.cat((in_batch, x_semantics.reshape(batch_size, nv, -1)), dim=-1)
    return in_batch


def compute_canonical_transform(global_orient):
    device = global_orient.device
    dtype = global_orient.dtype
    R = tgm.angle_axis_to_rotation_matrix(global_orient)  # [:, :3, :3].detach().cpu().numpy().squeeze()
    R_inv = R[:, :3, :3].reshape(3, 3).t()
    x, z, y = eulerangles.mat2euler(R[:, :3, :3].detach().cpu().numpy().squeeze(), 'sxzy')
    y = 0
    z = 0
    R_new = torch.tensor(eulerangles.euler2mat(x, z, y, 'sxzy'), dtype=dtype, device=device)
    return torch.matmul(R_new, R_inv)


def pkl_to_canonical(pkl_file_path, device, dtype, batch_size, 
                     gender='male', model_folder=None, vertices_clothed=None, 
                     **kwargs):
    R_can = torch.tensor(eulerangles.euler2mat(np.pi, np.pi, 0, 'syzx'), dtype=dtype, device=device)
    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi/2, 0, 0, 'sxyz'), dtype=dtype, device=device)
    
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f)
    num_pca_comps = 6
    body_model = misc_utils.load_body_model(model_folder, num_pca_comps, batch_size, param['gender']).to(device)
    body_param_list = [name for name, _ in body_model.named_parameters()]

    torch_param = {}
    for key in param.keys():
        if key in body_param_list:
            torch_param[key] = torch.tensor(param[key], dtype=torch.float32).to(device)
    print(torch_param.keys())
    torch_param['betas'] = torch_param['betas'][:, :10]
    torch_param['left_hand_pose'] = torch_param['left_hand_pose'][:, :num_pca_comps]
    torch_param['right_hand_pose'] = torch_param['right_hand_pose'][:, :num_pca_comps]

    faces_arr = body_model.faces
    body_model.reset_params(**torch_param)
    body_model_output = body_model(return_verts=True)
    pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)

    vertices = body_model_output.vertices.squeeze()
    # other_utils.viz_pc((vertices - pelvis).detach().cpu().numpy())
    vertices_can = torch.matmul(R_can, (vertices - pelvis).t()).t()
    # other_utils.viz_pc(vertices_can.detach().cpu().numpy())
    vertices = torch.matmul(R_smpl2scene, (vertices - pelvis).t()).t()
    # other_utils.viz_pc(vertices.detach().cpu().numpy())

    if vertices_clothed is not None:
        vertices_clothed /= 100
        vertices_clothed -= np.mean(vertices_clothed, axis=0)
        vertices_clothed = torch.tensor(vertices_clothed, dtype=dtype, device=device)
        vertices_clothed = torch.matmul(R_smpl2scene, (vertices_clothed - pelvis).t()).t()

    return vertices, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, vertices_clothed

def pkl_to_canonical_with_control(pkl_file_path, device, dtype, batch_size, action, 
                                  gender='neutral', model_folder=None, vertices_clothed=None, 
                                  **kwargs):
    R_can = torch.tensor(eulerangles.euler2mat(np.pi, np.pi, 0, 'syzx'), dtype=dtype, device=device)
    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi/2, 0, 0, 'sxyz'), dtype=dtype, device=device)
    
    with open(pkl_file_path, 'rb') as f:
        param = pickle.load(f)
    num_pca_comps = 6
    body_model = misc_utils.load_body_model(model_folder, num_pca_comps, batch_size, gender).to(device)
    body_param_list = [name for name, _ in body_model.named_parameters()]

    torch_param = {}
    for key in param.keys():
        if key in body_param_list:
            torch_param[key] = torch.tensor(param[key], dtype=torch.float32).to(device)

    #* set global orient manually (if no output)
    if 'global_orient' not in torch_param.keys():
        torch_param['global_orient'] = torch.zeros(3, dtype=torch.float32).to(device)
        if action == 'sit':
            torch_param['global_orient'][0] = -np.pi/4
        elif action == 'lie':
            torch_param['global_orient'][0] = -np.pi/2
    
    torch_param['transl'] = torch.zeros(3, dtype=torch.float32).to(device)
    
    faces_arr = body_model.faces
    body_model.reset_params(**torch_param)
    body_model_output = body_model(return_verts=True)
    pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)

    vertices = body_model_output.vertices.squeeze()
    # vertices_can = vertices
    # other_utils.viz_pc((vertices - pelvis).detach().cpu().numpy())
    vertices_can = torch.matmul(R_can, (vertices - pelvis).t()).t()
    # other_utils.viz_pc(vertices_can.detach().cpu().numpy())
    vertices = torch.matmul(R_smpl2scene, (vertices - pelvis).t()).t()
    # other_utils.viz_pc(vertices.detach().cpu().numpy())
    
    if vertices_clothed is not None:
        vertices_clothed /= 100
        vertices_clothed -= np.mean(vertices_clothed, axis=0)
        vertices_clothed = torch.tensor(vertices_clothed, dtype=dtype, device=device)
        vertices_clothed = torch.matmul(R_smpl2scene, (vertices_clothed - pelvis).t()).t()

    return vertices, vertices_can, faces_arr, body_model, R_can, pelvis, torch_param, vertices_clothed


# no longer used
def load_scene_data(scene_name, device, use_semantics, no_obj_classes, **kwargs):
    R = torch.tensor(eulerangles.euler2mat(np.pi/2, 0, 0, 'sxyz'), dtype=dtype, device=device)
    t = torch.zeros(1, 3, dtype=dtype, device=device)

    sdf_dir = osp.join(constants.posa_root, 'sdf')
    with open(osp.join(sdf_dir, scene_name + '.json'), 'r') as f:
        sdf_data = json.load(f)
        grid_dim = sdf_data['dim']
        badding_val = sdf_data['badding_val']
        grid_min = torch.tensor(np.array(sdf_data['min']), dtype=dtype, device=device)
        grid_max = torch.tensor(np.array(sdf_data['max']), dtype=dtype, device=device)
        voxel_size = (grid_max - grid_min) / grid_dim
        bbox = torch.tensor(np.array(sdf_data['bbox']), dtype=dtype, device=device)

    sdf = np.load(osp.join(sdf_dir, scene_name + '_sdf.npy')).astype(np.float32)
    sdf = sdf.reshape(grid_dim, grid_dim, grid_dim, 1)
    sdf = torch.tensor(sdf, dtype=dtype, device=device)

    semantics = scene_semantics = None
    if use_semantics:
        semantics = np.load(osp.join(sdf_dir, scene_name + '_semantics.npy')).astype(np.float32).reshape(grid_dim, grid_dim, grid_dim, 1)
        # Map `seating=34` to `Sofa=10`. `Seating is present in `N0SittingBooth only`
        semantics[semantics == 34] = 10
        # Map falsly labelled`Shower=34` to `lightings=28`.
        semantics[semantics == 25] = 28
        scene_semantics = torch.tensor(np.unique(semantics), dtype=torch.long, device=device)
        scene_semantics = torch.zeros(1, no_obj_classes, dtype=dtype, device=device).scatter_(-1, scene_semantics.reshape(1, -1), 1)
        semantics = torch.tensor(semantics, dtype=dtype, device=device)
    
    scene_data = {
        'R': R, 
        't': t, 
        'grid_dim': grid_dim, 
        'grid_min': grid_min,
        'grid_max': grid_max, 
        'voxel_size': voxel_size,
        'bbox': bbox, 
        'badding_val': badding_val,
        'sdf': sdf, 
        'semantics': semantics, 
        'scene_semantics': scene_semantics,
        'trans_mesh2sdf': None,
    }

    return scene_data


def load_data(data_dir=None, train_data=True, contact_threshold=0.05, use_semantics=False, **kwargs):
    if train_data:
        data_dir = osp.join(data_dir, 'train')
    else:
        data_dir = osp.join(data_dir, 'test')
    x = torch.tensor(np.load(osp.join(data_dir, 'x.npy')), dtype=torch.float)
    x = (x < contact_threshold).type(torch.float32)
    with open(osp.join(data_dir, 'recording_names.json'), 'r') as f:
        recording_names = json.load(f)
    with open(osp.join(data_dir, 'pkl_file_paths.json'), 'r') as f:
        pkl_file_paths = json.load(f)

    joints_can = torch.tensor(np.load(osp.join(data_dir, 'joints_can.npy')), dtype=torch.float)
    x_semantics = None
    vertices = torch.tensor(np.load(osp.join(data_dir, 'vertices.npy')), dtype=torch.float)
    vertices_can = torch.tensor(np.load(osp.join(data_dir, 'vertices_can.npy')), dtype=torch.float)

    if use_semantics:
        x_semantics = torch.tensor(np.load(osp.join(data_dir, 'x_semantics.npy')), dtype=torch.float)
        x_semantics = x * x_semantics
    
    # print('x: ', x.shape)                               # [64304, 655]
    # print('joints_can: ', joints_can.shape)             # [64304, 228]
    # print('vertices: ', vertices.shape)                 # [64304, 655, 3]
    # print('vertices_can: ', vertices_can.shape)         # [64304, 655, 3]
    # print('x_semantics: ', x_semantics.shape)           # [64304, 655]
    # print('recording_names: ', len(recording_names))    # [64304]
    # print('pkl_file_paths: ', len(pkl_file_paths))      # [64304]

    return x, joints_can, vertices, vertices_can, x_semantics, recording_names, pkl_file_paths

def load_data_with_control(data_dir=None, train_data=True, contact_threshold=0.05, use_semantics=False, **kwargs):
    if train_data:
        data_dir = osp.join(data_dir, 'train')
        proxs_path = osp.join(constants.proxs_root, 'interaction_semantics', 'train.pkl')
    else:
        data_dir = osp.join(data_dir, 'test')
        proxs_path = osp.join(constants.proxs_root, 'interaction_semantics', 'test.pkl')
        
    x = torch.tensor(np.load(osp.join(data_dir, 'x.npy')), dtype=torch.float)
    
    if kwargs['loss_type'] == 'mse':
        x = (contact_threshold - x) / contact_threshold
        x = torch.clamp(x, 0, 1)
    else:
        x = (x < contact_threshold).type(torch.float32)
    
    with open(osp.join(data_dir, 'recording_names.json'), 'r') as f:
        recording_names = json.load(f)
    with open(osp.join(data_dir, 'pkl_file_paths.json'), 'r') as f:
        pkl_file_paths = json.load(f)

    joints_can = torch.tensor(np.load(osp.join(data_dir, 'joints_can.npy')), dtype=torch.float)
    x_semantics = None
    vertices = torch.tensor(np.load(osp.join(data_dir, 'vertices.npy')), dtype=torch.float)
    vertices_can = torch.tensor(np.load(osp.join(data_dir, 'vertices_can.npy')), dtype=torch.float)

    if use_semantics:
        x_semantics = torch.tensor(np.load(osp.join(data_dir, 'x_semantics.npy')), dtype=torch.float)
        x_semantics = x * x_semantics
        
    with open(proxs_path, 'rb') as f:
        proxs_data = pickle.load(f)
    proxs_seq_frames = [sample['sequence'] + '-' + sample['frame_name'] for sample in proxs_data]
    posa_seq_frames = [recording_names[i] + '-' + pkl_file_paths[i] for i in range(len(recording_names))]
    
    category_dict, category_list = other_utils.load_mpcat40_data()
    
    #* select frames with semantic annotations
    select_idx = []
    objects = []
    new_recording_names = []
    new_pkl_file_paths = []
    for i, seq_frame in enumerate(posa_seq_frames):
        try:
            j = proxs_seq_frames.index(seq_frame)
            #* treat all the actions (stand, sit, lie, touch) the same
            object_idxs = [category_list.index(label.split('-')[1]) for label in proxs_data[j]['interaction_labels']]
            object_code = np.zeros(len(category_list))
            object_code[object_idxs] = 1
            select_idx.append(i)
            objects.append(object_code)
            new_recording_names.append(recording_names[i])
            new_pkl_file_paths.append(pkl_file_paths[i])
        except:
            pass
    
    x = x[select_idx]
    joints_can = joints_can[select_idx]
    vertices = vertices[select_idx]
    vertices_can = vertices_can[select_idx]
    x_semantics = x_semantics[select_idx]
    recording_names = new_recording_names
    pkl_file_paths = new_pkl_file_paths
    objects = torch.tensor(objects, dtype=torch.float)
    
    # print('x: ', x.shape)                               # [25294, 655]
    # print('joints_can: ', joints_can.shape)             # [25294, 228]
    # print('vertices: ', vertices.shape)                 # [25294, 655, 3]
    # print('vertices_can: ', vertices_can.shape)         # [25294, 655, 3]
    # print('x_semantics: ', x_semantics.shape)           # [25294, 655]
    # print('recording_names: ', len(recording_names))    # [25294]
    # print('pkl_file_paths: ', len(pkl_file_paths))      # [25294]
    # print('objects: ', objects.shape)                   # [25294, 42]
        
    return x, joints_can, vertices, vertices_can, x_semantics, recording_names, pkl_file_paths, objects
