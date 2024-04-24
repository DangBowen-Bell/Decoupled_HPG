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
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
import smplx
import cv2

from misc import constants
from utils import misc_utils, other_utils
from models import posa_models

def transform_vertices_trans(vertices, trans):
    """_summary_

    Args:
        vertices (tensor): [B, N, 3]
        trans (tensor): [4, 4]
    """
    
    rotmat = trans[:3, :3].unsqueeze(0)
    transl = trans[:3, 3]
    
    new_vertices = torch.bmm(rotmat, vertices.permute(0, 2, 1)).permute(0, 2, 1)
    new_vertices = new_vertices + transl

    return new_vertices    


def read_sdf(vertices, sdf_grid, grid_dim, grid_min, grid_max, mode='bilinear', trans=None):
    assert vertices.dim() == 3
    assert sdf_grid.dim() == 4
    
    if trans is not None:
        vertices = transform_vertices_trans(vertices, trans)
    
    batch_size = vertices.shape[0]
    nv = vertices.shape[1]
    
    sdf_grid = sdf_grid.unsqueeze(0).permute(0, 4, 1, 2, 3)
    norm_vertices = (vertices - grid_min) / (grid_max - grid_min) * 2 - 1
    
    x = F.grid_sample(sdf_grid,
                      norm_vertices[:, :, [2, 1, 0]].view(batch_size, nv, 1, 1, 3),
                      padding_mode='border', mode=mode)
    x = x.permute(0, 2, 3, 4, 1)
    return x


def smpl_in_new_coords(torch_param, Rcw, Tcw, rotation_center, **kwargs):
    body_model = misc_utils.load_body_model(**kwargs).to(Rcw.device)
    if 'betas' in torch_param.keys():
        body_model.reset_params(betas=torch_param['betas'])
    
    P = body_model().joints.detach().squeeze()[0, :].reshape(-1, 3)
    global_orient_c = torch_param['global_orient']
    Tc = torch_param['transl']

    Rc = tgm.angle_axis_to_rotation_matrix(global_orient_c.reshape(-1, 3))[:, :3, :3]
    Rw = torch.matmul(Rcw, Rc)
    global_orient_w = cv2.Rodrigues(Rw.detach().cpu().squeeze().numpy())[0]
    torch_param['global_orient'] = torch.tensor(global_orient_w, dtype=global_orient_c.dtype,
                                                device=global_orient_c.device).reshape(1, 3)

    torch_param['transl'] = torch.matmul(Rcw, (P + Tc - rotation_center).t()).t() + Tcw - P
    return torch_param


def compute_recon_loss(gt_batch, pr_batch, contact_w, semantics_w, use_semantics, loss_type, reduction='mean', **kwargs):
    batch_size = gt_batch.shape[0]
    device = gt_batch.device
    dtype = gt_batch.dtype
    recon_loss_dist = torch.zeros(1, dtype=dtype, device=device)
    recon_loss_semantics = torch.zeros(1, dtype=dtype, device=device)
    semantics_recon_acc = torch.zeros(1, dtype=dtype, device=device)
    if loss_type == 'bce':
        recon_loss_dist = contact_w * F.binary_cross_entropy(pr_batch[:, :, 0], gt_batch[:, :, 0], reduction=reduction)
    elif loss_type == 'mse':
        recon_loss_dist = contact_w * F.mse_loss(pr_batch[:, :, 0], gt_batch[:, :, 0], reduction=reduction)
    if use_semantics:
        targets = gt_batch[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
        recon_loss_semantics = semantics_w * F.cross_entropy(pr_batch[:, :, 1:].permute(0, 2, 1), targets,
                                                             reduction=reduction)

        semantics_recon_acc = torch.mean((targets == torch.argmax(pr_batch[:, :, 1:], dim=-1)).float())

    return recon_loss_dist, recon_loss_semantics, semantics_recon_acc


def rotmat2transmat(x):
    x = np.append(x, np.array([0, 0, 0]).reshape(1, 3), axis=0)
    x = np.append(x, np.array([0, 0, 0, 1]).reshape(4, 1), axis=1)
    return x


def create_init_points(bbox, mesh_grid_step, pelvis_z_offset=0.0):
    x_offset = 0.75
    y_offset = 0.75
    X, Y, Z = np.meshgrid(np.arange(bbox[1, 0] + x_offset, bbox[0, 0] - x_offset, mesh_grid_step),
                          np.arange(bbox[1, 1] + y_offset, bbox[0, 1] - y_offset, mesh_grid_step),
                          np.arange(bbox[1, 2] + pelvis_z_offset, bbox[1, 2] + pelvis_z_offset + 0.5, mesh_grid_step))
    # print(np.unique(X))
    # print(np.unique(Y))
    # print(np.unique(Z))
    init_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    return init_points

def get_grid_points(min_bound, max_bound, step):
    X, Y, Z = np.meshgrid(np.arange(min_bound[0], max_bound[0], step[0]),
                          np.arange(min_bound[1], max_bound[1], step[1]),
                          np.arange(min_bound[2], max_bound[2], step[2]))
    grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    return grid_points

def create_init_points_with_control(scene_bbox, instances, 
                                    position, object_class, 
                                    pelvis_z_offset, step_interval):
    all_init_points = None
    all_object_idxs = []
    for i in range(len(instances)):
        if instances[i]['class'] == object_class:
            bbox_o3d = instances[i]['bbox']
            bbox = np.array([bbox_o3d.min_bound, bbox_o3d.max_bound])
            bbox_size = bbox[1] - bbox[0]
            scene_bbox_size = scene_bbox[0] - scene_bbox[1]
            steps = scene_bbox_size / step_interval
            
            if object_class in []:
                offset_x_in = 0
                offset_x_out = bbox_size[0] / 4
                offset_y_in = 0
                offset_y_out = bbox_size[1] / 4
                offset_z = bbox_size[2]
            elif object_class in ['floor', 'bed', 'chair', 'sofa', 'table']:
                offset_x_in = bbox_size[0] / 8
                offset_x_out = bbox_size[0] / 4
                offset_y_in = bbox_size[1] / 8
                offset_y_out = bbox_size[1] / 4
                offset_z = bbox_size[2] / 2
            
            if position == 'around':
                z_min = scene_bbox[1, 2] + pelvis_z_offset
                z_max = z_min + 0.5
                init_points_1 = get_grid_points([bbox[0, 0] - offset_x_out, bbox[0, 1] - offset_y_out, z_min],
                                                [bbox[0, 0] + offset_x_in,  bbox[1, 1] + offset_y_out, z_max],
                                                steps)
                init_points_2 = get_grid_points([bbox[1, 0] - offset_x_in,  bbox[0, 1] - offset_y_out, z_min],
                                                [bbox[1, 0] + offset_x_out, bbox[1, 1] + offset_y_out, z_max],
                                                steps)
                init_points_3 = get_grid_points([bbox[0, 0] + offset_x_in,  bbox[0, 1] - offset_y_out, z_min],
                                                [bbox[1, 0] - offset_x_in,  bbox[0, 1] + offset_y_in,  z_max],
                                                steps)
                init_points_4 = get_grid_points([bbox[0, 0] + offset_x_in,  bbox[1, 1] - offset_y_in,  z_min],
                                                [bbox[1, 0] - offset_x_in,  bbox[1, 1] + offset_y_out, z_max],
                                                steps)
                init_points = np.concatenate((init_points_1, init_points_2, init_points_3, init_points_4), axis=0)
            elif position == 'on':
                # z_min = bbox[0, 2] + offset_z
                # z_max = bbox[1, 2] + pelvis_z_offset + 0.5
                # offset_x_in = 0
                # offset_y_in = 0
                z_min = scene_bbox[1, 2] + pelvis_z_offset - 0.1
                z_max = scene_bbox[1, 2] + pelvis_z_offset + 0.5
                init_points = get_grid_points([bbox[0, 0] + offset_x_in, bbox[0, 1] + offset_y_in, z_min],
                                              [bbox[1, 0] - offset_x_in, bbox[1, 1] - offset_y_in, z_max],
                                              steps)
                
            if all_init_points is None:
                all_init_points = init_points
            else:
                all_init_points = np.concatenate((all_init_points, init_points), axis=0)
            
            all_object_idxs.extend([i] * len(init_points))
    
    all_object_idxs = np.array(all_object_idxs, dtype=np.uint8)
    
    return all_init_points, all_object_idxs


def load_body_model(model_folder, num_pca_comps=6, batch_size=1, gender='male', **kwargs):
    model_params = dict(model_path=model_folder,
                        model_type='smplx',
                        ext='npz',
                        num_pca_comps=num_pca_comps,
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        batch_size=batch_size)

    body_model = smplx.create(gender=gender, **model_params)
    return body_model


def load_model_checkpoint(device, model_name='POSA', load_checkpoint=None, use_cuda=False, checkpoints_dir=None,
                          checkpoint_path=None, **kwargs):
    model = posa_models.load_model(model_name, use_cuda=use_cuda, **kwargs)
    model.eval()
    if checkpoint_path is not None:
        print('loading {}'.format(checkpoint_path))
        if not use_cuda:
            checkpoint = torch.load(osp.join(checkpoint_path),
                                    map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif load_checkpoint > 0:
        print('loading stats of epoch {} from {}'.format(load_checkpoint, checkpoints_dir))
        if not use_cuda:
            checkpoint = torch.load(osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(load_checkpoint)),
                                    map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(osp.join(checkpoints_dir, 'epoch_{:04d}.pt'.format(load_checkpoint)))
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No checkpoint found')
        sys.exit(0)
    
    return model

def load_model_checkpoint_with_control(train_cfg_name, use_cuda=False, **kwargs):
    checkpoint_path = osp.join(constants.hpg_result_root, 'posa_training', train_cfg_name, 'checkpoints', 'epoch_0015.pt')
    model = posa_models.load_model('POSA_control', **kwargs)
    model.eval()
    print('loading {}'.format(checkpoint_path))
    if not use_cuda:
        checkpoint = torch.load(osp.join(checkpoint_path),
                                map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model   


def concat_resplit(x1, x2, ind1, ind2):
    if torch.is_tensor(x1):
        x = torch.cat((x1, x2))
    else:
        x = np.concatenate((x1, x2))
    x1 = x[ind1]
    x2 = x[ind2]
    return x1, x2


def rot_mat_to_euler(rot_mats):
    # order of rotation zyx
    R_y = -torch.asin(rot_mats[:, 2, 0])
    R_x = torch.atan2(rot_mats[:, 2, 1] / torch.cos(R_y), rot_mats[:, 2, 2] / torch.cos(R_y))
    R_z = torch.atan2(rot_mats[:, 1, 0] / torch.cos(R_y), rot_mats[:, 0, 0] / torch.cos(R_y))
    return R_z, R_y, R_x


def eval_physical_metric(vertices, scene_data, 
                         vol_den, contact_thre, action, 
                         vertices_ds=None, object_pc=None):
    device, dtype = other_utils.get_tensor_type()
    
    vertices = torch.tensor(vertices).to(dtype).to(device).unsqueeze(0)

    nv = float(vertices.shape[1])
    x = misc_utils.read_sdf(vertices, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
    if x.lt(0).sum().item() < 1:
        non_collision_score = torch.tensor(1)
        contact_score = torch.tensor(0.0)
    else:
        non_collision_score = (x > 0).sum().float() / nv
        contact_score = torch.tensor(1.0)
        
    volume_points = other_utils.vertices_to_volume_points(vertices, vol_den)    
    nvol = float(volume_points.shape[1])
    x_vol = misc_utils.read_sdf(volume_points, scene_data['sdf'],
                                scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                                mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
    if x_vol.lt(0).sum().item() < 1:
        vol_non_collision_score = torch.tensor(1)
    else:
        vol_non_collision_score = (x_vol > 0).sum().float() / nvol
    
    semantic_contact_score = torch.tensor(0.0)
    if vertices_ds is not None and object_pc is not None:
        vertices_ds = torch.tensor(vertices_ds).to(dtype).to(device).unsqueeze(0)
        object_pc = torch.tensor(object_pc).to(dtype).to(device).unsqueeze(0)
        dist, _, _, _ = other_utils.chamfer_dist(vertices_ds, object_pc)
        dist = dist.squeeze().detach().cpu().numpy()
        contact_prob = constants.contact_statistics['probability'][action + ' on']
        contact_score_thre = constants.contact_statistics['score'][action + ' on'] * 0.8
        semantic_contact_score = np.sum((dist < contact_thre) * contact_prob)
        semantic_contact_score = torch.tensor(1.0) if semantic_contact_score >= contact_score_thre else torch.tensor(0.0)
    
    contact_body_parts = constants.action_body_part_mapping[action]
    smplx_contact_ids = []
    for body_part in contact_body_parts:
        smplx_contact_ids += constants.smplx_contact_ids[body_part]
    smplx_contact_mask = np.zeros(int(nv))
    smplx_contact_mask[smplx_contact_ids] = 1
    smplx_contact_mask = torch.tensor(smplx_contact_mask).to(torch.uint8).to(device)
    semantic_accuracy_score = torch.tensor(1.0) if x[smplx_contact_mask].mean() < contact_thre else torch.tensor(0.0)
    
    eval_scores = {
        'non_collision_score': float(non_collision_score.detach().cpu().squeeze()),
        'vol_non_collision_score': float(vol_non_collision_score.detach().cpu().squeeze()),
        'contact_score': float(contact_score.detach().cpu().squeeze()),
        'semantic_contact_score': float(semantic_contact_score.detach().cpu().squeeze()),
        'semantic_accuracy_score': float(semantic_accuracy_score.detach().cpu().squeeze())
    }

    return eval_scores
