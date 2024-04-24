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

import torch
import torch.nn.functional as F
import numpy as np
import torchgeometry as tgm
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import torch.nn as nn
import open3d as o3d
import os.path as osp

from lib import eulerangles
from misc import constants
from utils import misc_utils, other_utils 

# no longer used
def get_contact_object_pc(init_pos, _object, instances_categories, bbox_o3ds, instance_o3ds, 
                          device=None, dtype=None, **kwargs):
    min_dist = 999.0
    object_idx = -1
    for j in range(len(instances_categories)):
        if instances_categories[j] == _object:
            bb_center = (bbox_o3ds[j].min_bound + bbox_o3ds[j].max_bound) / 2
            dist = np.linalg.norm(init_pos - bb_center)
            if dist < min_dist:
                object_idx = j
    object_pc = instance_o3ds[object_idx].sample_points_poisson_disk(constants.object_points_num[_object])
    object_pc = np.asarray(object_pc.points)
    # other_utils.viz_pc(object_pc)
    object_pc = torch.tensor(object_pc, dtype=dtype, device=device).unsqueeze(0)
    return object_pc


def transform_vertices(vertices, pos, ang):
    """_summary_

    Args:
        vertices (tensor): [B, N, 3]
        pos (tensor): [3]
        ang (tensor): [1]
    """
    
    rot_aa = torch.cat((torch.zeros((1, 2), device=vertices.device), ang.reshape(1, 1)), dim=1)
    rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3]
    
    new_vertices = torch.bmm(rot_mat, vertices.permute(0, 2, 1)).permute(0, 2, 1)
    new_vertices = new_vertices + pos
    
    return new_vertices


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist
    
contact_robustifier = GMoF(rho=0.05)    


def compute_afford_loss(for_eval=False, vertices_ds=None, 
                        scene_data=None, gen_batch=None, 
                        no_obj_classes=42, use_semantic_loss=True, 
                        use_contact_prox=False, object_pc=None, 
                        object_idx=-1, use_vol_pene=False, 
                        vertices_org=None, use_contact_prob=False, 
                        use_mask_for_pene=True, **kwargs):
    batch_size = vertices_ds.shape[0]
    device = vertices_ds.device
    
    #* baseline:
    #   - contact loss + penetration loss
    #       - contact loss: fixted mask
    #       - penetration loss: all
    #* posa / cposa
    #   - contact loss + penetration loss + semantic loss
    #       - contact loss: generated mask
    #       - penetration loss: all
    #* ours
    #   - contact loss + penetration loss 
    #       - contact loss: generated mask, weighted
    #       - penetration loss: all
   
    if for_eval:
        contact_w = kwargs['contact_w_eval']
        semantics_w = kwargs['semantics_w_eval']
        pene_w = kwargs['pene_w_eval']
        contact_prox_w = kwargs['contact_prox_w_eval']
        vol_pene_w = kwargs['vol_pene_w_eval']
    else:
        contact_w = kwargs['contact_w']
        semantics_w = kwargs['semantics_w']
        pene_w = kwargs['pene_w']
        contact_prox_w = kwargs['contact_prox_w']
        vol_pene_w = kwargs['vol_pene_w']
    
    x = misc_utils.read_sdf(vertices_ds, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
    
    contact_mask = (gen_batch[:, :, 0] > 0.5).flatten()     # [655]
    contact_prob = gen_batch[:, contact_mask, 0].squeeze(0) # [N]
    
    #* contact loss: ds vertices + grid
    contact_dist = x[contact_mask] ** 2
    if use_contact_prob:
        contact_dist = contact_prob * contact_dist
    contact_loss = contact_w * torch.sum(contact_dist)
    
    #* penetration loss: ds vertices + no contact 
    pene_loss = torch.tensor(0.0)
    if use_mask_for_pene:
        pene_mask = x.lt(0).flatten().int() + (~contact_mask).int()
        x_neg = torch.abs(x[pene_mask == 2])
    else:
        pene_mask = x.lt(0).flatten().int()
        x_neg = torch.abs(x[pene_mask == 1])
    if len(x_neg) > 0:
        pene_loss = pene_w * x_neg.sum()

    #* semantic loss: ds vertices
    semantics_loss = torch.tensor(0.0)
    if use_semantic_loss:
        # print('[DEBUG]: using semantic loss')
        x_semantics = misc_utils.read_sdf(vertices_ds, scene_data['semantics'], 
                                          scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'], 
                                          mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
        x_semantics = contact_mask.float() * x_semantics.unsqueeze(0)
        x_semantics = torch.zeros(x_semantics.shape[0], x_semantics.shape[1], no_obj_classes, device=device).\
            scatter_(-1, x_semantics.unsqueeze(-1).type(torch.long), 1.)
        targets = gen_batch[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)        
        semantics_loss = semantics_w * F.cross_entropy(x_semantics.permute(0, 2, 1), targets, reduction='sum')

    #* contact loss (prox): ds vertices + cd
    contact_prox_loss = torch.tensor(0.0)
    if use_contact_prox:
        # print('[DEBUG]: using contact loss (prox)')
        #* all contact vertices
        contact_vertices = vertices_ds[:, contact_mask, :]
        
        #* only vertices contact with object
        # targets = gen_batch[:, :, 1:].argmax(dim=-1).type(torch.long).reshape(batch_size, -1)
        # contact_vertices = vertices_ds[targets==object_idx].unsqueeze(0)
        
        dist, _, _, _ = other_utils.chamfer_dist(contact_vertices, object_pc)
        dist = contact_robustifier(dist.sqrt())
        if use_contact_prob:
            dist = contact_prob.unsqueeze(0) * dist
        contact_prox_loss = contact_prox_w * dist.sum()

    #* volume penetration loss: org vertices
    vol_pene_loss = torch.tensor(0.0)
    if use_vol_pene:
        print('[DEBUG]: using volume penetration loss')
        volume_points = other_utils.vertices_to_volume_points(vertices_org, kwargs['vol_den'])
        x_vol = misc_utils.read_sdf(volume_points, scene_data['sdf'],
                                    scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                                    mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
        pene_mask = x.lt(0).flatten().int()
        x_vol_neg = torch.abs(x[pene_mask == 1])
        if len(x_vol_neg) > 0:
            vol_pene_loss = vol_pene_w * x_vol_neg.sum()
    
    # no weight
    # print('contact: ', contact_loss / contact_w)
    # print('pene: ', pene_loss / pene_w)
    # print('semantics: ', semantics_loss / semantics_w)
    # print('contact (prox): ', contact_prox_loss / contact_prox_w)
    # print('volume pene: ', vol_pene_loss / vol_pene_w)
    
    # with weight
    # print('contact: ', contact_loss)
    # print('pene: ', pene_loss)
    # print('semantics: ', semantics_loss)
    # print('contact (prox): ', contact_prox_loss)
    # print('volume pene: ', vol_pene_loss)
    
    afford_loss = contact_loss + pene_loss + semantics_loss + contact_prox_loss + vol_pene_loss

    return afford_loss


def eval_init_points(init_pos=None, init_ang=None, 
                     vertices_ds=None, scene_data=None, 
                     gen_batch=None, **kwargs):
    with torch.no_grad():
        losses = []
        init_pos_batches = init_pos.split(1)

        for i in range(len(init_pos_batches)):
            curr_init_pos = init_pos_batches[i]
            rot_aa = torch.cat((torch.zeros((1, 2), device=vertices_ds.device), init_ang[i].reshape(1, 1)), dim=1)
            rot_mat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3]
            curr_vertices = torch.bmm(rot_mat, vertices_ds.permute(0, 2, 1)).permute(0, 2, 1)
            curr_vertices = curr_vertices + curr_init_pos
            loss = compute_afford_loss(for_eval=True, 
                                       vertices_ds=curr_vertices, 
                                       scene_data=scene_data, 
                                       gen_batch=gen_batch, **kwargs)
            losses.append(loss.item())

        # sort initial positions and orientations from best to wrost
        losses = np.array(losses)
        ids = np.argsort(losses)
        losses = losses[ids]
        init_pos = init_pos[ids]
        init_ang = init_ang[ids]

        return losses, init_pos, init_ang

def init_points_culling(init_pos=None, vertices_ds=None, 
                        scene_data=None, gen_batch=None, 
                        max_init_points=8, angle_step=2, 
                        instance_idxs=None, **kwargs):
    init_ang = []
    angles = torch.arange(0, 2*np.pi, np.pi/angle_step, device=vertices_ds.device)
    angles[0] = 1e-9
    for ang in angles:
        init_ang.append(ang * torch.ones(init_pos.shape[0], 1, device=vertices_ds.device))
    init_ang = torch.cat(init_ang).to(init_pos.device)
    init_pos = init_pos.repeat(angles.shape[0], 1, 1)
    instance_idxs = np.tile(instance_idxs, angles.shape[0])

    rnd_ids = np.random.choice(init_pos.shape[0], init_pos.shape[0], replace=False)
    init_pos = init_pos[rnd_ids, :]
    init_ang = init_ang[rnd_ids, :]
    instance_idxs = instance_idxs[rnd_ids]

    losses, init_pos, init_ang = eval_init_points(init_pos=init_pos, init_ang=init_ang,
                                                  vertices_ds=vertices_ds.unsqueeze(0),
                                                  scene_data=scene_data, gen_batch=gen_batch, **kwargs)
    # select only a subset from initial points for optimization
    if init_pos.shape[0] > max_init_points:
        init_pos = init_pos[:max_init_points]
        init_ang = init_ang[:max_init_points]
        instance_idxs = instance_idxs[:max_init_points]

    return init_pos, init_ang, instance_idxs


def has_severe_penetration(vertices, faces, scene_data, **kwargs):    
    vertices_sdf = misc_utils.read_sdf(vertices, scene_data['sdf'],
                                       scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                                       mode="bilinear", trans=scene_data['trans_mesh2sdf']).squeeze()
    
    new_faces = []
    for f in faces:
        v1, v2, v3 = f
        if (vertices_sdf[v1] * vertices_sdf[v2] < 0) or \
           (vertices_sdf[v2] * vertices_sdf[v3] < 0) or \
           (vertices_sdf[v1] * vertices_sdf[v3] < 0):
            pass
        else:
            new_faces.append(f)

    faces_mat = np.eye(len(vertices_sdf))
    for f in new_faces:
        v1, v2, v3 = f
        faces_mat[v1][v2] = 1
        faces_mat[v2][v1] = 1
        faces_mat[v2][v3] = 1
        faces_mat[v3][v2] = 1        
        faces_mat[v1][v3] = 1
        faces_mat[v3][v1] = 1
    
    csr_mat = csr_matrix(faces_mat)
    n_components, labels = connected_components(csgraph=csr_mat, directed=False, return_labels=True)
    
    vertices_np = vertices.squeeze(0).detach().cpu().numpy()
    random_colors = other_utils.get_random_colors(n_components)
    
    pos_num = 0
    neg_num = 0
    body_o3ds = []
    for i in range(n_components):
        i_idxs = np.where(labels==i)[0]
        if len(i_idxs) > (len(labels) // 20):
            body_o3d = o3d.geometry.PointCloud()
            body_o3d.points = o3d.utility.Vector3dVector(vertices_np[i_idxs])
            body_o3d.paint_uniform_color(random_colors[i])
            body_o3ds.append(body_o3d)
            if vertices_sdf[i_idxs[0]] > 0:
                pos_num += 1
            else:
                neg_num += 1
    
    flag = (pos_num > 1)
                
    if kwargs['show_bad_pos_pene'] and flag:
        print('-' * 20)
        print('pos: {}, neg: {}'.format(pos_num, neg_num))
        print('-' * 20)
        axis_o3d = o3d.geometry.TriangleMesh().create_coordinate_frame()
        scene_o3d = o3d.io.read_triangle_mesh(osp.join(constants.posa_root, 'scenes', kwargs['scene_name'] + '.ply'))
        viz_data = [scene_o3d, axis_o3d] + body_o3ds
        o3d.visualization.draw_geometries(viz_data)
            
    return flag

def has_contact(vertices_ds, gen_batch, scene_data, **kwargs):
    x = misc_utils.read_sdf(vertices_ds, scene_data['sdf'],
                            scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                            mode='bilinear', trans=scene_data['trans_mesh2sdf']).squeeze()
    contact_mask = (gen_batch[:, :, 0] > 0.5).flatten()
    neg_mask = (x > -kwargs['contact_threshold']) & (x < 0)
    neg_contact_mask = (contact_mask & neg_mask).detach().cpu().numpy()
    neg_contact_num = np.sum(neg_contact_mask)
    
    flag = (neg_contact_num > kwargs['min_contact_points'])
    
    if kwargs['show_bad_pos_cont'] and flag:
        print('-' * 20)
        print('negative contact: ', neg_contact_num)
        print('-' * 20)
        axis_o3d = o3d.geometry.TriangleMesh().create_coordinate_frame()
        scene_o3d = o3d.io.read_triangle_mesh(osp.join(constants.posa_root, 'scenes', kwargs['scene_name'] + '.ply'))
        body_o3d = o3d.geometry.PointCloud()
        vertices_ds_np = vertices_ds.squeeze(0).detach().cpu().numpy()
        body_o3d.points = o3d.utility.Vector3dVector(vertices_ds_np)
        colors = np.zeros(vertices_ds_np.shape)
        colors[neg_contact_mask==1, 0] = 1.0
        body_o3d.colors = o3d.utility.Vector3dVector(colors)
        viz_data = [scene_o3d, axis_o3d, body_o3d]
        o3d.visualization.draw_geometries(viz_data)
    
    return flag

def init_points_culling_with_control(init_pos=None, vertices_ds=None, 
                                     scene_data=None, gen_batch=None, 
                                     max_init_points=8, angle_step=2, 
                                     instance_idxs=None, instances=None,
                                     vertices_pt=None, faces_pt=None, 
                                     remove_bad_pos_pene=False, remove_bad_pos_cont=False, 
                                     vertices_org=None,  **kwargs):
    # print('before: ')
    # print('init_pos: ', init_pos.shape)
    # print('instance_idxs: ', instance_idxs.shape)
    
    dtype = vertices_ds.dtype
    device = vertices_ds.device
    
    #* (1) initialize angles for every position
    angles = torch.arange(0, 2*np.pi, np.pi/angle_step, device=device)
    angles[0] = 1e-9
    
    init_ang = []
    for ang in angles:
        init_ang.append(ang * torch.ones(init_pos.shape[0], 1, device=device))
    init_ang = torch.cat(init_ang).to(init_pos.device)
    init_pos = init_pos.repeat(angles.shape[0], 1, 1)
    instance_idxs = np.tile(instance_idxs, angles.shape[0])
    
    rnd_ids = np.random.choice(init_pos.shape[0], init_pos.shape[0], replace=False)
    init_pos = init_pos[rnd_ids, :]
    init_ang = init_ang[rnd_ids, :]
    instance_idxs = instance_idxs[rnd_ids]
    
    #* (2) evaluate every position and angle
    vertices_org = vertices_org.unsqueeze(0)
    vertices_ds = vertices_ds.unsqueeze(0)
    with torch.no_grad():
        losses = []
        init_pos_batches = init_pos.split(1)

        for i in tqdm(range(len(init_pos_batches)), desc='Selecting'):   
            curr_vertices_org = transform_vertices(vertices_org, init_pos_batches[i], init_ang[i])
            curr_vertices_ds = transform_vertices(vertices_ds, init_pos_batches[i], init_ang[i])
            object_pc = torch.tensor(instances[instance_idxs[i]]['pc'], dtype=dtype, device=device).unsqueeze(0)
            loss = compute_afford_loss(for_eval=True, 
                                       vertices_ds=curr_vertices_ds, 
                                       scene_data=scene_data, 
                                       gen_batch=gen_batch,
                                       object_pc=object_pc,
                                       vertices_org=curr_vertices_org,
                                       **kwargs)
            losses.append(loss.item())
        
        losses = np.array(losses)
        sorted_idxs = np.argsort(losses)

    init_pos = init_pos[sorted_idxs]
    init_ang = init_ang[sorted_idxs]
    instance_idxs = instance_idxs[sorted_idxs]

    #* (3) remove situations with severe penetrations
    if remove_bad_pos_pene or remove_bad_pos_cont:
        vertices_pt = vertices_pt.unsqueeze(0)
        with torch.no_grad():
            print('removing bad positions.')
            final_idxs = []
            init_pos_batches = init_pos.split(1)
            
            i = 0
            while (len(final_idxs) < max_init_points) and (i < len(init_pos_batches)):
                curr_vertices_pt = transform_vertices(vertices_pt, init_pos_batches[i], init_ang[i])
                curr_vertices_ds = transform_vertices(vertices_ds, init_pos_batches[i], init_ang[i])
                
                flag = True
                if remove_bad_pos_pene:
                    pene_flag = (not has_severe_penetration(curr_vertices_pt, faces_pt, scene_data, **kwargs))
                    flag = (flag and pene_flag)
                if remove_bad_pos_cont:
                    cont_flag = has_contact(curr_vertices_ds, gen_batch, scene_data, **kwargs)
                    flag = (flag and cont_flag)

                if flag:
                    final_idxs.append(i)
                
                i += 1
        
        init_pos = init_pos[final_idxs]
        init_ang = init_ang[final_idxs]
        instance_idxs = instance_idxs[final_idxs]
    else:
        init_pos = init_pos[:max_init_points]
        init_ang = init_ang[:max_init_points]
        instance_idxs = instance_idxs[:max_init_points]

    # print('after: ')
    # print('init_pos: ', init_pos.shape)
    # print('init_ang: ', init_ang.shape)
    # print('instance_idxs: ', instance_idxs.shape)

    return init_pos, init_ang, instance_idxs


class opt_wrapper(object):
    def __init__(self, vertices_ds=None, scene_data=None,
                 down_sample_fn=None, down_sample_fn2=None, 
                 gen_batch=None, body_model=None, 
                 init_body_pose=None, optimizer=None, 
                 object_pc=None, **kwargs):
        
        self.vertices_ds = vertices_ds
        self.scene_data = scene_data
        self.down_sample_fn = down_sample_fn
        self.down_sample_fn2 = down_sample_fn2
        self.gen_batch = gen_batch
        self.body_model = body_model
        self.init_body_pose = init_body_pose
        self.optimizer = optimizer
        self.R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi/2, 0, 0, 'sxyz'), dtype=kwargs['dtype'], device=kwargs['device'])
        
        self.opt_pose = kwargs['opt_pose']
        self.pose_w = kwargs['pose_w']
        self.kwargs = kwargs
        
        self.object_pc = object_pc

    def compute_vertices(self, t_free, ang_free):
        if self.opt_pose:
            body_model_output = self.body_model(return_verts=True)
            pelvis = body_model_output.joints[:, 0, :].reshape(1, 3)
            vertices = body_model_output.vertices.squeeze()
            
            vertices = torch.matmul(self.R_smpl2scene, (vertices - pelvis).t()).t()
            vertices.unsqueeze_(0)
            vertices = transform_vertices(vertices, t_free, ang_free)        
            
            vertices_ds = self.down_sample_fn.forward(vertices.permute(0, 2, 1))
            vertices_ds = self.down_sample_fn2.forward(vertices_ds).permute(0, 2, 1)
        else:
            vertices = None
            vertices_ds = transform_vertices(self.vertices_ds, t_free, ang_free)
            
        return vertices, vertices_ds

    def compute_loss(self, t_free, ang_free):
        pose_loss = torch.tensor(0.0)
        if self.opt_pose:
            pose_loss = self.pose_w * F.mse_loss(self.body_model.body_pose, self.init_body_pose)
            
        vertices, vertices_ds = self.compute_vertices(t_free, ang_free)
        afford_loss = compute_afford_loss(vertices_ds=vertices_ds, 
                                          scene_data=self.scene_data, 
                                          gen_batch=self.gen_batch, 
                                          object_pc=self.object_pc, 
                                          vertices_org=vertices, **self.kwargs)
        
        total_loss = pose_loss + afford_loss
        return total_loss

    def create_fitting_closure(self, t_free, ang_free):
        def fitting_func():
            self.optimizer.zero_grad()
            loss_total = self.compute_loss(t_free, ang_free)
            loss_total.backward(retain_graph=True)
            return loss_total

        return fitting_func
