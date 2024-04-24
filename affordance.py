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

import os
import os.path as osp
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
import random
import trimesh
import glob
import yaml
import pickle
import torchgeometry as tgm
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from scipy import stats
from PIL import ImageColor
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from lib.optimizers import optim_factory
from lib import eulerangles
from misc import constants
from misc.cmd_parser import parse_config
from utils import posa_utils, viz_utils, misc_utils, data_utils, opt_utils, other_utils


if __name__ == '__main__':
    args, args_dict = parse_config()
    
    device = torch.device('cuda' if args.use_cuda else 'cpu')
    dtype = torch.float32
    args_dict['device'] = device
    args_dict['dtype'] = dtype
    args_dict['batch_size'] = 1

    #! ******************************************************************************************
    #! make folders
    
    afd_name = args.scene_name + '-' + args.action + '-' + args.object
    affordance_dir = osp.join(args_dict.get('affordance_dir'), args.cfg_name, afd_name)

    os.makedirs(affordance_dir, exist_ok=True)
    
    pkl_folder = osp.join(affordance_dir, 'pkl')
    os.makedirs(pkl_folder, exist_ok=True)
    
    physical_metric_folder = osp.join(affordance_dir, 'physical_metric')
    os.makedirs(physical_metric_folder, exist_ok=True)
    
    rendering_folder = osp.join(affordance_dir, 'renderings')
    os.makedirs(rendering_folder, exist_ok=True)
    
    meshes_folder = osp.join(affordance_dir, 'meshes')
    os.makedirs(meshes_folder, exist_ok=True)
    
    other_folder = osp.join(affordance_dir, 'other')
    os.makedirs(other_folder, exist_ok=True)
    
    init_pkl_folder = osp.join(affordance_dir, 'init_pkl')
    os.makedirs(init_pkl_folder, exist_ok=True)
    
    init_meshes_folder = osp.join(affordance_dir, 'init_meshes')
    os.makedirs(init_meshes_folder, exist_ok=True)
    
    #! ******************************************************************************************
    #! generate or select poses
    
    if args.init_poses_online:
        print('=====> 0. generating init poses online.')
        ckpt_path = osp.join(constants.hpg_root, 'generated_pose', args.posevae_cfg, 'new_best_model.pt')
        other_utils.gen_init_poses_online(init_pkl_folder, ckpt_path, args.action, constants.action_num_dict[args.action])
    else:
        print('=====> 0. generating init poses offline.')
        other_utils.gen_init_poses_offline(init_pkl_folder, args.action, constants.action_num_dict[args.action])
    
    if args.init_poses_ours:
        pkl_file_dir = init_pkl_folder
        pkl_file_paths = glob.glob(osp.join(pkl_file_dir, '*.pkl'))    
    else:
        pkl_file_dir = osp.join(constants.hpg_root, 'generated_pose', args.posevae_cfg, args.action)
        pkl_file_paths = glob.glob(osp.join(pkl_file_dir, '*.pkl'))        
        
    pkl_file_paths = sorted(pkl_file_paths)
    
    if len(os.listdir(meshes_folder)) == (len(pkl_file_paths) * min(args.max_init_points, args.num_per_sample)):
        exit(0)
        
    if args.pose_debug != -1:
        pkl_file_paths = [pkl_file_paths[args.pose_debug]]
            
    #! ******************************************************************************************
    #! load model
    
    A_1, U_1, D_1 = posa_utils.get_graph_params(args.ds_us_dir, 1, args.use_cuda)
    down_sample_fn = posa_utils.ds_us(D_1).to(device)
    up_sample_fn = posa_utils.ds_us(U_1).to(device)
    A_2, U_2, D_2 = posa_utils.get_graph_params(args.ds_us_dir, 2, args.use_cuda)
    down_sample_fn2 = posa_utils.ds_us(D_2).to(device)
    up_sample_fn2 = posa_utils.ds_us(U_2).to(device)
    
    if args.use_contactvae: 
        if args.use_control:
            model = misc_utils.load_model_checkpoint_with_control(**args_dict).to(device)
        else:
            if not args.use_semantics:
                args.checkpoint_path = args.checkpoint_path_no_semantics
            model = misc_utils.load_model_checkpoint(**args_dict).to(device)
    
    contact_body_parts = constants.action_body_part_mapping[args.action]
    smplx_contact_ids = []
    for body_part in contact_body_parts:
        smplx_contact_ids += constants.smplx_contact_ids[body_part]
    smplx_contact_ids = np.array(smplx_contact_ids, dtype=np.int8)

    #! ******************************************************************************************
    #! modify args & args_dict
    
    #* mpcat data
    category_dict, category_list = other_utils.load_mpcat40_data()
    
    #* update object index
    args.object_idx = category_list.index(args.object)
    args_dict['object_idx'] = args.object_idx
        
    scene_name = args.scene_name
    if scene_name in constants.mp3d_scene_region_list:
        args.use_mp3d_scene = True
        args_dict['use_mp3d_scene'] = True 
        args.select_semantics = False
        args_dict['select_semantics'] = False
        args.use_semantic_loss = False
        args_dict['use_semantic_loss'] = False

    #! ******************************************************************************************
    #! load scene data

    scene_data = other_utils.load_scene_data(**args_dict)
    instances = scene_data['instances']
    scene_data['instances'] = None
            
    #! ******************************************************************************************
    #! visualization
    
    scene_o3d = o3d.io.read_triangle_mesh(scene_data['scene_path'])
    axis_o3d = o3d.geometry.TriangleMesh().create_coordinate_frame()
    
    scene_seg_o3d = instances[0]['mesh']
    for i in range(len(instances)-1):
        scene_seg_o3d += instances[i+1]['mesh']
    
    basic_viz_data = [axis_o3d, scene_seg_o3d]
    
    #! ******************************************************************************************
    
    for pkl_file_path in tqdm(pkl_file_paths, desc='PKL'):
        pkl_file_basename = osp.splitext(osp.basename(pkl_file_path))[0]
        
        # if args.init_poses_ours:
        #     pkl_name = args.posevae_cfg + '-' + args.action + '-' + pkl_file_basename 
        # else:
        #     pkl_name = pkl_file_basename   
                            
        pkl_name = pkl_file_basename
        
        #! ******************************************************************************************
        
        print('loading body: {}'.format(pkl_file_path))
        # if args.init_poses_ours:
        #     vertices_org, vertices_can, faces_arr, body_model, _, pelvis, torch_param, _ = data_utils.pkl_to_canonical_with_control(pkl_file_path, **args_dict)
        # else:
        #     vertices_org, vertices_can, faces_arr, body_model, _, pelvis, torch_param, _ = data_utils.pkl_to_canonical(pkl_file_path, **args_dict)
        vertices_org, vertices_can, faces_arr, body_model, _, pelvis, torch_param, _ = data_utils.pkl_to_canonical_with_control(pkl_file_path, **args_dict)
       
        #? vertices_org: centered at the origin, in the scene coordinate
        #? vertices_can: centered at the origin, *not* in the scene coordinate (used in the posa model)
        
        #* foot height
        pelvis_z_offset = -vertices_org.detach().cpu().numpy().squeeze()[:, 2].min()
        pelvis_z_offset = pelvis_z_offset.clip(min=0.5)
        
        init_body_pose = body_model.body_pose.detach().clone()
        
        #! ******************************************************************************************

        #* downsample vertices
        vertices_org_ds = down_sample_fn.forward(vertices_org.unsqueeze(0).permute(0, 2, 1))
        vertices_org_ds = down_sample_fn2.forward(vertices_org_ds).permute(0, 2, 1).squeeze()
        vertices_can_ds = down_sample_fn.forward(vertices_can.unsqueeze(0).permute(0, 2, 1))
        vertices_can_ds = down_sample_fn2.forward(vertices_can_ds).permute(0, 2, 1).squeeze()

        #* generate and select semantic map
        # retrive all the generated feature map, select the best one 
        # the best one need to have contact vertices with the scene
        # do this if the scene semantces exists
        # else only generate one contact map
        print('=====> 1. generating feature map.')
        
        smplx_contact_mask_us = np.zeros((1, len(vertices_org), 1 + args.no_obj_classes))
        smplx_contact_mask_us[:, smplx_contact_ids, :] = 1.0
        smplx_contact_mask_us = torch.tensor(smplx_contact_mask_us, dtype=dtype, device=device)
        smplx_contact_mask = down_sample_fn.forward(smplx_contact_mask_us.clone().transpose(1, 2))
        smplx_contact_mask = down_sample_fn2.forward(smplx_contact_mask).transpose(1, 2)
        
        if args.use_contactvae:
            if args.use_control:
                # TODO: add contact object code
                z = torch.tensor(np.random.normal(0, 1, (1, args.z_dim)).astype(np.float32)).to(device)
                
                object_code = np.zeros(len(category_list))
                object_code[args.object_idx] = 1
                object_code = torch.tensor(object_code).to(dtype).to(device)
                object_code = object_code.unsqueeze(0)
                z = torch.cat((z, object_code), dim=1)
                z = model.fc(z)
                
                gen_batches = model.decoder(z, vertices_can_ds.unsqueeze(0).expand(1, -1, -1)).detach()
            else:
                if args.select_semantics:
                    scene_semantics = scene_data['scene_semantics']
                    scene_obj_ids = np.unique(scene_semantics.nonzero().detach().cpu().numpy().squeeze()).tolist()
                    
                    n = 50
                    z = torch.tensor(np.random.normal(0, 1, (n, args.z_dim)).astype(np.float32)).to(device)
                    gen_batches = model.decoder(z, vertices_can_ds.unsqueeze(0).expand(n, -1, -1)).detach()
                    
                    selected_batch = None
                    for i in range(gen_batches.shape[0]):
                        x, x_semantics = data_utils.batch2features(gen_batches[i], **args_dict)
                        x_semantics = np.argmax(x_semantics, axis=-1)
                        modes = stats.mode(x_semantics[x_semantics != 0])
                        most_common_obj_id = modes.mode[0]
                        if most_common_obj_id not in scene_obj_ids:
                            continue
                        selected_batch = i
                        break

                    if selected_batch is not None:
                        gen_batches = gen_batches[i].unsqueeze(0)
                    else:
                        print('No good semantic feat found - Results might be suboptimal')
                        gen_batches = gen_batches[0].unsqueeze(0)
                else:
                    z = torch.tensor(np.random.normal(0, 1, (1, args.z_dim)).astype(np.float32)).to(device)
                    gen_batches = model.decoder(z, vertices_can_ds.unsqueeze(0).expand(1, -1, -1)).detach()

            if args.use_contact_mask:
                gen_batches = smplx_contact_mask * gen_batches
        else:
            gen_batches = smplx_contact_mask

        #! ******************************************************************************************
        
        # there is only 1 sample 
        # because we only select the best or 
        # generate one candidate in previous step
        for sample_id in range(gen_batches.shape[0]):
            result_filename = pkl_name + '_{}'.format(sample_id)
            
            gen_batch = gen_batches[sample_id, :, :].unsqueeze(0)
            gen_batch_us = up_sample_fn2.forward(gen_batch.clone().transpose(1, 2))
            gen_batch_us = up_sample_fn.forward(gen_batch_us).transpose(1, 2)

            #* viz feature map
            if args.show_gen_sample:
                gen_sample = viz_utils.show_sample(vertices_org, gen_batch_us, faces_arr, **args_dict)
                o3d.visualization.draw_geometries(gen_sample)
                
            #* render feature map
            if args.render_gen_sample:
                gen_sample_img = viz_utils.render_sample(gen_batch_us, vertices_org, faces_arr, **args_dict)[0]
                gen_sample_img.save(osp.join(rendering_folder, pkl_name + '_gen.png'))

            #* get vertex colors using feature map
            x, x_semantics = data_utils.batch2features(gen_batch_us, args.use_semantics)
            x_contact = (x > 0.5).astype(np.int)
            if args.use_semantics:
                x_semantics = np.argmax(x_semantics, axis=1)
            
            # 0/1
            # vertex_colors = np.ones((vertices_org.shape[0], 3)) * np.array(viz_utils.default_color)
            # vertex_colors[x_contact.flatten()==1, :3] = [0.0, 0.0, 1.0]
            
            # 0~1
            x_clip = np.clip(x, 0, 1)      
            vertex_colors = np.ones((vertices_org.shape[0], 3)) * np.array([0.4, 0.0, 0.0])
            vertex_colors[:, 1] = x_clip / 2
            vertex_colors[:, 2] = (1.0 - x_clip) / 2

            #! ******************************************************************************************            
            print('=====> 2. sampling init points.')        
            if args.init_points_ours:
                init_pos, instance_idxs = misc_utils.create_init_points_with_control(scene_data['bbox'].detach().cpu().numpy(),
                                                                                   instances, args.position, args.object, 
                                                                                   pelvis_z_offset, args.init_points_step)
            else: 
                init_pos = misc_utils.create_init_points(scene_data['bbox'].detach().cpu().numpy(), 
                                                         args.affordance_step, pelvis_z_offset)
                instance_idxs = np.zeros(len(init_pos), dtype=np.uint8)

            if init_pos is None:
                exit(0)
            init_pos = torch.tensor(init_pos, dtype=dtype, device=device).reshape(-1, 1, 3)
            
            if args.show_init_pos:
                viz_data = basic_viz_data[:]
                for i in range(len(init_pos)):
                    viz_data.append(viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                o3d.visualization.draw_geometries(viz_data)
            
            #! ******************************************************************************************

            print('=====> 3. evaling init points & angles.')
            if args.eval_points_ours:
                # use downsampled vertices to faster speed
                vertices_pt = down_sample_fn.forward(vertices_org.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).squeeze()
                faces_pt = trimesh.load(osp.join(args.ds_us_dir, 'mesh_{}.obj'.format(1)), process=False).faces
                
                init_pos, init_ang, instance_idxs = opt_utils.init_points_culling_with_control(
                    init_pos=init_pos, vertices_ds=vertices_org_ds, 
                    scene_data=scene_data, gen_batch=gen_batch, 
                    instance_idxs=instance_idxs, instances=instances,
                    vertices_pt=vertices_pt, faces_pt=faces_pt, 
                    vertices_org=vertices_org, **args_dict)
            else:
                init_pos, init_ang, instance_idxs = opt_utils.init_points_culling(
                    init_pos=init_pos, vertices_ds=vertices_org_ds, 
                    scene_data=scene_data, gen_batch=gen_batch, 
                    instance_idxs=instance_idxs, **args_dict)
        
            if args.show_eval_pos:
                points = []
                vertices_np = vertices_org.detach().cpu().numpy()
                bodies = []
                objects = []
                for i in range(len(init_pos)):
                    points.append(viz_utils.create_o3d_sphere(init_pos[i].detach().cpu().numpy().squeeze(), radius=0.03))
                    
                    rot_aa = torch.cat((torch.zeros((1, 2), device=device), init_ang[i].reshape(1, 1)), dim=1)
                    rotmat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3].detach().cpu().numpy().squeeze()
                    v = np.matmul(rotmat, vertices_np.transpose()).transpose() + init_pos[i].detach().cpu().numpy()
                    body = viz_utils.create_o3d_mesh_from_np(vertices=v, faces=faces_arr, vertex_colors=vertex_colors)
                    bodies.append(body)
                    
                    interact_object = instances[instance_idxs[i]]['mesh']
                    interact_object.paint_uniform_color([0, 0, 0])
                    objects.append(interact_object)

                viz_data = basic_viz_data[:]
                # viz_data = viz_data + points
                viz_data = viz_data + bodies + objects
                o3d.visualization.draw_geometries(viz_data)
            
            #! ******************************************************************************************
            
            final_vertices = []
            final_vertices_ds = []
            final_losses = []
            final_body_poses = []
            final_rotmats = []
            final_transls = []
            final_instance_idxs = []
            final_contact_feats = []
            final_init_vertices = []
            
            print('=====> 4. optimization.')      
            for i in tqdm(range(init_pos.shape[0]), desc='Optimization'):
                body_model.reset_params(**torch_param)
                
                t_free = init_pos[i].reshape(1, 1, 3).clone().detach().requires_grad_(True)
                ang_free = init_ang[i].reshape(1, 1).clone().detach().requires_grad_(True)
                free_param = [t_free, ang_free]
                if args.opt_pose:
                    free_param += [body_model.body_pose]
                    
                optimizer, _ = optim_factory.create_optimizer(free_param, optim_type='lbfgsls',
                                                              lr=args_dict.get('affordance_lr'), 
                                                              ftol=1e-9, gtol=1e-9,
                                                              max_iter=args.max_iter)
                
                object_pc = torch.tensor(instances[instance_idxs[i]]['pc'], dtype=dtype, device=device).unsqueeze(0)
                
                opt_wrapper_obj = opt_utils.opt_wrapper(vertices=vertices_org_ds.unsqueeze(0), scene_data=scene_data,
                                                        down_sample_fn=down_sample_fn, down_sample_fn2=down_sample_fn2,
                                                        optimizer=optimizer, gen_batch=gen_batch, 
                                                        body_model=body_model, init_body_pose=init_body_pose, 
                                                        object_pc=object_pc, **args_dict)
                
                init_vertices, _ = opt_wrapper_obj.compute_vertices(t_free, ang_free)
                final_init_vertices.append(init_vertices.squeeze().detach().cpu().numpy())
                
                closure = opt_wrapper_obj.create_fitting_closure(t_free, ang_free)
                for _ in range(args.max_opt_step):
                    loss = optimizer.step(closure)
                
                vertices, vertices_ds = opt_wrapper_obj.compute_vertices(t_free, ang_free)
                final_vertices.append(vertices.squeeze().detach().cpu().numpy())
                final_vertices_ds.append(vertices_ds.squeeze().detach().cpu().numpy())
            
                if torch.is_tensor(loss):
                    loss = float(loss.detach().cpu().squeeze().numpy())
                final_losses.append(loss)

                final_body_poses.append(body_model.body_pose.clone().squeeze())
                
                rot_aa = torch.cat((torch.zeros((1, 2), device=device), ang_free.reshape(1, 1)), dim=1)
                rotmat = tgm.angle_axis_to_rotation_matrix(rot_aa.reshape(-1, 3))[:, :3, :3].reshape(3, 3)
                final_rotmats.append(rotmat)
                
                final_transls.append(t_free[0, 0, :].detach().cpu().numpy())
                
                final_instance_idxs.append(instance_idxs[i])
                
                final_contact_feats.append(gen_batch[0, :, 0].clone().detach().squeeze(0).squeeze(-1).cpu().numpy())

            #! ******************************************************************************************
            
            print('=====> 5. saving result.')    
            final_losses = np.array(final_losses)
            if len(final_losses > 0):
                sorted_idxs = np.argsort(final_losses)
                for i in tqdm(range(min(args.num_per_sample, len(final_losses))), desc='Saving'):
                    real_idx = sorted_idxs[i]
                    vertices = final_vertices[real_idx]
                    vertices_ds = final_vertices_ds[real_idx]
                    body_pose = final_body_poses[real_idx]
                    rotmat = final_rotmats[real_idx]
                    transl = final_transls[real_idx]
                    instance_idx = final_instance_idxs[real_idx]
                    object_pc = instances[instance_idx]['pc']
                    contact_feat = final_contact_feats[real_idx]
                    init_vertices = final_init_vertices[real_idx]
                    
                    #* save parameters
                    R_smpl2scene = torch.tensor(eulerangles.euler2mat(np.pi/2, 0, 0, 'sxyz'), dtype=dtype, device=device)
                    Rcw = torch.matmul(rotmat, R_smpl2scene)
                    torch_param['body_pose'] = body_pose
                    torch_param = misc_utils.smpl_in_new_coords(torch_param, Rcw, t_free.reshape(1, 3), rotation_center=pelvis, **args_dict)
                    param = {}
                    for key in torch_param.keys():
                        param[key] = torch_param[key].detach().cpu().numpy()
                    with open(osp.join(pkl_folder, '{}_{}.pkl'.format(result_filename, i)), 'wb') as f:
                        pickle.dump(param, f)
                        
                    #* evaluate physical metric                    
                    eval_scores = misc_utils.eval_physical_metric(vertices, scene_data, 
                                                                  args.vol_den, args.contact_threshold, args.action, 
                                                                  vertices_ds, object_pc)
                    with open(osp.join(physical_metric_folder, '{}_{}.yaml'.format(result_filename, i)), 'w') as f:
                        yaml.dump(eval_scores, f)
                
                    #* viz result
                    if args.viz_result:
                        viz_data = basic_viz_data[:]
                        body = viz_utils.create_o3d_mesh_from_np(vertices=vertices, faces=faces_arr, vertex_colors=vertex_colors)
                        viz_data.append(body)
                        o3d.visualization.draw_geometries(viz_data)

                    body = trimesh.Trimesh(vertices, faces_arr, process=False)
                    body_semantics = trimesh.Trimesh(vertices, faces_arr, vertex_colors=vertex_colors, process=False)
                    init_body_semantics = trimesh.Trimesh(init_vertices, faces_arr, vertex_colors=vertex_colors, process=False)
                    
                    #* save meshes
                    if args.save_meshes:
                        body_semantics.export(osp.join(meshes_folder, '{}_{}.obj'.format(result_filename, i)))
                        init_body_semantics.export(osp.join(init_meshes_folder, '{}_{}.obj'.format(result_filename, i)))
                        
                    #* render
                    if args.render: 
                        clothed_body = None
                                            
                        scene_mesh = other_utils.o3d_to_trimesh(scene_o3d)
                        img_collage = viz_utils.render_interaction_snapshot(body, scene_mesh, clothed_body,
                                                                            body_center=True,
                                                                            collage_mode='horizantal', **args_dict)
                        img_collage.save(osp.join(rendering_folder, '{}_{}.png'.format(result_filename, i)))
                        
                        scene_mesh_semantics = other_utils.o3d_to_trimesh(scene_seg_o3d)
                        img_collage = viz_utils.render_interaction_snapshot(body_semantics, scene_mesh_semantics, clothed_body,
                                                                            body_center=True,
                                                                            collage_mode='horizantal', **args_dict)
                        img_collage.save(osp.join(rendering_folder, '{}_{}_semantics.png'.format(result_filename, i)))
                        
                    #* other: 
                    other_data = {
                        'instance_idx': instance_idx,
                        'contact_feat': contact_feat, 
                        'body_pose': body_pose.detach().cpu().numpy(),
                        'transl': transl,
                        'rotmat': rotmat.detach().cpu().numpy(),
                        'vertices_ds': vertices_ds,
                    }
                    other_path = osp.join(other_folder, '{}_{}.pkl'.format(result_filename, i))
                    with open(other_path, 'wb') as f:
                        pickle.dump(other_data, f)
                        