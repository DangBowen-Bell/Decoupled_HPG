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

import configargparse

base_dir = '/data/POSA'
result_dir = '/data/HPG_Result'


def parse_config():
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'Interaction capture'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='Interaction capture')
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    
    #! ******************************************************************************************
    #! paths
    parser.add_argument('--base_dir', type=str, default=base_dir)
    parser.add_argument('--data_dir', type=str, default=base_dir + '/data')
    parser.add_argument('--ds_us_dir', type=str, default=base_dir + '/mesh_ds')
    parser.add_argument('--model_folder', type=str, default=base_dir + '/smplx_models')
    parser.add_argument('--checkpoint_path', type=str, default=base_dir + '/trained_models/contact_semantics.pt')
    parser.add_argument('--checkpoint_path_no_semantics', type=str, default=base_dir + '/trained_models/contact.pt')
    
    parser.add_argument('--feature_dir', type=str, default=result_dir + '/feature')
    parser.add_argument('--affordance_dir', type=str, default=result_dir + '/posa_release')
    
    #! ******************************************************************************************
    #! posa training
    parser.add_argument('--save_checkpoints', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--load_checkpoint', type=int, default=0)
    
    parser.add_argument('--train_recordings_list', type=str, nargs='*')
    parser.add_argument('--test_recordings_list', type=str, nargs='*')
    parser.add_argument('--recordings_list', type=str, nargs='*')
    parser.add_argument('--recording_name', type=str)
    
    parser.add_argument('--train_data', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--test', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--contact_threshold', type=float, default=0.05)
    parser.add_argument('--trunc_val', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--reduce_lr', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--kl_w', type=float, default=1e-3)
    parser.add_argument('--reduction', default='mean', type=str, choices=['mean', 'sum'],)
    parser.add_argument('--loss_type', default='bce', type=str, choices=['l1', 'mse', 'bce'])
    parser.add_argument('--tensorboard', default=False, type=lambda x: x.lower() in ['true', '1'])
    
    #! ******************************************************************************************
    #! posa model
    parser.add_argument('--x_dim', type=int, default=655)
    parser.add_argument('--cond_dim', type=int, default=0)
    parser.add_argument('--f_dim', type=int, default=1)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=1)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--no_obj_classes', type=int, default=42)
    parser.add_argument('--seq_length', type=int, default=9)
    parser.add_argument('--block_size', type=int, default=1)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--normalization_mode', default='group_norm', type=str)
    parser.add_argument('--drop_out', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--input_dim', type=int, default=2619)
    parser.add_argument('--nv', type=int, default=655)
    parser.add_argument('--num_groups', type=int, default=8)

    #! ******************************************************************************************
    #! general
    parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--float_dtype', type=str, default='float32')
    
    parser.add_argument('--num_pca_comps', type=int, default=6)
    parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--down_sample', type=int, default=2)
    parser.add_argument('--up_sample', default=False, type=lambda x: x.lower() in ['true', '1'])
    
    parser.add_argument('--make_collage', default=True, type=lambda x: x.lower() in ['true', '1'])
    
    #! ******************************************************************************************
    #! generation
    parser.add_argument('--affordance_step', type=float, default=0.2)
    parser.add_argument('--opt_pose', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_clothed_mesh', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_scene', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_contact', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_semantics', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--num_rendered_samples', type=int, default=1)

    #! ****************************************************************************************** 
    #! ours 
    parser.add_argument('--cfg_name', type=str, default='')
    parser.add_argument('--scene_name', type=str, default='')
    parser.add_argument('--use_mp3d_scene', default=False, type=lambda x: x.lower() in ['true', '1'])
    
    #* feature vae
    parser.add_argument('--use_contactvae', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--select_semantics', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_semantics', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_control', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--train_cfg_name', default='', type=str)
    parser.add_argument('--num_rand_samples', type=int, default=3)
    parser.add_argument('--use_contact_mask', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--use_mask_for_pene', default=True, type=lambda x: x.lower() in ['true', '1'])
    
    #* viz & render & save
    parser.add_argument('--show_gen_sample', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--render_gen_sample', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_init_pos', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_eval_pos', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_bad_pos_pene', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--show_bad_pos_cont', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--viz_result', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--render', default=True, type=lambda x: x.lower() in ['true', '1'])    
    parser.add_argument('--save_meshes', default=True, type=lambda x: x.lower() in ['true', '1'])

    #* init poses
    parser.add_argument('--pose_debug', type=int, default=-1)
    parser.add_argument('--init_poses_ours', default=False, type=lambda x: x.lower() in ['true', '1'])    
    parser.add_argument('--posevae_cfg', type=str, default='')
    parser.add_argument('--action', type=str, default='sit')
    parser.add_argument('--init_poses_online', default=False, type=lambda x: x.lower() in ['true', '1'])    

    #* init positions
    parser.add_argument('--init_points_ours', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--init_points_step', type=int, default=20)
    parser.add_argument('--eval_points_ours', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--remove_bad_pos_pene', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--remove_bad_pos_cont', default=False, type=lambda x: x.lower() in ['true', '1']) 
    parser.add_argument('--min_contact_points', type=int)   
    parser.add_argument('--max_init_points', type=int, default=8)
    parser.add_argument('--angle_step', type=int, default=2)
    parser.add_argument('--object', type=str, default='chair')
    parser.add_argument('--object_idx', type=int)
    parser.add_argument('--position', type=str, default='on')
    
    #* eval & optimization
    parser.add_argument('--affordance_lr', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--max_opt_step', type=int, default=10)
    parser.add_argument('--num_per_sample', type=int, default=4)
    parser.add_argument('--contact_w', type=float)
    parser.add_argument('--contact_w_eval', type=float)
    parser.add_argument('--use_semantic_loss', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--semantics_w', type=float)
    parser.add_argument('--semantics_w_eval', type=float)
    parser.add_argument('--pene_w', type=float)
    parser.add_argument('--pene_w_eval', type=float)
    parser.add_argument('--pose_w', type=float)
    parser.add_argument('--use_vol_pene', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--vol_pene_w', type=float)    
    parser.add_argument('--vol_pene_w_eval', type=float)    
    parser.add_argument('--vol_den', type=int, default=5)
    parser.add_argument('--use_contact_prox', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--num_object_points', type=int, default=2048)
    parser.add_argument('--contact_prox_w', type=float)
    parser.add_argument('--contact_prox_w_eval', type=float)
    parser.add_argument('--use_contact_prob', default=False, type=lambda x: x.lower() in ['true', '1'])
    
    #* other
    parser.add_argument('--object_touch', type=str, default='table')
    parser.add_argument('--object_touch_idx', type=int)
    parser.add_argument('--object_look', type=str, default='wall')

    #! ******************************************************************************************            
    
    args = parser.parse_args()
    args_dict = vars(args)

    return args, args_dict
