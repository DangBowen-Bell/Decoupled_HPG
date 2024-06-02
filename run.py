import os
import os.path as osp

from misc import constants


def run_train_posa():
    cfg_path = 'cfg_files/ours.yaml'
    
    cmd = 'python train_posa.py'
    cmd = cmd + ' --config ' + cfg_path
    
    os.system(cmd)
    

def run_affordance(cfg, debug=False):
    if 'baseline' in cfg: 
        cfg_path = 'cfg_files/baseline.yaml'
    elif 'cposa' in cfg:
        cfg_path = 'cfg_files/cposa.yaml'
    elif 'posa' in cfg:
        cfg_path = 'cfg_files/posa.yaml'
    elif 'ours' in cfg:
        cfg_path = 'cfg_files/ours.yaml'

    cmd = 'python affordance.py'
    cmd = cmd + ' --config ' + cfg_path
    cmd = cmd + ' --cfg_name ' + cfg
    
    if debug:
        scenes = ['MPH16']
        # scenes = ['MPH1Library']
        # scenes = ['N0SittingBooth']
        # scenes = ['N3OpenArea']
    
        # scenes = ['17DRP5sb8fy-0']
        # scenes = ['17DRP5sb8fy-7']
        # scenes = ['17DRP5sb8fy-8']
        # scenes = ['sKLMLpTHeUy-1'] 
        # scenes = ['X7HyMhZNoso-16'] 
        # scenes = ['zsNo4HB9uLZ-0']
        # scenes = ['zsNo4HB9uLZ-13']
        
        action_objects = ['sit-bed']
    
        pose_debug = 4
        cmd = cmd + ' --pose_debug ' + str(pose_debug)
        
        # cmd = cmd + ' --show_gen_sample ' + str(1)
        # cmd = cmd + ' --show_init_pos ' + str(1)
        # cmd = cmd + ' --show_eval_pos ' + str(1)
        # cmd = cmd + ' --show_bad_pos_pene ' + str(1)
        # cmd = cmd + ' --show_bad_pos_cont ' + str(1)
        # cmd = cmd + ' --viz_result ' + str(1)
    else:
        scenes = constants.scene_objects_dict.keys()
        # scenes = constants.prox_splits['test']
        # scenes = constants.mp3d_scene_region_list
    
    for scene in scenes:
        cmd = cmd + ' --scene_name ' + scene
        if not debug:
            action_objects = constants.scene_action_objects_dict[scene]
            # action_objects = constants.scene_action_objects_dict_uncommon[scene]
        if scene in constants.special_scene_list:
            cmd = cmd + ' --angle_step ' + str(4)
        for action_object in action_objects: 
            action, _object = action_object.split('-')
            print('*' * 40)
            print('Generating: ', scene, ' ', action, ' ', _object)
            print('*' * 40)
            cmd = cmd + ' --action ' + action
            cmd = cmd + ' --object ' + _object
            os.system(cmd)


if __name__ == '__main__':    
    #! train contact generator: /posa_training/{train_cfg_name}
    # run_train_posa()
    
    #! pose generation: /posa_release/{cfg_name}
    # run_affordance('posa_debug', debug=True)
    # run_affordance('cposa_debug', debug=True)
    # run_affordance('ours_debug', debug=True)
    
    pass