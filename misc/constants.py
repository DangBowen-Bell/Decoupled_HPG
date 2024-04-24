import os
import os.path as osp
import json


#! ######################################################################
#! paths

#* prox
proxq_root = '/data/PROX/quantitative'
prox_root = '/data/PROX/qualitative'
prox_kpt_root = '/data/PROX/keypoints'
proxd_fit_root = '/data/PROX/proxd_fittings'
proxe_root = '/data/PROX-E'
proxs_root = '/data/PROX-S'
posa_root = '/data/POSA'

#* smplx
smpl_root = '/data/SMPL-X/models'
vposer_path = '/data/SMPL-X/vposerDecoderWeights.npz'
part_segm_path = '/data/SMPL-X/other/smplx_parts_segm.pkl'

#* volume matching
vertice_pairs_path = '/data/SMPL-X/other/vertice_pairs.npy'

#* other
lemo_fit_root = '/data/LEMO/PROXD_temp'
babel_root = '/data/BABEL'
amass_root = '/data/AMASS'


#! ######################################################################
#! recordings (prox)
proxq_recording_dir = osp.join(proxq_root, 'recordings')
proxq_recordings = sorted(os.listdir(proxq_recording_dir))

prox_recording_dir = osp.join(prox_root, 'recordings')
prox_recordings = sorted(os.listdir(prox_recording_dir))

proxd_recordings = sorted(os.listdir(proxd_fit_root))

lemo_recordings = sorted(os.listdir(lemo_fit_root))

prox_recordings_test = proxd_recordings


#! ######################################################################
#! frames (prox)

step = 30

prox_splits = {
    'train': ['N0Sofa', 'N3Library', 'N3Office', 'Werkraum', 'BasementSittingBooth', 'MPH8', 'MPH11', 'MPH112'],
    'test': ['MPH16', 'MPH1Library', 'N0SittingBooth', 'N3OpenArea']
}


#! ######################################################################
#! other

H, W = 1080, 1920
fx, fy = 1060.53, 1060.38
cx, cy = 951.30, 536.77

render_size = H

smplx_segm_path = '/data/SMPL-X/other/smplx_parts_segm.json'
# smplx_segm_v_path = '/data/SMPL-X/other/smplx_parts_segm_v.npy'
smplx_segm_v_path = '/data/SMPL-X/other/smplx_parts_segm_v_2.npy'
smplx_part_num = 8

contact_thre = 0.02


#! ######################################################################
#! human pose generation

hpg_root = '/data/HPG/'
hpg_result_root = '/data/HPG_Result'

#* action labes
action_labels = {
    'stand': 0,
    'sit': 1, 
    'lie': 2,
    'touch': 3
}

#* amass subset names
amass_subset_names = [
    # 'ACCAD',    
    # 'MPIHDM05',
    'CMU',
    # 'BMLrub'
]

amass_action_intervals = {
    'stand': 10,
    'sit': 10,
    'lie': 10 
}

#* actions & objects
scene_objects_dict = {
    'MPH16': ['floor', 'chair', 'bed', 'table'],
    'MPH1Library': ['floor', 'chair', 'table'],
    'N0SittingBooth': ['floor', 'sofa', 'table'],
    'N3OpenArea': ['floor', 'chair', 'sofa', 'table'],
    
    '17DRP5sb8fy-0': ['floor', 'bed'],
    '17DRP5sb8fy-7': ['floor', 'chair', 'sofa'],
    '17DRP5sb8fy-8': ['floor', 'chair', 'table'],     
    'sKLMLpTHeUy-1': ['floor', 'chair', 'sofa', 'table'],
    'X7HyMhZNoso-16': ['floor', 'chair', 'sofa', 'table'],
    'zsNo4HB9uLZ-0': ['bed', 'table'], 
    'zsNo4HB9uLZ-13': ['floor', 'sofa', 'table']
}

scene_action_objects_dict = {
    'MPH16': ['stand-floor', 'sit-chair', 'sit-bed', 'lie-bed'],
    'MPH1Library': ['stand-floor', 'sit-chair'],
    'N0SittingBooth': ['stand-floor', 'sit-sofa'],
    # 'N0SittingBooth': ['stand-floor', 'sit-sofa', 'sit-seating'],
    'N3OpenArea': ['stand-floor', 'sit-chair', 'sit-sofa', 'lie-sofa'],
    
    '17DRP5sb8fy-0': ['stand-floor', 'sit-bed', 'lie-bed'], 
    '17DRP5sb8fy-7': ['stand-floor', 'sit-chair', 'sit-sofa'], # lie-sofa
    '17DRP5sb8fy-8': ['stand-floor', 'sit-chair'], 
    'sKLMLpTHeUy-1': ['stand-floor', 'sit-sofa', 'lie-sofa'], # sit-chair
    'X7HyMhZNoso-16': ['stand-floor', 'sit-chair', 'sit-sofa'], # lie-sofa
    'zsNo4HB9uLZ-0': ['sit-bed', 'lie-bed'], 
    'zsNo4HB9uLZ-13': ['stand-floor', 'sit-sofa'] # lie-sofa
}

scene_action_objects_dict_uncommon = {
    'MPH16': ['stand-chair', 'stand-bed', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'],
    'MPH1Library': ['stand-chair', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'],
    'N0SittingBooth': ['stand-sofa', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-table'],
    'N3OpenArea': ['stand-chair', 'stand-sofa', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'],
    
    '17DRP5sb8fy-0': ['stand-bed', 'sit-floor', 'lie-floor'], 
    '17DRP5sb8fy-7': ['stand-chair', 'stand-sofa', 'sit-floor', 'lie-floor', 'lie-chair'], # lie-sofa
    '17DRP5sb8fy-8': ['stand-chair', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'], 
    'sKLMLpTHeUy-1': ['stand-chair', 'stand-sofa', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'], # sit-chair
    'X7HyMhZNoso-16': ['stand-chair', 'stand-sofa', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-chair', 'lie-table'], # lie-sofa
    'zsNo4HB9uLZ-0': ['stand-bed', 'stand-table', 'sit-table', 'lie-table'], 
    'zsNo4HB9uLZ-13': ['stand-sofa', 'stand-table', 'sit-floor', 'sit-table', 'lie-floor', 'lie-table'] # lie-sofa
}

special_scene_list = [
    '17DRP5sb8fy-7',
    '17DRP5sb8fy-8'
]

action_objects_dict = {
    'stand': ['floor'],
    'sit': ['chair', 'bed', 'sofa'],
    'lie': ['bed', 'sofa']
}

# action_num_dict = {
#     'stand': 5,
#     'sit': 5,
#     'lie': 10
# }

action_num_dict = {
    'stand': 5,
    'sit': 5,
    'lie': 5
}

actions = ['stand', 'sit', 'lie']

positions = ['on', 'around']

objects = ['floor', 'chair', 'sofa', 'table', 'bed']

objects_touch = []

objects_look = []

#* prox dataset
proxs2posa_categories_dict = {
    'MPH16': {},
    'MPH1Library': {},
    'N0SittingBooth': {'seating': 'sofa', 
                       'sofa': 'seating'},
    'N3OpenArea': {}
}

#* mp3d dataset
mp3d_root = '/data/MP3D'

mp3d_scene_region_list = [
    '17DRP5sb8fy-0',
    '17DRP5sb8fy-7',
    '17DRP5sb8fy-8',     
    'sKLMLpTHeUy-1',
    'X7HyMhZNoso-16',
    'zsNo4HB9uLZ-0', 
    'zsNo4HB9uLZ-13'
]

mp3d_scene_room_list = [    
    '17DRP5sb8fy-bedroom', 
    '17DRP5sb8fy-livingroom', 
    '17DRP5sb8fy-familyroomlounge', 
    'sKLMLpTHeUy-familyname_0_1', 
    'X7HyMhZNoso-livingroom_0_16', 
    'zsNo4HB9uLZ-bedroom0_0', 
    'zsNo4HB9uLZ-livingroom0_13'
]

mp3d_scene_region_room_dict = {
    '17DRP5sb8fy': ['0-bedroom', '7-livingroom', '8-familyroomlounge'], 
    'sKLMLpTHeUy': ['1-familyroom'], 
    'X7HyMhZNoso': ['16-livingroom'], 
    'zsNo4HB9uLZ': ['0-bedroom', '13-livingroom']
}

#* eval
metric_list = [
    'non_collision_score', 
    'vol_non_collision_score',
    'contact_score',    
    'semantic_contact_score',
    'semantic_accuracy_score'
]

with open(osp.join(proxs_root, 'contact_statistics.json'), 'r') as f:
    contact_statistics = json.load(f)
        
contact_body_parts = ['L_Leg', 'R_Leg', 'L_Hand', 'R_Hand', 'gluteus', 'back', 'thighs']
smplx_contact_ids = {}
for body_part in contact_body_parts:
    with open(osp.join(prox_root, 'body_segments', body_part + '.json'), 'r') as f:
        smplx_contact_ids[body_part] = json.load(f)['verts_ind']
        
action_body_part_mapping = {
    'sit': ['gluteus', 'thighs'],
    'lie': ['back', 'gluteus', 'thighs'],
    'stand': ['L_Leg', 'R_Leg'],
    'touch': ['L_Hand', 'R_Hand'],
}

#* posevae
action_labels = {
    'stand': 0,
    'sit': 1, 
    'lie': 2,
    'touch': 3
}
