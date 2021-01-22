import sys 
import os

proj_folder = os.getcwd() + '/'
base_folder = proj_folder + 'data/' # P01/S01/A_2019-09-26-17-20-15/'

lifting = False
openpose = True
openpose_models = '/home/mtung/openpose/models/'
openpose_display = False
save_poses = True
save_features = True

All_SUBJ = 34#42
All_SESS = 2
#notCounted = [8,12,18,21,29,35,36,37,38,39,40,41,42]
notCounted = []
tablet_folder = 'vid.mp4'
tablet_video = 'vid.mp4'
raspi_folder = 'RP_'
raspi_video = 'RP.mp4'

test_model = False
torch = proj_folder + 'face_rec/openface_nn4.small2.v1.t7'
protoPath = "face_rec/face_detection_model/deploy.prototxt"
modelPath = "face_rec/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

main_folder = base_folder + 'parent-child/'
read_folder = '/storage1/mtung/projects/body_pose/data/parent-child/'
# read_folder = main_folder +'parent-child_vids/'
vid_folder = main_folder +'vid1/'
face_3d_folder = main_folder+'vid1_face_3d/'
parents = ['dmom', 'nmom', 'parent']
children = ['derek', 'nick', 'child']

body_folder = main_folder + 'body_features/'
op_body_folder = main_folder + 'op_body_features/'
NUM_BODY_FEATURES = 32
NUM_BODY_STATS = NUM_BODY_FEATURES * 10 * 3
NUM_SUBJ = 34

head_folder = main_folder + 'head_features/'
op_head_folder = main_folder + 'op_head_features/'
NUM_HEAD_FEATURES = 10
NUM_HEAD_STATS = NUM_HEAD_FEATURES * 10 * 3

continuous = False
frame_window = 50
overlap_prop = 0.5

height = 480
width = 640

SAVED_SESSIONS_DIR = 'Interaction_analysis/data/saved_sessions/'
SESSION_PATH = SAVED_SESSIONS_DIR + 'init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + 'prob_model/' +  'prob_model_params.mat'

feature_config = {'lifting': lifting,
				  'openpose': openpose,
				  'openpose_models': openpose_models,
				  'openpose_display': openpose_display,
				  'save_poses': save_poses,
				  'save_features': save_features,

				  'base_folder': base_folder,

				  'ALL_SUBJ': All_SUBJ,
				  'ALL_SESS': All_SESS,
				  'notCounted': notCounted,
				  'tablet_folder': tablet_folder,
				  'tablet_video': tablet_video,
				  'raspi_folder': raspi_folder,
				  'raspi_video': raspi_video,
				  
				  'torch': torch,
				  'protoPath': protoPath,
				  'modelPath': modelPath,
				  'test_model': test_model,

				  'main_folder': main_folder,
				  'read_folder': read_folder,
				  'vid_folder': vid_folder,
				  'face_3d_folder': face_3d_folder,

				  'parents': parents,
				  'children': children,

				  'body_folder': body_folder,
				  'op_body_folder': op_body_folder,
				  'NUM_BODY_FEATURES': NUM_BODY_FEATURES,
				  'NUM_BODY_STATS': NUM_BODY_STATS,
				  'head_folder': head_folder,
				  'op_head_folder': op_head_folder,
				  'NUM_HEAD_FEATURES': NUM_HEAD_FEATURES,
				  'NUM_HEAD_STATS': NUM_HEAD_STATS,				  
				  'NUM_SUBJ': NUM_SUBJ,
				  'continuous': continuous,
				  'frame_window': frame_window,
				  'overlap_prop' : overlap_prop,

				  'height': height,
				  'width': width,
				  'frame_skip': 30,
				  'startframe': 1800,
				  'endframe': 15000,
				  'SAVED_SESSIONS_DIR': SAVED_SESSIONS_DIR,
				  'SESSION_PATH': SESSION_PATH,
				  'PROB_MODEL_PATH': PROB_MODEL_PATH}