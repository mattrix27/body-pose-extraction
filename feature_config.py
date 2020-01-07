import sys 
from os.path import dirname, realpath

base_folder = '/Users/prg/projects/mtung/body-pose-extraction/data/' # P01/S01/A_2019-09-26-17-20-15/'

All_SUBJ = 2#42
All_SESS = 7
notCounted = [8,12,18,21,29,35,36,37,38,39,40,41,42]
tablet_folder = 'A_'
tablet_video = 'vid.mp4'
raspi_folder = 'RP_'
raspi_video = ''

test_model = False

main_folder = base_folder + 'parent-child/'
read_folder = main_folder +'parent-child_vids/'
vid_folder = main_folder +'vid1/'
face_3d_folder = main_folder+'vid1_face_3d/'

body_folder = main_folder + 'body_features/'
NUM_BODY_FEATURES = 32
NUM_BODY_STATS = NUM_BODY_FEATURES * 10 * 3
NUM_SUBJ = 34

head_folder = main_folder + 'head_features/'
NUM_HEAD_FEATURES = 10
NUM_HEAD

height = 480
width = 540
DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH)
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

feature_config = {'base_folder': base_folder,

				  'All_SUBJ': All_SUBJ,
				  'All_SESS': All_SESS,
				  'notCounted': notCounted,
				  'tablet_folder': tablet_folder,
				  'tablet_video': tablet_video,
				  'raspi_folder': raspi_folder,
				  'raspi_video': raspi_video,
				  
				  'test_model': test_model,

				  'main_folder': main_folder,
				  'read_folder': read_folder,
				  'vid_folder': vid_folder,
				  'face_3d_folder': face_3d_folder,

				  'body_folder': body_folder,
				  'NUM_BODY_FEATURES': NUM_BODY_FEATURES,
				  'NUM_BODY_STATS': NUM_BODY_STATS,
				  'NUM_SUBJ': NUM_SUBJ,



				  'height': height,
				  'width': width,
				  'time_frame': 5,
				  'DIR_PATH': DIR_PATH,
				  'PROJECT_PATH': PROJECT_PATH,
				  'SAVED_SESSIONS_DIR': SAVED_SESSIONS_DIR,
				  'SESSION_PATH': SESSION_PATH,
				  'PROB_MODEL_PATH': PROB_MODEL_PATH}