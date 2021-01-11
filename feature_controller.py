from feature_config import feature_config

import face_rec.save_face_images as save_face
import face_rec.model_training as model_train
import face_rec.model_testing as model_test

import Interaction_analysis.myPose as mypose
import Interaction_analysis.find_3d_face as find_face

import Interaction_analysis.body_pose_stats as body_stats
import Interaction_analysis.body_pose_header as body_header
import Interaction_analysis.body_pose_analysis as body_analysis
import Interaction_analysis.head_pose_stats as head_stats
import Interaction_analysis.head_pose_header as head_header
import Interaction_analysis.head_pose_analysis as head_analysis

import cv2
import numpy as np

class Feature_Controller:
    def get_estimators(self):
        self.pose_estimator = mypose.get_pose_estimator((feature_config['height'],feature_config['width']), feature_config['SESSION_PATH'], feature_config['PROB_MODEL_PATH'])
        self.fa = find_face.get_face_aligner()

    def initialize_stats(self, num_subject, num_stats):
        s1_stats = np.zeros((num_subject, num_stats))
        s2_stats = np.zeros((num_subject, num_stats))
        combine_stats = np.zeros((num_subject, num_stats)) 

        return s1_stats, s2_stats, combine_stats

    def model_faces(self, num_sessions, base_folder, tablet_folder, tablet_res, raspi_folder, raspi_res):
        for subjs in range(1, feature_config['ALL_SUBJ']+1):
            if subjs in feature_config['notCounted']:
                continue
            col_count = 0
            subj = str(subjs).zfill(2)
            face_folder = base_folder + 'P' + subj + '/face_dataset/'
            if not exists(face_folder):
                try:
                    mkdir(face_folder)
                    mkdir(face_folder+ 'P' + subj)
                    mkdir(face_folder+ 'unknown')
                except Exception as e:
                    print('Error in Subj '+str(subjs)+' while creating the folder '+str(face_folder)+' file : '+str(e))
                    break

            face_model_folder = base_folder + 'P' + subj + '/face_model/'
            if not exists(face_model_folder):
                try:
                    mkdir(face_model_folder)
                except Exception as e:
                    print('Error in Subj '+str(subjs)+' while creating the folder '+str(face_folder)+' file : '+str(e))
                    break

            save_face.get_subject(num_sessions, subj, subjs, base_folder, tablet_folder, tablet_res, raspi_folder, raspi_res)

            model_train.modeling_features(face_folder, face_model_folder)
            model_train.modeling(face_model_folder)

            if feature_config['test_model']:
                print('Testing Model ' + str(subj))
                model_test.test_subject(num_sessions, subj, subjs, base_folder, tablet_folder, tablet_res, raspi_folder, raspi_res)

    def get_positions(self, main_folder, read_folder, vid_folder, face_3d_folder):
        self.get_estimators()
        self.pose_estimator.initialise()

        onlyfiles = [f for f in listdir(read_folder) if isfile(join(read_folder, f))]

        for ivid in range(0,len(onlyfiles)):
            video_filename = onlyfiles[ivid]
            # print(str(ivid))
            print(vid_folder+video_filename)

            try:
                os.mkdir(face_3d_folder+video_filename)
                os.mkdir(main_folder+'2d_pose/'+video_filename)
                os.mkdir(main_folder+'3d_pose/'+video_filename)
                os.mkdir(main_folder+'visibility/'+video_filename)
            except:
                pass

        vs = cv2.VideoCapture(vid_folder+video_filename)
        iframe = 0
        while 1:
            ret, frame = vs.read()
            iframe += 1
            if (not ret):
                break

            if iframe >= 22623 and iframe <=22630:
                # show the output frame
                # cv2.imshow("Frame", frame)
 
                # frame = imutils.resize(frame, width=427)
                mypose.save_estimate(main_folder, video_filename, self.pose_estimator, frame, iframe)
                find_face.save_face(face_3d_folder, video_filename, self.fa, frame, iframe)

            key = cv2.waitKey(1) & 0xFF

            if iframe == 22630+1:
                break

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()

        pose_estimator.close()

    def get_features(self, body_folder, num_body_features, head_folder, num_head_features, filename):

        body_s1_stats, body_s2_stats, body_combine_stats = self.initialize_stats(feature_config['NUM_SUBJ'], feature_config['NUM_BODY_STATS'])
        head_s1_stats, head_s2_stats, head_combine_stats = self.initialize_stats(feature_config['NUM_SUBJ'], feature_config['NUM_HEAD_STATS'])

        for i in range(feature_config['NUM_SUBJ']):
            #for combined session
            body_combined_session = np.zeros((1,num_body_features))
            head_combined_session = np.zeros((1,num_head_features))

            #each session
            for j in range(1,3):
                body_s1_stats, body_s2_stats, body_combined_session = body_stats.get_stats(i, j, num_body_features, body_folder, body_s1_stats, body_s2_stats, body_combined_session)
                head_s1_stats, head_s2_stats, head_combined_session = body_stats.get_stats(i, j, num_head_features, head_folder, head_s1_stats, head_s2_stats, head_combined_session)

            if body_combined_session.shape[0]>1:
                body_combined_session_stats = body_stats.statstic_features(body_combined_session)
                body_combine_stats[i, :] = body_combined_session_stats

            if head_combined_session.shape[0]>1:
                head_combined_session_stats = body_stats.statstic_features(head_combined_session)
                head_combine_stats[i, :] = head_combined_session_stats

        # numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='n', header='', footer='', comments='# ', encoding=None)[source]¶
        filename = filename
        body_headers = body_header.body_get_headers()
        head_headers = head_header.head_get_headers()

        np.savetxt(filename + 'body_session1_stats.csv',s1_stats, fmt='%10.5f', delimiter=',', headers=body_headers)
        np.savetxt(filename + 'body_session2_stats.csv',s2_stats, fmt='%10.5f',delimiter=',', headers=body_headers)
        np.savetxt(filename + 'body_combined_sessions_stats.csv', combine_stats, fmt='%10.5f', delimiter=',', headers=body_headers)

        np.savetxt(filename + 'head_session1_stats.csv',s1_stats, fmt='%10.5f', delimiter=',', headers=body_headers)
        np.savetxt(filename + 'head_session2_stats.csv',s2_stats, fmt='%10.5f',delimiter=',', headers=body_headers)
        np.savetxt(filename + 'head_combined_sessions_stats.csv', combine_stats, fmt='%10.5f', delimiter=',', headers=body_headers)

    def analyze_features(self, main_folder, body_folder, head_folder):
        body_3d_folder = main_folder+'3d_pose/'
        bodyfolders = [f for f in listdir(body_3d_folder) if not isfile(join(body_3d_folder, f))]

        head_3d_folder = main_folder+'3d_pose/'
        headfolders = [f for f in listdir(head_3d_folder) if not isfile(join(head_3d_folder, f))]

        for ivid in range(0,len(bodyfolders)):
            foldername = bodyfolders[ivid]
            print(foldername)

            bodyPose_features = []
            for iframe in range(9000,27001):#only 2min was extracted
                try:
                    pose_3d = np.load(body_3d_folder + foldername + '/' + str(iframe) +'.npy')
                    body_sync_features = body_sync(pose_3d)
                    if body_sync_features:
                        bodyPose_features.append(body_sync_features)
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    #bodyPose_features.append(np.zeros(NUM_body_FEATURES))
                    pass
            np.save(body_folder+foldername+'.npy',bodyPose_features)
            print(len(bodyPose_features))

        for ivid  in range(0,len(headfolders)):
            foldername = headfolders[ivid]
            print(foldername)

            headPose_features = []
            for iframe in range(9000,12600):#only 2min was extracted
                try:
                    pose_3d = np.load(head_3d_folder + foldername + '/' + str(iframe) +'.npy')
                    head_sync_features = head_sync(pose_3d)
                    if head_sync_features:
                        headPose_features.append(head_sync_features)
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    #headPose_features.append(np.zeros(NUM_HEAD_FEATURES))
                    pass
            np.save(head_folder+foldername+'.npy',headPose_features)
            print(len(headPose_features))


def main():
    print('Hello')
    feature_controller = Feature_Controller()
    
    print("Creating Models for faces...")
    feature_controller.model_faces(feature_config['ALL_SESS'], feature_config['base_folder'], feature_config['tablet_folder'], feature_config['tablet_res'], feature_config['raspi_folder'], feature_config['raspi_res'])

    print("Getting Head and Body Positions...")
    #feature_controller.get_positions(feature_config['main_folder'], feature_config['read_folder'], feature_config['vid_folder'], feature_config['face_3d_folder'])

    print("Extracting features...")
    #feature_controller.get_Features(feature_config['body_folder'], feature_config['NUM_BODY_FEATURES'], feature_config[body_filename])

    print("Analyzing features...")
    #feature_controller.analyze_features(feature_config['main_folder'], feature_config['body_folder'], feature_config['head_folder'])
if __name__ == "__main__": main()