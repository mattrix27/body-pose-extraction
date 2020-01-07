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

class Feature_Controller:
    def get_estimators(self):
        self.pose_estimator = mypose.get_pose_estimator((feature_config['height'],feature_config['width']), feature_config['SESSION_PATH'], feature_config['PROB_MODEL_PATH'])
        self.fa = find_face.get_face_aligner()

    def initialize_stats(self, num_subject, num_body_stats):
        s1_stats = np.zeros((num_subject, num_body_stats))
        s2_stats = np.zeros((num_subject, num_body_stats))
        combine_stats = np.zeros((num_subject, num_body_stats)) 

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

    def get_features(self, body_folder, num_body_features, filename):

        s1_stats, s2_stats, combine_stats = self.initialize_stats(feature_config['NUM_SUBJ'], feature_config['NUM_BODY_STATS'])

        for i in range(feature_config['NUM_SUBJ']):
            #for combined session
            combined_session = np.zeros((1,num_body_features))

            #each session
            for j in range(1,3):
                s = np.zeros((1,num_body_features))
                startfilename = "p"+"%02d" % (i+1,)+"_s"+str(j)+"_vid_parent_annotation_" #p15_s2_vid_parent_annotation_2019-03-31-13-04-39.mp4.npy
                onlysessionfile = [f for f in listdir(body_folder) if isfile(join(body_folder, f)) and f.startswith(startfilename)]
                if len(onlysessionfile) < 1:
                    # print(startfilename)
                    # print(onlysessionfile)
                    pass
                elif len(onlysessionfile) > 1:
                    #merg the files into one array
                    for abc in range(len(onlysessionfile)):
                        print(onlysessionfile[abc])
                        if abc == 0:
                            temp = np.load(body_folder + onlysessionfile[abc])
                            if temp.shape[0] > 1:
                                s = temp
                        else:
                            temp = np.load(body_folder + onlysessionfile[abc])
                            if temp.shape[0] > 1:
                                if s.shape[0] > 1:
                                    s = np.append(s, temp, axis=0)
                                else:
                                    s = temp
                else:
                    print(onlysessionfile[0])
                    s = (np.load(body_folder + onlysessionfile[0]))

                if s.shape[0]>1:
                    #extract the session statstics
                    session_stats = statstic_features(s)
                    if j == 1:
                        s1_stats[i,:] = session_stats
                    else:
                        s2_stats[i,:] = session_stats


                    #combine the two sessions for combined analysis
                    if combined_session.shape[0]>1:
                        combined_session = np.append(combined_session, s, axis=0)
                    else:
                        combined_session = s

            if combined_session.shape[0]>1:
                combined_session_stats = statstic_features(combined_session)
                combine_stats[i, :] = combined_session_stats

        # numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='n', header='', footer='', comments='# ', encoding=None)[source]¶
        filename = filename
        headers = self.body_header.body_get_headers()

        np.savetxt(filename + 'body_session1_stats.csv',s1_stats, fmt='%10.5f', delimiter=',', headers=headers)
        np.savetxt(filename + 'body_session2_stats.csv',s2_stats, fmt='%10.5f',delimiter=',', headers=headers)
        np.savetxt(filename + 'body_combined_sessions_stats.csv', combine_stats, fmt='%10.5f', delimiter=',', headers=headers)


    def analyze_features(self, main_folder):
        body_3d_folder = main_folder+'3d_pose/'

        for ivid in range(0,len(onlyfolders)):
            foldername = onlyfolders[ivid]
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
            np.save(save_body_features+foldername+'.npy',bodyPose_features)
            print(len(bodyPose_features))

def main():
    print('Hello')
    feature_controller = Feature_Controller()
    
    print("Creating Models for faces...")
    feature_controller.model_faces(feature_config['ALL_SESS'], feature_config['base_folder'], feature_config['tablet_folder'], feature_config['tablet_res'], feature_config['raspi_folder'], feature_config['raspi_res'])

    print("Getting Head and Body Positions...")
    #feature_controller.get_positions(feature_config['main_folder'], feature_config['read_folder'], feature_config['vid_folder'], feature_config['face_3d_folder'])

    print("Analyzing features...")
    #feature_controller.get_Features(feature_config['body_folder'], feature_config['NUM_BODY_FEATURES'], feature_config[body_filename])
if __name__ == "__main__": main()