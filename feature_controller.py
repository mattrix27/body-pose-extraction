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

import Interaction_analysis.body_openpose_analysis as openpose_analysis

import sys
sys.path.append('/usr/local/python');
from openpose import pyopenpose as op

import cv2
import numpy as np
import pandas as pd
import sys
import csv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
from tensorflow.python.util import deprecation
tensorflow.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
deprecation._PRINT_DEPRECATION_WARNINGS = False

from os.path import isfile, join, exists
from os import listdir, mkdir
import datetime


class Feature_Controller:
    def __init__(self):
        self.height = 0
        self.width = 0
        self.fa = None
        self.bodies = dict()

        self.started = True
        self.head_frames = []
        self.body_frames = []
        self.head_overlap = []
        self.body_overlap = []
        self.over_frame = feature_config['overlap_prop'] * feature_config['frame_window']

        self.head_raw = [] #TODO
        self.body_raw = [] #TODO
        self.head_session_stats = []
        self.body_session_stats = []
        self.cnt = 0

        self.accuracy = dict()

    def get_face_estimator(self):
        '''
            Initialize estimator for face alignment
        '''
        self.fa = find_face.get_face_aligner()

    def get_pose_estimator(self, height, width):
        '''
            Initialize estimator with new resolution for 'lifting' pose detection
        '''
        self.pose_estimator = mypose.get_pose_estimator(height, width, feature_config['SESSION_PATH'], feature_config['PROB_MODEL_PATH'])
        self.pose_estimator.initialise()
        self.height = height
        self.width = width

    def assign_body(self, faces, face_index, poses, image_size):
        '''
            Takes poses info from either openpose or lifting and assigns it to either parent or child

        '''
        #keeps track of body ojbects that have been already been assigned in case of duplicates
        processed = [] 
        assigned = set()

        #constants for determining confidence of assignment
        threshold = 0.2 #proportion of image for distance 
        k = 0.75 #initial confidence
        f = 1 #gain

        for name in faces:
            confidence = 0
            ((startX, startY), (endX, endY)) = faces[name]
            for i in range(len(poses)):
                noseX, noseY, noseZ = poses[i][face_index]

                #When pose data for nose is in the window of the face detection
                if noseX >= startX and noseX <= endX:
                    if noseY >= startY and noseY <= endY:
                        if name in self.bodies:
                            prevX, prevY, prevZ = self.bodies[name]['pose'][face_index]
                            diff = ((abs(noseX-prevX)/image_size[1])**2 + (abs(noseY-prevY)/image_size[0])**2)**0.5
                            #Check the distance difference between the new overlap and the previous pose in case the face detection was wrong
                            if diff > threshold:
                                #print("WOAH, it looks like they moved a lot")

                                #Checks how confident we were in our old pose to either to overwrite or forgo this overlap
                                if diff > self.bodies[name]['confidence']:
                                    data = dict()
                                    data['pose'] = poses[i]
                                    data['confidence'] = k
                                    data['times_found'] = 1
                                    self.bodies[name] = data
                                    processed.append(name)
                                    assigned.add(i)
                                # else:
                                    #print("POSSIBLE MISCLASSIFICATION, ignoring")

                            #updates the confidence of our pose assignments
                            else:
                                self.bodies[name]['times_found'] += 1
                                new_conf = f * ((self.bodies[name]['confidence']+(k*(1-diff))/self.bodies[name]['times_found']))

                                self.bodies[name]['confidence'] = min(new_conf, 1.0)
                                self.bodies[name]['pose'] = poses[i]
                                processed.append(name)
                                assigned.add(i)
                        #if the face has never been assigned, we assign it
                        else:
                            data = dict()
                            data['pose'] = poses[i]
                            data['confidence'] = k
                            data['times_found'] = 1
                            self.bodies[name] = data
                            processed.append(name)
                            assigned.add(i)

        #Goes through all the bodies/people previously found when faces aren't detected
        for name in self.bodies:
            if name not in processed:
                #print("estimating: ", name)
                min_dist = int((image_size[0]**2 + image_size[1]**2)**0.5)
                min_index = -1
                for i in range(len(poses)):
                    if i not in assigned:
                        noseX, noseY, noseZ = poses[i][face_index]
                        prevX, prevY, prevZ = self.bodies[name]['pose'][face_index]
                        dist = ((abs(noseX-prevX)/image_size[1])**2 + (abs(noseY-prevY)/image_size[0])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            min_index = i
                #Assign the body pose that is closest to its previous to same person
                if min_index != -1:
                    self.bodies[name]['confidence'] -= max(0,(k * min_dist))
                    self.bodies[name]['pose'] = poses[min_index]
                    if not self.bodies[name]['times_found']:
                        self.bodies[name]['times_found'] = 0
                    processed.append(name)
                    assigned.add(min_index)

    def assign_head(self, heads, face_index, head_index, image_size):
        assigned = set()
        if len(heads) < 2:
            print("WARNING!!! LESS THAN 2 HEADS FOUND")
        for name in self.bodies:
            poseX, poseY, poseZ = self.bodies[name]['pose'][face_index]
            min_dist = int((image_size[0]**2 + image_size[1]**2)**0.5)
            min_index = -1
            for i in range(len(heads)):
                if i not in assigned:
                    noseX, noseY, noseZ = heads[i][head_index]
                    dist = ((abs(noseX-poseX)/image_size[1])**2 + (abs(noseY-poseY)/image_size[0])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        min_index = i
            if min_index != -1 and min_dist < 0.05:
                self.bodies[name]['face'] = heads[min_index]
                assigned.add(min_index)

    def initialize_stats(self, num_subject, num_stats):
        s1_stats = np.zeros((num_subject, num_stats))
        s2_stats = np.zeros((num_subject, num_stats))
        combine_stats = np.zeros((num_subject, num_stats)) 

        return s1_stats, s2_stats, combine_stats

    def model_faces(self, num_sessions, base_folder, read_folder, tablet_folder, tablet_video, raspi_folder, raspi_video):
        for subjs in range(1, feature_config['ALL_SUBJ']+1):
            if subjs in feature_config['notCounted']:
                continue
            col_count = 0
            subj = str(subjs).zfill(2)
            face_folder = base_folder + 'face_recognition/' + 'P' + subj + '/face_dataset/'
            if not exists(face_folder):
                try:
                    if not exists(base_folder + 'face_recognition/' + 'P' + subj):
                        mkdir(base_folder + 'face_recognition/' + 'P' + subj)
                    mkdir(face_folder)
                    #mkdir(face_folder + 'unknown')
                except Exception as e:
                    print('Error in Subj '+str(subjs)+' while creating the folder '+str(face_folder)+' file : '+str(e))
                    break

            face_model_folder = base_folder + 'face_recognition/'+ 'P' + subj + '/face_model/'
            if not exists(face_model_folder):
                try:
                    mkdir(face_model_folder)
                    #mkdir(face_folder + 'unknown')
                except Exception as e:
                    print('Error in Subj '+str(subjs)+' while creating the folder '+str(face_folder)+' file : '+str(e))
                    break

            # save_face.get_subject(num_sessions, subj, subjs, face_folder, read_folder + 'face_video/' + 'P' + subj + '/', tablet_folder, tablet_video, raspi_folder, raspi_video)
            # print("Faces Saved")
            model_train.modeling_features(face_folder, face_model_folder, feature_config['torch'])
            model_train.modeling(face_model_folder)

            # files = [f for f in listdir(read_folder) if f[1:3] == subj]
            # print(files)
            # for file in files:

            #     save_face.getfaces(str(file), read_folder, '', file, face_folder)

            if feature_config['test_model']:
                print('Testing Model ' + str(subj))
                model_test.test_subject(num_sessions, subj, subjs, read_folder + 'parent-child_vids/', face_model_folder, tablet_folder, tablet_video, raspi_folder, raspi_video, feature_config['torch'])

    def get_positions(self, main_folder, read_folder, face_3d_folder):

        onlyfiles = [f for f in listdir(read_folder) if isfile(join(read_folder, f))]
        protoPath = feature_config['protoPath']
        modelPath = feature_config['modelPath']
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        if not self.fa:
            self.get_face_estimator()

        if not exists(face_3d_folder):
            try:
                mkdir(face_3d_folder)
                mkdir(main_folder+'2d_pose/')
                mkdir(main_folder+'3d_pose/')
                mkdir(main_folder+'visibility/')
                mkdir(main_folder+'openpose/')

                mkdir(main_folder+'failed_head/')
                mkdir(main_folder+'failed_body/')
            except:
                pass

        if feature_config['openpose']:
            params = dict()
            params["model_folder"] = feature_config['openpose_models']

            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()


        for ivid in range(0,len(onlyfiles)):
            video_filename = onlyfiles[ivid]

            self.bodies = dict()
            self.frames = []
            self.head_overlap = []
            self.body_overlap = []
            self.over = False
            self.headers = False
            subject = str(video_filename[0:3]).upper()
            session = str(video_filename[4:6]).upper()

            face_model_folder = feature_config['base_folder'] + 'face_recognition/'+ subject + '/face_model/'

            if not exists(face_3d_folder + video_filename + '/'):
                try:
                    mkdir(face_3d_folder + video_filename + '/')
                    mkdir(main_folder+'2d_pose/' + video_filename + '/')
                    mkdir(main_folder+'3d_pose/' + video_filename + '/')
                    mkdir(main_folder+'visibility/' + video_filename + '/')
                    mkdir(main_folder+'openpose/' + video_filename + '/')

                    mkdir(main_folder+'failed_head/' + video_filename + '/')
                    mkdir(main_folder+'failed_body/' + video_filename + '/')
                except:
                    pass

            self.process_video(main_folder, read_folder, video_filename, face_3d_folder, face_model_folder, detector, subject, session)

            data = pd.DataFrame.from_dict(self.accuracy, orient='index', columns=['start frame', 'end frame', 'total frames', 'missed head frames', '% missed head', 'missed open', '% missed open', 'missed lift', '% missed lift'])
            data.to_csv(str(datetime.datetime.now()) + '.csv')

    def process_video(self, main_folder, read_folder, video_filename, face_3d_folder, face_model_folder, detector, subject, session):
        vs = cv2.VideoCapture(read_folder+video_filename)
        iframe = 0
        print(video_filename)
        ret, frame = vs.read()

        if (self.height, self.width) != frame.shape and feature_config['lifting']:
            self.get_pose_estimator(frame.shape[0], frame.shape[1])

        missed_head = 0
        missed_open = 0
        missed_lift = 0

        start, end = feature_config['startframe'], feature_config['endframe']
        total_frames = end-start

        while 1:
            ret, frame = vs.read()

            if (not ret):
                break

            if iframe >= start and iframe <= end:
                # show the output frame
                if iframe%feature_config['frame_skip'] == 0:
                    faces = model_test.findface(frame, feature_config['torch'], detector, face_model_folder, True)

                    preds = self.fa.get_landmarks(frame)
                    if len(preds) < 2:
                        missed_head += 1

                    if feature_config['lifting']:
                        try:
                            pose_2d, visibility, pose_3d = self.pose_estimator.estimate(frame)
                            if len(pose_3d) < 2:
                                missed_lift += 1

                            self.assign_body(faces, 2, pose_3d, frame.shape)
                            self.assign_head(preds, 2, 30, frame.shape)
                        except:
                            missed_lift += 1

                    if feature_config['openpose']:
                        datum = op.Datum()
                        datum.cvInputData = frame
                        self.opWrapper.emplaceAndPop([datum])

                        if len(datum.poseKeypoints) < 2:
                            missed_open += 1

                        self.assign_body(faces, 0, datum.poseKeypoints, frame.shape)
                        self.assign_head(preds, 0, 30, frame.shape)

                        if feature_config['openpose_display']:
                            image = datum.cvOutputData
                            for face in faces:
                                text = str(face)
                                ((startX, startY), (endX, endY)) = faces[face]
                                y = startY - 10 if startY - 10 > 10 else startY + 10

                                cv2.rectangle(image, (startX, startY), (endX, endY),
                                              (0, 0, 255), 2)
                                cv2.putText(image, text, (startX, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                            for name in self.bodies:
                                text = "{}: {:.2f}%".format(name, self.bodies[name]['confidence'] * 100)
                                cv2.putText(image, text, (self.bodies[name]['pose'][0][0], self.bodies[name]['pose'][0][1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                                if 'face' in self.bodies[name]:
                                    cv2.putText(image, name, (self.bodies[name]['face'][30][0], int(self.bodies[name]['face'][30][1] - 20)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                            cv2.imshow(str(video_filename), image)
                            key = cv2.waitKey(1) & 0xFF

                    if feature_config['continuous']:
                        self.process_frames(self.bodies, feature_config['frame_window'], self.over_frame, subject, session, main_folder, video_filename, frame.shape)

                    if feature_config['save_poses']:
                        find_face.save_face(face_3d_folder, video_filename, iframe, preds)

                        if feature_config['lifting']:
                            mypose.save_estimate(main_folder, video_filename, iframe, pose_2d, visibility, pose_3d)

                        if feature_config['openpose']:
                            data = [iframe]
                            for name in self.bodies:
                                data.append(self.bodies[name]['pose'])
                            savefolder = main_folder+'openpose/'+video_filename+'/'
                            if not exists(savefolder):
                                try:
                                    mkdir(savefolder)
                                except:
                                    print("FOLDER DON'T EXIST")
                            savefile = savefolder+str(iframe)
                            np.save(savefile, np.array(data))

            if iframe > end:
                if feature_config['save_features']:
                    self.save_features(main_folder, video_filename)
                # if feature_config['save_poses']:
                #     self.save_poses(main_folder, video_filename)\

                self.accuracy[video_filename] = [str(start), str(end), str(total_frames), str(missed_head), str(missed_head/total_frames), str(missed_open), str(missed_open/total_frames), str(missed_lift), str(missed_lift/total_frames)]

                break

            iframe += 1

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.release()

    def process_frames(self, body, window, overlap, subject, session, main_folder, video_filename, frame_size):
        head_features = head_analysis.head_sync(body, feature_config['parents'], feature_config['children'], frame_size)
        body_features = openpose_analysis.body_sync(body, feature_config['parents'], feature_config['children'], frame_size)

        self.cnt += 1

        if head_features and body_features:
            self.body_frames.append(body_features)
            self.head_frames.append(head_features)

            if self.cnt >= window - overlap or self.over:
                self.head_overlap.append(head_features)
                self.body_overlap.append(body_features)
                self.over = True

        if self.cnt == window:
            self.cnt = 0
            if len(self.head_frames) > 0 and len(self.body_frames) > 0:

                head_data = np.array(self.head_frames)
                body_data = np.array(self.body_frames)
                print("features")
                head_session_stats = head_stats.statstic_features(head_data)
                body_session_stats = body_stats.statstic_features(body_data)

                if feature_config['save_features']:
                    self.head_session_stats.append(head_session_stats)
                    self.body_session_stats.append(body_session_stats)

            self.head_frames = self.head_overlap
            self.body_frames = self.body_overlap
            self.head_overlap = []
            self.body_overlap = []

    def save_features(self, main_folder, filename):
        if feature_config['openpose']:
            headfile = feature_config['op_head_folder'] + filename 
            bodyfile = feature_config['op_body_folder'] + filename
        else: 
            headfile = feature_config['head_folder'] + filename
            bodyfile = feature_config['body_folder'] + filename

        head_stats = np.zeros((len(self.head_session_stats), feature_config['NUM_HEAD_STATS']))
        body_stats = np.zeros((len(self.body_session_stats), feature_config['NUM_BODY_STATS']))

        for i in range(len(self.head_session_stats)):
            head_stats[i,:] = self.head_session_stats[i]
            body_stats[i,:] = self.body_session_stats[i]

        body_headers = body_header.body_get_headers()
        head_headers = head_header.head_get_headers()

        np.savetxt(headfile + '_head_stats.csv', head_stats, fmt='%10.5f', delimiter=",", header=head_headers)
        np.savetxt(bodyfile + '_body_stats.csv', body_stats, fmt='%10.5f', delimiter=",", header=body_headers)
        print("SAVED")

    def save_poses(self, main_folder, filename):
        headfile = main_folder + "something/" + filename
        bodyfile = main_folder + "something/" + filename

        head_headers = ''
        body_headers = body_header.body_get_pose_headers()

        np.savetxt(headfile + '_head_poses.csv', head_poses, fmt='%10.5f', delimiter=",", header=head_headers)
        np.savetxt(bodyfile + '_body_stats.csv', body_poses, fmt='%10.5f', delimiter=",", header=body_headers)
        

    def get_features(self, body_3d_folder, face_3d_folder, body_folder, num_body_features, head_folder, num_head_features, filename):

        body_s1_stats, body_s2_stats, body_combine_stats = self.initialize_stats(feature_config['NUM_SUBJ'], feature_config['NUM_BODY_STATS'])
        head_s1_stats, head_s2_stats, head_combine_stats = self.initialize_stats(feature_config['NUM_SUBJ'], feature_config['NUM_HEAD_STATS'])

        if not exists(face_3d_folder):
            try:
                mkdir(body_folder)
                mkdir(head_folder)
            except:
                print("ERROR, these folders don't exist")

        for i in range(feature_config['ALL_SUBJ']):
            #for combined session
            body_combined_session = np.zeros((1,num_body_features))
            head_combined_session = np.zeros((1,num_head_features))

            #each session
            for j in range(1,feature_config['ALL_SESS'] + 1):
                body_s1_stats, body_s2_stats, body_combined_session = body_stats.get_stats(i, j, num_body_features, body_3d_folder, body_s1_stats, body_s2_stats, body_combined_session)
                head_s1_stats, head_s2_stats, head_combined_session = head_stats.get_stats(i, j, num_head_features, face_3d_folder, head_s1_stats, head_s2_stats, head_combined_session)

            if body_combined_session.shape[0]>1:
                body_combined_session_stats = body_stats.statstic_features(body_combined_session)
                body_combine_stats[i, :] = body_combined_session_stats

            if head_combined_session.shape[0]>1:
                head_combined_session_stats = head_stats.statstic_features(head_combined_session)
                head_combine_stats[i, :] = head_combined_session_stats

        # numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='n', header='', footer='', comments='# ', encoding=None)[source]¶
        filename = filename
        body_headers = body_header.body_get_headers()
        head_headers = head_header.head_get_headers()
        print(body_combined_session_stats.shape)
        # print(len(body_headers))

        print("SAVING")

        np.savetxt(body_folder + filename + 'body_session1_stats.csv', body_s1_stats, fmt='%10.5f', delimiter=",", header=body_headers)
        np.savetxt(body_folder + filename + 'body_session2_stats.csv', body_s2_stats, fmt='%10.5f',delimiter=",", header=body_headers)
        np.savetxt(body_folder + filename + 'body_combined_sessions_stats.csv', body_combine_stats, fmt='%10.5f', delimiter=",", header=body_headers)

        np.savetxt(head_folder + filename + 'head_session1_stats.csv', head_s1_stats, fmt='%10.5f', delimiter=",", header=head_headers)
        np.savetxt(head_folder + filename + 'head_session2_stats.csv', head_s2_stats, fmt='%10.5f',delimiter=",", header=head_headers)
        np.savetxt(head_folder + filename + 'head_combined_sessions_stats.csv', head_combine_stats, fmt='%10.5f', delimiter=",", header=head_headers)

    def extract_features(self, main_folder, body_folder, head_folder, timestamp):

        bodyfolders = [f for f in listdir(body_folder) if not isfile(join(body_folder, f))]

        headfolders = [f for f in listdir(head_folder) if not isfile(join(head_folder, f))]

        startframe = feature_config['startframe']
        endframe = feature_config['endframe']
        frame_skip = feature_config['frame_skip']

        for ivid in range(0,len(bodyfolders)):
            foldername = bodyfolders[ivid]
            print(body_folder + foldername)

            bodyPose_features = []
            for iframe in range(startframe, endframe, frame_skip):#only 2min was extracted
                #print(body_folder + foldername + '/' + str(iframe) +'.npy')
                try:
                    file = body_folder + foldername + '/' + str(iframe) +'.npy'
                    pose_3d = np.load(file)
                    body_sync_features = body_analysis.body_sync(pose_3d)
                    if body_sync_features:
                        bodyPose_features.append(body_sync_features)
                        print('BODY FRAME: ', iframe)
                    else:
                        with open('failed_body' + str(timestamp) +'.csv', 'a', newline='') as csvfile:
                            spamwriter = csv.writer(csvfile, delimiter=' ',
                                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            spamwriter.writerow([str(file),str(iframe)])

                except Exception as e:
                    #print("Unexpected error:", e)
                    #bodyPose_features.append(np.zeros(NUM_body_FEATURES))
                    pass        
            np.save(body_folder+foldername+'.npy', bodyPose_features)
            print(len(bodyPose_features))

        for ivid  in range(0,len(headfolders)):
            foldername = headfolders[ivid]
            print(head_folder + foldername)

            headPose_features = []
            for iframe in range(startframe, endframe, frame_skip):#only 2min was extracted
                try:
                    pose_3d = np.load(head_folder + foldername + '/' + str(iframe) +'.npy')
                    head_sync_features = head_analysis.head_sync(pose_3d)
                    if head_sync_features:
                        headPose_features.append(head_sync_features)
                        print('HEAD FRAME: ', iframe)
                    else:
                        with open('failed_head' + str(timestamp) +'.csv', 'a', newline='') as csvfile:
                            spamwriter = csv.writer(csvfile, delimiter=' ',
                                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            spamwriter.writerow([str(file),str(iframe)])
            
                except Exception as e:
                    # print("Unexpected error:", sys.exc_info()[0])
                    #headPose_features.append(np.zeros(NUM_HEAD_FEATURES))
                    pass   
            np.save(head_folder+foldername+'.npy', headPose_features)
            print(len(headPose_features))


def main():

    timestamp = datetime.datetime.now()

    print('Hello')
    feature_controller = Feature_Controller()
    
    # print("Creating Models for faces...")
    # feature_controller.model_faces(feature_config['ALL_SESS'], feature_config['base_folder'], feature_config['read_folder'], feature_config['tablet_folder'], feature_config['tablet_video'], feature_config['raspi_folder'], feature_config['raspi_video'])

    print("Getting Head and Body Positions...")
    feature_controller.get_positions(feature_config['main_folder'], feature_config['read_folder'], feature_config['face_3d_folder'])

    # print("Extracting features...")
    # feature_controller.extract_features(feature_config['main_folder'], feature_config['main_folder'] + '3d_pose/', feature_config['face_3d_folder'], timestamp)

    # print("Getting feature stats...")
    # feature_controller.get_features(feature_config['main_folder'] + '3d_pose/', feature_config['face_3d_folder'], feature_config['body_folder'], feature_config['NUM_BODY_FEATURES'], feature_config['head_folder'], feature_config['NUM_HEAD_FEATURES'], "")

if __name__ == "__main__": main()