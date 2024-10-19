
import cv2
import numpy as np
import os, sys
import subprocess
from subprocess import check_output, call
from platform import python_version

from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
import operator
import csv
import six
import traceback
import logging
import shlex
import time
import glob
import tempfile
# from combine_offline_app import prediction
# from all_three import prediction
# from only_yolo8 import prediction
# from all_clip_yolo_vqa import prediction
# from all_three_multiple_persons import prediction
from phone.modeling_phone_detection import prediction
# from only_clip import prediction



# logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging
path = os.getcwd()

# new_baseline_list = []
# cv2.ocl.setUseOpenCL(False)
# dlib_face_detector = None


# the offline class will divide the video into parts and generate offline_worker to run them.
class Offline:

    def __init__(
            self,
            # model_image,  # a list of image
            # onboarding_model_image,  # list of most recent onboarding images
            session_video_path,  # a string where the target video is stored
            # monitoring_snapshots,  # list of monitoring snaps
            # face_detector,  # an face detector instance. it must have a method 'track'
            # face_landmark_predictor_file,
            fps=1,  # how many frames to analyze per second.
            incident_type=[],
            settings={}):

        self.session_video_path = session_video_path
        # self.face_detector = face_detector
        # self._is_facial_model_correct = True
        # self._is_onbording_facial_model_correct = True
        # self.histogram_analyser = MatchHistogram()
        # self.face_landmark_predictor = dlib.shape_predictor(face_landmark_predictor_file)
        # self.head_angles_facescan = np.asarray([90, 90, 90])
        # self.head_angles_onboarding_facescan = np.asarray([90, 90, 90])
        # self.model_image = []  # detected face container
        # self.model_image_confidence = []  # detected face confidence container
        # self.model_image_pose = []  # detected face pose index (frantal = 3.0, right = 2.0 ,left = 1.0) container
        # self.face_positions_in_facescan = []
        # self.face_shape_in_facescan = [],
        # self.face_shape_in_onboarding_facescan = [],
        # self.onboarding_model_image = []
        # self.onboarding_model_image_confidence = []
        # self.onboarding_model_image_pose = []
        # self.face_positions_in_onboarding_facescan = []
        self.video_length = 0
        self.all_video_piece_list = None
        self.current_video_piece_list = None

        # for image in model_image:
        #     if isinstance(image, six.string_types):
        #         # image is a string. Hence, its a file-path and not a numpy-array
        #         image = cv2.imread(image)

        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     rects, score, pose_index = self.face_detector.detect_hog(image, 1)

        #     if rects.__len__() < 1:
        #         rects, score = self.face_detector.detect_cnn(image, 1)
        #         pose_index = [None]

        #     if rects.__len__() == 1:
        #         self.face_positions_in_facescan.append(rects[0])
        #         self.model_image.append(image)
        #         self.model_image_confidence.append(score[0])
        #         self.model_image_pose.append(pose_index[0])

        #     elif rects.__len__() > 1:
        #         # if we have two or more than two faces in facescan then we are selecting one which have
        #         # large area , assuming that user is more closer to camera
        #         rects_array = np.asarray(rects)
        #         face_area = []
        #         for det in (rects_array):
        #             face_area.append((det[2] - det[0]) * (det[3] - det[1]))

        #         max_face_index = np.argmax(face_area)
        #         self.model_image.append(image)
        #         self.model_image_confidence.append(score[max_face_index])
        #         self.model_image_pose.append(pose_index[max_face_index])
        #         self.face_positions_in_facescan.append(rects[max_face_index])

        #     if score.__len__() < 1:
        #         self.face_positions_in_facescan.append([])
        #         self.model_image_confidence.append(None)
        #         self.model_image_pose.append(None)

        # self.face_shape_in_facescan = self.get_ypr(model_image, self.face_positions_in_facescan)

        # if onboarding_model_image is not None:
        #     for image in onboarding_model_image:
        #         if isinstance(image, six.string_types):
        #             # image is a string. Hence, its a file-path and not a numpy-array
        #             image = cv2.imread(image)

        #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #         rects, score, pose_index = self.face_detector.detect_hog(image, 1)

        #         if rects.__len__() < 1:
        #             rects, score = self.face_detector.detect_cnn(image, 1)
        #             pose_index = [None]

        #         if rects.__len__() == 1:
        #             self.face_positions_in_onboarding_facescan.append(rects[0])
        #             self.onboarding_model_image.append(image)
        #             self.onboarding_model_image_confidence.append(score[0])
        #             self.onboarding_model_image_pose.append(pose_index[0])

        #         elif rects.__len__() > 1:
        #             # if we have two or more than two faces in facescan then we are selecting one which have
        #             # large area , assuming that user is more closer to camera
        #             rects_array = np.asarray(rects)
        #             face_area = []
        #             for det in (rects_array):
        #                 face_area.append((det[2] - det[0]) * (det[3] - det[1]))

        #             max_face_index = np.argmax(face_area)
        #             self.face_positions_in_onboarding_facescan.append(rects[max_face_index])
        #             self.onboarding_model_image.append(image)
        #             self.onboarding_model_image_confidence.append(score[max_face_index])
        #             self.onboarding_model_image_pose.append(pose_index[max_face_index])

        #         if score.__len__() < 1:
        #             self.face_positions_in_onboarding_facescan.append([])
        #             self.onboarding_model_image_confidence.append(None)
        #             self.onboarding_model_image_pose.append(None)

        #     self.face_shape_in_onboarding_facescan = self.get_ypr(onboarding_model_image,
        #                                                           self.face_positions_in_onboarding_facescan)

        # if self.model_image.__len__() == 0:
        #     logger.info("Task: get_incidents_from_video. Offline object initialization. "
        #                  "No faces are detected in snapshots")
        #     self._is_facial_model_correct = False

        # if self.onboarding_model_image.__len__() == 0:
        #     logger.info("Task: get_incidents_from_video. Offline object initialization. "
        #                  "No faces are detected while onboarding")
        #     self._is_onbording_facial_model_correct = False

        self.fps = fps
        self.type = incident_type
        self.file_list = []
        # self.final_track_list = []

        self.images_path = []
        # logger.info("Himanshu: Created dlib's shape predictor.")
        # self.head_angles_output = np.asarray([90, 90, 90])  # Just an initialization. We will use this list for
        # head_pose_estimation
        # default settings

        # pd_prototxt_path = path + os.path.sep + 'visiontasks' + os.path.sep + 'deep_models/ssdlitefpn_mobilenet.pbtxt'
        # pd_model_path = path + os.path.sep + 'visiontasks' + os.path.sep + 'deep_models/ssdlitefpn_mobilenet.pb'

        # if os.path.isfile(pd_prototxt_path) is False:
        #     raise IOError('Cannot read file ' + pd_prototxt_path)
        # if os.path.isfile(pd_model_path) is False:
        #     raise IOError('Cannot read file ' + pd_model_path)

        # face_pose_classifier_path = path + os.path.sep + 'visiontasks' + os.path.sep + 'deep_models/' \
        #                                                                                'face_pose_classifier_tf2_v3'
        # if not os.path.isfile(face_pose_classifier_path) and not os.path.exists(face_pose_classifier_path):
        #     raise IOError('Cannot read file ' + face_pose_classifier_path)

        # if person_detector_type == 'YOLO':
        #     pd_net = {}
        #     wt = path + os.path.sep + 'visiontasks' + os.path.sep + "deep_models/yolo_tiny_improved_mobile_5_may_21/enet-coco-train_8000.weights"
        #     cfg = path + os.path.sep + 'visiontasks' + os.path.sep + "deep_models/yolo_tiny_improved_mobile_5_may_21/enet-coco-train-higher-res.cfg"
        #     pd_net['net'] = cv2.dnn.readNet(wt, cfg)
        #     layer_names = pd_net['net'].getLayerNames()
        #     pd_net['output_layers'] = [layer_names[i[0] - 1] for i in pd_net['net'].getUnconnectedOutLayers()]
        # elif person_detector_type == 'YOLOv5':
        #     pd_net = cv2.dnn.readNetFromONNX(path + os.path.sep + 'visiontasks' + os.path.sep +
        #                                             "deep_models/Yolov5/yolov5_general_class_22_jan_23.onnx")
        # elif person_detector_type == 'TF':
        #     pd_net = tf.saved_model.load("visiontasks/deep_models/SSD_Resnet_640X640_V2/saved_model")
        # else:
        #     pd_net = None

        self.settings = dict(
            ALL_TEMP_PATH=settings['ALL_TEMP_PATH'],
            # COMPUTE_HEAD_ANGLES=settings['COMPUTE_HEAD_ANGLES'],
            LOG_LEVEL=settings['LOG_LEVEL'],
            PREVIEW_IMAGE_PATH=settings['PREVIEW_IMAGE_PATH'],
            # BG_MOTION_MODEL=None,
            # FACE_TRACKER_OBJ=None,
            # PERSON_TRACKER_OBJ=None,
            TESTSESSION_ID=settings['TESTSESSION_ID'],
            VIDEO_PIECE_ID=None,
            ENABLE_PERSON_SEGMENTER=False,
            # PERSON_SEGMENTER=None,
            # PERSON_DETECTOR=pd_net,
            # PERSON_DETECTOR_TYPE=person_detector_type,
            # BM_WITH_TF_MODEL=True,
            # LOAD_FACE_POSE_WITH_OPENCV=True if os.path.isfile(face_pose_classifier_path) else False,
            # FACE_POSE_DETECTOR=None,
            # FSLA_WITH_FACE_POSE_CLASSIFIER=False,
            # PERSON_DETECTOR_SSDLITEFPN=cv2.dnn.readNetFromTensorflow(pd_model_path, pd_prototxt_path),
            # DLIB_FEATURE_EXTRACTOR=get_feature_extractor(),
            # new parameter since 'terminatable'
            # ALL_PROCESSER               = 1,
            VIDEO_PIECE_LENGTH=1200 * fps,  # length 20min x fps
            # VIDEO_PIECE_STEP=1140 * fps,  # step of 19min x fps #overlap of 1min
            VIDEO_PIECE_STEP=1200 * fps,
            # 'terminatable'

            # ANALYSE_INPUTS=None,

            # BACKGROUND_MOTION_THRESHOLD=0.25,
            # BACKGROUND_MOTION_PIXELS=12,

            # WRONG_FACE_THRESHOLD_DLIB=0.58,
            # WRONG_FACE_THRESHOLD=0.68,
            # WRONG_FACE_THRESHOLD_VGG=0.69,
            # WRONG_FACE_IN_FACESCAN_THRESHOLD_VGG=0.71,
            # WRONG_FACE_SPEEDUP=10,  # Consecutive frames are not picked to match the faces.
            # FACE_MATCH_THRESHOLD=5 * fps,  # 5 secs between two attempts of face-matching
            # WRONG_FACE_MIN_FRAMES=3,
            # NUMBER_BASELINE_ALLOWED=50,

            FILTER=dict(
                NO_FACE=(3 * fps, .5, 10 * fps, 5), 
                MULTIPLE_FACES=(3, .5, 10 * fps, 0),
                WRONG_FACE=(3 * fps, .5, 10 * fps, 3),
                # (filter_length, thresh, combine_closer_than,remove_shorter_than)
                BACKGROUND_MOTION=(3, .5, 10 * fps, 0),
                # DEFAULT=(1, 1, 0, 0),
                DEFAULT=(0, 1, 0, 0),
            ),
        )

        # for setting in settings:
        #     if setting not in self.settings:
        #         logger.warning(str(setting) + str(" is not a keyword in settings"))

        # self.settings.update(settings)

        # self.frame_list = monitoring_snapshots
        # self.frames_all = monitoring_snapshots
        self.result = None
        self.certainity = None
        self.all_result = []
        self.all_certainty = []
        self.all_video_length = 0


    def get_tasks(self):
        """
        Each task corresponds to the processing of a video chunk. Chunks are processed sequentially.
        :return: Num of video chunks to be processed.
        """
        # some code to divide the video
        if self.session_video_path:
            self.video_length = self.get_video_length()
            self.current_video_piece_list = self.generate_video_pieces()
            self.frame_list = None
        else:
            self.video_length = int(len(self.frame_list) / self.fps)
            # Following setting enforces to process all snapshots in one go.
            # Hence self.video_piece_list will have only one element
            self.settings['VIDEO_PIECE_LENGTH'] = len(self.frame_list)
            self.settings['VIDEO_PIECE_STEP'] = len(self.frame_list)
            self.current_video_piece_list = [(0, len(self.frame_list) - 1)]  # self.generate_video_pieces()

        if self.all_video_piece_list is not None:
            offset_to_add = self.all_video_piece_list[-1][0] + self.all_video_piece_list[-1][1]
            list_to_extend = [(x[0] + offset_to_add, x[1]) for x in self.current_video_piece_list]
            self.all_video_piece_list.extend(list_to_extend)
        else:
            self.all_video_piece_list = self.current_video_piece_list

        self.all_video_length = self.all_video_piece_list[-1][0] + self.all_video_piece_list[-1][1]

        self.result = [None] * self.current_video_piece_list.__len__()
        self.certainity = [None] * self.current_video_piece_list.__len__()

        return self.current_video_piece_list.__len__()

    def run(self, video_piece_id, analyse_data=None):
        """
        This function passes the video to the processor.
        :param video_piece_id: sequence number of the chunk to be processed
        :return:
        """

        try:  # basically catch wrong id
            # **** TERMINTABLE ****
            # check savedata
            # if exists, read it and return
            # filename:
            # $(setting['ALL_TEMP_PATH'])/$(video_name)_$(fps)_$(start)_$(length)
            self.settings['VIDEO_PIECE_ID'] = video_piece_id
            # self.settings['ANALYSE_INPUTS'] = analyse_data
            video_name = os.path.basename(self.session_video_path)
            save_name = self.settings['ALL_TEMP_PATH'] + '/' + video_name + '_' + str(self.fps) + '_' \
                        + str(self.current_video_piece_list[video_piece_id][0]) + '_' \
                        + str(self.current_video_piece_list[video_piece_id][1])
            # print ('save_name', save_name)
            self.file_list.append(save_name)
            # try:
            #     data = self._load_data_from_file(save_name)
            #     self.result[video_piece_id] = data
            #     # print ('data read from tmp file, over')
            #     return True
            # except:  # if the file is not readable
            #     # print ('no data read from tmp file')
            #     pass

            # print ('start computing')
            # else
            # do:
            # *********************
            if self.result[video_piece_id] is None:
                # some code to process that video
                offline_worker = OfflineWorker()

                # set onboarding model to none after first chunk processed to avoid reprocessing same task
                # if video_piece_id > 0:
                #     self.onboarding_model_image = None

                if self.session_video_path:
                    gf_st = time.time()
                    self.frames_all = self.generate_frames(start_frame=self.current_video_piece_list[video_piece_id][0],
                                                           length=self.current_video_piece_list[video_piece_id][1],
                                                           video_piece_id=video_piece_id)
                    if self.settings['LOG_LEVEL'] == '1':
                        logging.info(f"Total frames extracted: {len(self.frames_all)}")
                        logging.info("TestSession: {0} | Time taken to extract frames is {1} secs.".format(
                            self.settings['TESTSESSION_ID'], time.time() - gf_st))
                    # self.frame_list = self.frames_all[::3]
                    # print(len(self.frame_list), len(self.frames_all))
                # fps is set to 1 so that it returns frames instead of time.
                # self.result[video_piece_id] = offline_worker.detect_violations(self.frames_all, self.fps,
                #                                                                self.model_image,
                #                                                                self.face_shape_in_facescan,
                #                                                                self.onboarding_model_image,
                #                                                                self.face_shape_in_onboarding_facescan,
                #                                                                self.model_image_pose,
                #                                                                self.face_detector,
                #                                                                self.face_landmark_predictor,
                #                                                                self.type, self.settings,
                #                                                                self.histogram_analyser).copy()

                # CLIP Logic
                self.result[video_piece_id] = prediction(video_piece_id, self.frames_all)
                logger.info(f"Results for video_piece_id {video_piece_id} : {self.result[video_piece_id]}")

                delete_path = self.frames_all[0].split("1.jpg")[0]
                # face_pos = offline_worker._face_position
                # self.frame_list = offline_worker.frame_list
                if self.video_length == 0:
                    self.video_length = offline_worker.frame_list.__len__()
                    self.current_video_piece_list[0][1] = self.video_length

                # if self.settings['COMPUTE_HEAD_ANGLES']:
                #     self.update_head_angles_output(offline_worker.head_angles_output, video_piece_id)

                # self.settings['BG_MOTION_MODEL'] = offline_worker.bg_motion_model

                # self.settings['FACE_TRACKER_OBJ'] = offline_worker.face_tracker_obj
                # self.settings['PERSON_TRACKER_OBJ'] = offline_worker.person_tracker_obj
                # if self.settings['PERSON_SEGMENTER'] is None:
                #     self.settings['PERSON_SEGMENTER'] = offline_worker.person_segmenter
                # if self.settings['PERSON_DETECTOR'] is None:
                #     self.settings['PERSON_DETECTOR'] = offline_worker.person_detector
                # exit()
                # FIXME # HACK
                # somehow i have to delete all the members before i delete the instance
                # only by doing this can the memery be successfully freed.
                # del offline_worker._frame_feature
                # del offline_worker.frame_list[:]
                # del offline_worker.frames_all[:]
                # del offline_worker._face_position
                # del offline_worker._face_count
                # del offline_worker._track_list
                # del offline_worker._reverse_track_list
                # del offline_worker._model_image_feature
                # del offline_worker._onboarding_model_image_feature
                # del offline_worker._face_list

                # del offline_worker.model_image
                # del offline_worker.onboarding_model_image
                # del offline_worker.face_detector
                # del offline_worker.function_mapping
                # del offline_worker.settings

                del offline_worker
                # print('memory freed')
                # **** TERMINATABLE ****
                return True
            else:
                logger.warn('task #' + str(video_piece_id) + ' has been processed already!')
                return False
        except Exception as e:
            logging.exception("TestSession: {0} | cannot find task id: {1} Error : {2}".format(
                self.settings['TESTSESSION_ID'], video_piece_id, e))
            # print(str(e))
            traceback.print_exception(type(e), e, e.__traceback__)
            return False


    def get_result(self, create_preview_image=False):
        """
        This function filters the initial results and discard much shorter incidents.
        :return: Filtered incidents
        """
        if None in self.result:
            return None

        incidents = {}
        certainity = {}
        self.all_result.extend(self.result)
        self.all_certainty.extend(self.certainity)
        preview_images_dict = {}
        # self.settings['PREVIEW_IMAGE_PATH'] = "/tmp//video_incident_previews_ajeet/12345_-2/"
        preview_imgs_tmp_path = self.settings['PREVIEW_IMAGE_PATH']

        preview_imgs_incident_folder = preview_imgs_tmp_path + 'temp_images' + '/'

        if create_preview_image:
            call(shlex.split('mkdir -p ' + preview_imgs_tmp_path), shell=False)
            call(shlex.split('mkdir -p ' + preview_imgs_incident_folder), shell=False)

        # strategy
        # if one frame is good (no incidents) in at least one video piece
        # it is reported safe.
        for incident in self.type:
            # print (incidents)
            # print ('start', incident)
            if incident in self.all_result[0]:
                incident_list = [True] * self.all_video_length  # 1 list (of Ture/False) which includes all video pieces
                time_stamps = []  # 1 list of incidents (start, end) which includes all video pieces
                confidence_measures = []  # 1 list of certainities which includes all video pieces
                for video_id in range(self.all_video_piece_list.__len__()):
                    starting = 0
                    offset = self.all_video_piece_list[video_id][0]
                    piece_length = self.all_video_piece_list[video_id][1]
                    incident_list[offset:offset + piece_length] = [True] * piece_length
                    for idx, time_period in enumerate(self.all_result[video_id][incident][0]):
                        ending = int(time_period[0])
                        incident_list[starting + offset:ending + offset] = [False] * (ending - starting)
                        start_inci = (time_period[0] + offset) / self.fps
                        end_inci = (time_period[1] + offset) / self.fps
                        time_stamps.append((start_inci, end_inci))
                        try:
                            confidence_measures.append(self.all_result[video_id][incident][1][idx])
                        except:
                            confidence_measures.append(0)
                        starting = int(time_period[1] + 1)
                    incident_list[starting + offset:offset + piece_length] = [False] * (piece_length - starting)
                    # print ('after video_id', video_id)
                    # print (incident_list)
                # incident_list = frame_filter(incident_list, 30, .5)
                if incident in self.settings['FILTER']:
                    incident_list = frame_filter(incident_list,
                                                 self.settings['FILTER'][incident][0],
                                                 self.settings['FILTER'][incident][1])
                else:
                    incident_list = frame_filter(incident_list,
                                                 self.settings['FILTER']['DEFAULT'][0],
                                                 self.settings['FILTER']['DEFAULT'][1])

                incidents[incident] = convert_to_time(incident_list, self.fps)
                # since frame_filter() function modifies the incidents, we need to modify
                # confidence_measures accordingly
                certainity[incident] = compare_incidents(incidents[incident], time_stamps, confidence_measures)

                if incident in self.settings['FILTER']:
                    frame_filter2(incidents[incident], certainity[incident],
                                  self.settings['FILTER'][incident][2], self.fps)
                    remove_shorter_than(incidents[incident], certainity[incident],
                                        self.settings['FILTER'][incident][3])
                else:
                    frame_filter2(incidents[incident], certainity[incident],
                                  self.settings['FILTER']['DEFAULT'][2], self.fps)
                    remove_shorter_than(incidents[incident], certainity[incident],
                                        self.settings['FILTER']['DEFAULT'][3])

                if create_preview_image:
                    preview_images_dict[incident] = create_preview_image_from_incident(incident,
                                                    incidents[incident], self.settings['ALL_TEMP_PATH'],
                                                    self.settings['ENABLE_PERSON_SEGMENTER'],
                                                    self.settings['TESTSESSION_ID'], self.settings['LOG_LEVEL'],
                                                                                       preview_imgs_tmp_path)

        # FSLA
        # if self.settings['COMPUTE_HEAD_ANGLES']:
        #     incidents['FSLA'], certainity['FSLA'] = self.generate_violations_pose_index()
        #     if create_preview_image:
        #         preview_images_dict['FSLA'] = create_preview_image_from_incident('FSLA',
        #                                                                          incidents['FSLA'],
        #                                                                          self.settings['ALL_TEMP_PATH'],
        #                                                                          self.settings['ENABLE_PERSON_SEGMENTER'],
        #                                                                          self.settings['TESTSESSION_ID'],
        #                                                                          self.settings['LOG_LEVEL'],
        #                                                                          preview_imgs_tmp_path)
        #         call(shlex.split('rm -rf ' + preview_imgs_incident_folder), shell=False)
        return incidents, certainity, preview_images_dict

    # def __del__(self):
    #     """
    #     Finally cleans-up any remaining files
    #     :return:
    #     """
    #     # some code to deconstruct
    #     # print 'del is called'
    #     # clean screen shots
    #     # clean dictionary
    #     '''
    #     try:
    #         if os.path.exists(self.session_video_path):  # clean-up /tmp
    #             # os.remove(self.session_video_path)
    #             logger.info("Himanshu: Deleted the video file: %s"% str(self.session_video_path))
    #     except:
    #         logger.info("Himanshu: Unable to delete the video file: %s"% str(self.session_video_path))
    #     '''
    #     for filename in self.file_list:
    #         try:
    #             if os.path.exists(filename):  # clean-up /tmp
    #                 os.remove(filename)
    #         except:
    #             # print ('cannot delete file '+filename)
    #             pass
    #     # delete image files
    #     try:
    #         if os.path.exists(self.settings['ALL_TEMP_PATH']):
    #             call(shlex.split('rm -rf ' + self.settings['ALL_TEMP_PATH']), shell=False)
    #     except Exception as e:
    #         msg = 'TestSession: {0} | Error while deleting folder: {1}'.format(
    #             self.settings['TESTSESSION_ID'], e)
    #         logger.info(msg)
    # # def __exit__(self):
    # #   print ('exit is called')

    def get_video_length(self):
        """
        This function computes the duration of the monitoring video.
        Duration is used to compute the number of chunks to be processed.
        :return: total number of frames to be processed = fps x duration_in_seconds
        """
        try:
            args = ['ffprobe', '-loglevel', 'panic',  '-i', self.session_video_path, '-show_format', '-v', 'quiet']
            args2 = ['sed', '-n', '''s/duration=//p''']
            process_curl = subprocess.Popen(args, stdout=subprocess.PIPE,
                                            shell=False)
            process_wc = subprocess.Popen(args2, stdin=process_curl.stdout,
                                          stdout=subprocess.PIPE, shell=False)
            # Allow process_curl to receive a SIGPIPE if process_wc exits.
            process_curl.stdout.close()

            duration_secs = process_wc.communicate()[0]

            if type(duration_secs) == str:
               duration_secs = int(duration_secs.split('.')[0])
            else:
               duration_secs = int(duration_secs.decode("utf-8").split('.')[0])
        except Exception as e:
            logging.error("TestSession: {0} |  Exception while determining video length -> Video ID: {1} Error : "
                          "{2}".format(self.settings['TESTSESSION_ID'], self.settings['VIDEO_PIECE_ID'], e))
            duration_secs = 0

        return self.fps * duration_secs

    # extract $(fps) frames per second from the video and save them to $(self.settings['ALL_TEMP_PATH'])
    # return the frames extracted
    # if error, raise Exception (and return 0)
    def generate_frames(self, start_frame=None, length=None, video_piece_id=0):
        """
        This function extract frames belonging to a particular video chunk. Enhances the contrast of
        images, reads them into the memory and deletes from the tempporary path.
        :param start_frame: index of the start of chunk
        :param length: length of the chunk
        :return: list of images (data type: list of numpy.ndarray elements)
        """
        # if fps is 0:
        #     fps = self.fps
        # if self.settings['ENABLE_PERSON_SEGMENTER']:
        #     fps = 9
        #     # fps = 3
        # else:
        #     fps = 3
        #     # fps = 1

        fps = 1

        tmp = self.settings['ALL_TEMP_PATH']
        video_filename = os.path.basename(self.session_video_path)

        frame_list = []
        for frame_num in range(start_frame, start_frame + length):
            filename = tmp + str(video_piece_id) + '_' + str(frame_num) + '.png'
            if os.path.exists(filename):
                frame = cv2.imread(filename)
                img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                # equalize the histogram of the Y channel
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                # convert the YUV image back to RGB format
                frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                frame_list.append(frame)
        if frame_list:
            return frame_list

        if not os.access(tmp, os.W_OK):
            raise Exception(tmp + " is not writable")
            return 0

        # delete existing images
        all_img_filenames = tmp + str(video_piece_id) + '_*.jpg'
        call(shlex.split("rm " + all_img_filenames), shell=False)

        if start_frame is not None and length is not None:
            if self.settings['ENABLE_PERSON_SEGMENTER']:
                cmd = "ffmpeg -loglevel panic -ss " + str(3 * start_frame / fps) + " -i " + self.session_video_path + " -t " \
                      + str((3 * length) / fps) + " -r " + str(fps) + \
                      " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'
            else:
                cmd = "ffmpeg -loglevel panic -ss " + str(start_frame / fps) + " -i " + self.session_video_path + " -t " \
                  + str(length / fps) + " -r " + str(fps) + \
                  " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'
        else:
            cmd = "ffmpeg -i " + self.session_video_path + " -r " + \
                  str(fps) + " -qscale:v 5 " + tmp + str(video_piece_id) + '_%d.jpg'

        runcmd = call(shlex.split(cmd), shell=False)

        if runcmd != 0:
            raise Exception('failed to run ffmpeg. Possible reasons: ffmpeg not in path, or disk is full, or '
                            + self.session_video_path + ' is not accessible')
            return 0

        frame_list = []
        frameid = 1  # first frame is 0, but 1 in disk, so '+1'
        if self.settings['ENABLE_PERSON_SEGMENTER']:
            num_img_files_extracted = int(length * 3)
        else:
            num_img_files_extracted = int(length)
        # logging.info("TestSession {0} | Start {1}. Length {2}. Num_imgs: {3}"
        #              .format(self.settings['TESTSESSION_ID'], start_frame, length, num_img_files_extracted))
        while frameid <= num_img_files_extracted:
            filename = tmp + str(video_piece_id) + '_' + str(frameid) + '.jpg'
            # frame = cv2.imread(filename)
            # if frame is None:
            #     break
            if os.path.exists(filename):
                frame_list.append(filename)

            frameid += 1

        return frame_list

    # based on current settings,
    # generate a list,
    # list = [(start1, length1), (start2, length2) ...]
    # each element in the list the video pieces
    # THE FIRST FRAME IS FRAME #0
    def generate_video_pieces(self):
        """
        This function computes the indices using which a video-chunk will be extracted
        from the monitoring video
        :return:
        """

        video_pieces = []
        now = 0
        length = self.settings['VIDEO_PIECE_LENGTH']
        step = self.settings['VIDEO_PIECE_STEP']

        if self.video_length == 0:
            video_pieces = [(0, -1)]  # [(start_time, length)]
            return video_pieces

        while True:
            if now + length > self.video_length:
                read_length = self.video_length - now
            else:
                read_length = length

            video_pieces.append((now, read_length))
            if now + read_length >= self.video_length:
                break
            else:
                now += step

        '''
        while now+length < self.video_length+step:
            #prevent too long that will cause read frames after video ends
            read_length = length
            if now+length > self.video_length:
                read_length = self.video_length - now
            video_pieces.append((now, read_length))
#            now += self.settings['VIDEO_PIECE_STEP']
            now += step
        '''
        return video_pieces


class OfflineWorker:

    def __init__(self):
        """
        Basic initialization
        :return:
        """
        # input parameter
        # self.model_image = None
        # self.model_image_face_shape = None
        # self.onboarding_model_image_face_shape = None
        # self.onboarding_model_image = None
        # self.model_pos_hog = None
        self.frame_list = None
        # self.face_detector = None
        # global new_baseline_list
        # print('[Info] Baseline length: {}'.format(len(new_baseline_list)))
        # self._tfgraph = tf.compat.v1.get_default_graph() if tf.__version__[0] == '2' else tf.get_default_graph()
        # self.face_landmark_detector = None
        # self.head_angles_output = [90, 90, 90]  # Just an initialization. We will use this list for head_pose_estimation
        # self.is_head_angles_computed = False
        # self._no_motion_list = None
        # self._bg_motion_list = None
        # self.unchecked_frames = []
        # self._bg_motion_confidence = None
        # self.landmark_shape = []
        # # private properties
        # self._face_position = None
        # self._face_pose_hog = None
        # self._face_confidence_hog = None

        self._face_count = None  # list
        # face_count[i] indicates number of faces
        # found in frame #i by face detector.
        # self._track_list = None  # list
        # track_list[i] = {0, 1}
        # 1 indicates that the face in
        # frame i is still trackable.
        # otherwise, ...

        # self._reverse_track_list = None  # just create another list now
        # but finally it should be something like
        # each frame has a list that it can be tracked from
        # say: _new_track_list[10] = [9, 11, 15]
        # means frame 10 can be tracked from #9, #11 and #15.
        # then do an union find algorithm to find some stable time periods
        # but now, just two lists.
        # TODO

        # self._model_image_feature = None
        # self._onboarding_model_image_feature = None
        # self._face_list = None
        # self._detected_or_tracked = []
        # self.facescan_imposer_score = None
        # self._frame_feature = None
        # self._fps = None
        # self.bg_motion_model = None
        # self.face_tracker_obj = None
        # self.person_tracker_obj = None
        # self.person_segmenter = None
        # self.person_detector = None
        # self.person_detector_frames = None
        # self.person_detector_ssd_light_fpn = None
        # register the keyword in type and the function to handle it
        # self.function_mapping = dict(
        #     NO_FACE=self.analyzer_no_face,
        #     MULTIPLE_FACES=self.analyzer_multiple_faces,
        #     WRONG_FACE=self.analyzer_wrong_face,
        #     BACKGROUND_MOTION=self.analyzer_background_motion,
        #     WRONG_FACE_FACESCAN=self.analyzer_wrong_face_with_onboarding,
        # )

    #    def __del__(self):
    #        print ('worker __del__() is called')

    #    def __exit__(self):
    #        print ('worker __exit__() is called')

    # main method
    # check google doc for documentation

    # Extract every frame form given video.
    # return a list that contains every frame in the order
    def extractFrame(self, video_path, fps, start_frame=None, length=None):
        """
        This function extract frames belonging to a particular video chunk. Enhances the contrast of
        images, reads them into the memory and deletes from /tmp/.
        :param video_path: video file path
        :param fps: frame rate
        :param start_frame: index of the start of chunk
        :param length: length of the chunk
        :return: list of images
        """
        tmp = self.settings['ALL_TEMP_PATH'] + '/'
        video_filename = os.path.basename(video_path)
        #        if os.access(tmp, os.W_OK) == False:
        #            raise Exception(tmp + "is not writable")
        #            return None
        #
        if start_frame and length:
            cmd = "ffmpeg -loglevel panic -ss " + str(start_frame / fps) + " -i " + video_path + " -to " \
                  + str((start_frame + length) / fps) + " -r " + str(fps) + " " + tmp + video_filename + '_%d.png'
        elif length == -1:
            cmd = "ffmpeg -loglevel panic -ss " + str(start_frame / fps) + " -i " + video_path \
                  + " -r " + str(fps) + " " + tmp + video_filename + '_%d.png'
        else:
            cmd = "ffmpeg -loglevel panic " + " -i " + video_path + " -r " + str(fps) + " " \
                  + tmp + video_filename + '_%d.png'
        # print (cmd, start_frame, length)
        runcmd = call(shlex.split(cmd), shell=False)
        if runcmd != 0:
            logging.error("TestSession: {0} |  Error while extracting frames with ffmpeg: -> Video ID: {1} "
                          "Video exist status: {2}".format(self.settings['TESTSESSION_ID'],
                                                           self.settings['VIDEO_PIECE_ID'], os.path.exists(video_path)))
            return []

        frame_list = []
        frameid = 1  # first frame is 0, but 1 in disk, so '+1'
        while frameid <= length:
            filename = tmp + video_filename + '_' + str(frameid) + '.png'
            frame = cv2.imread(filename)
            if frame.size == 0:
                break
            # frame = cv2.flip(frame, 1)

            img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            # equalize the histogram of the Y channel
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

            # convert the YUV image back to RGB format
            frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            '''
            frame[:,:,0] = cv2.equalizeHist(frame[:,:,0]) #histogram equalization for all three channels-BGR
            frame[:,:,1] = cv2.equalizeHist(frame[:,:,1])
            frame[:,:,2] = cv2.equalizeHist(frame[:,:,2])
            '''
            frame_list.append(frame)
            frameid += 1
            # if frameid <= start_frame + self.settings['VIDEO_PIECE_STEP'] + 1:
            os.remove(filename)  # let caller's destroctor remove the file
        # print ('frame_list.__len__()', frame_list.__len__())
        return frame_list


def convert_to_time(list, fps):
    time_list = []

    # add False to starting point and ending point of list
    newlist = [False] + list + [False]
    for frame_id in range(0, newlist.__len__() - 1):
        if newlist[frame_id] == False and newlist[frame_id + 1] == True:
            time_start = frame_id
        elif newlist[frame_id] == True and newlist[frame_id + 1] == False:
            time_list.append((time_start / float(fps), (frame_id - 1) / float(fps)))
    return time_list


def frame_filter(frame_list, filter_length, filter_threshold):
    half_length = int(filter_length / 2)
    filter_length = half_length * 2 + 1
    frame_length = frame_list.__len__()

    result = []

    if filter_length > frame_length:
        if sum(frame_list) >= frame_length * filter_threshold:
            result = [True] * frame_length
        else:
            result = [False] * frame_length
    else:
        now_sum = sum(frame_list[:filter_length])
        if now_sum >= filter_length * filter_threshold:
            result.append(True)
        else:
            result.append(False)
        for i in range(filter_length, frame_length):
            now_sum += frame_list[i] - frame_list[i - filter_length]
            # print (now_sum)
            if now_sum >= filter_length * filter_threshold:
                result.append(True)
            else:
                result.append(False)
            # print (result)
        result = [result[0]] * half_length + result + [result[-1]] * half_length

    return result


def compare_incidents(incidents, time_stamps, confidence_measures):
    """
    This function compares filtered incidents with the old ones (time-stamps) in order to modify confidence measures
    if required.
    :param incidents:
    :param time_stamps:
    :param confidence_measures:
    :return:
    """
    new_confidence_measures = []
    for incident in incidents:
        time_to_be_combined = []
        confidence_to_be_combined = []
        for i, ts in enumerate(time_stamps):
            if (min(ts[1], incident[1]) - max(ts[0], incident[0])) / (ts[1] - ts[0] + 0.00000001) >= 0.9:
                # if a previous incident is overlapping (>90%) with the new one,
                # use it while computing new confidence measure.
                time_to_be_combined.append(ts[1] - ts[0])
                confidence_to_be_combined.append(confidence_measures[i])
        # compute weighted average of confidence measures
        new_conf_value = sum([float(i) * float(j) for i, j in zip(time_to_be_combined, confidence_to_be_combined)]) \
                         / (sum(time_to_be_combined) + 0.000001)
        new_confidence_measures.append(new_conf_value)

    return new_confidence_measures


def frame_filter2(frame_list, certainity, length_threshold, fps):
    """
    This filter combines close incidents into one
    :param frame_list:
    :param certainity:
    :param length_threshold:
    :param fps:
    :return:
    """
    time_id = 0
    while time_id < len(frame_list) - 1:
        if frame_list[time_id + 1][0] - frame_list[time_id][1] <= length_threshold / float(fps):
            # frame_list[time_id][1] = frame_list[time_id+1][1]
            # tuple object does not support item assignment :(
            frame_list[time_id] = (frame_list[time_id][0], frame_list[time_id + 1][1])

            del frame_list[time_id + 1]
            del certainity[time_id + 1]
            # frame_list.remove(frame_list[time_id+1])
        else:
            time_id += 1

    return


def remove_shorter_than(frame_list, certainity, length_second):
    """
    This filter removes short incidents
    :param frame_list:
    :param certainity:
    :param length_second: minimum duration of incident to be valid
    :return:
    """
    tid = 0
    while tid < len(frame_list):
        if frame_list[tid][1] - frame_list[tid][0] < float(length_second):
            del frame_list[tid]
            del certainity[tid]
        else:
            tid += 1



def generate_preview_image_from_images(input_file_dir, preview_image_name):
    tile_width_img = 4
    tile_height_img = 2  # create 4x2 tile
    preview_img_command = 'ffmpeg -loglevel panic -pattern_type glob -i "{0}*.jpg" -filter_complex scale=iw/2:-1,tile={1}x{2} {3}'.format(
        input_file_dir, tile_width_img, tile_height_img, preview_image_name)
    # check the image file name and end-time - start-time
    call(shlex.split(preview_img_command), shell=False)
    if os.path.exists(preview_image_name):
        return preview_image_name  # return image path if it exists
    else:
        return False


def create_preview_image_from_incident(key, timestamp, tmp_path, enable_person_segmenter=True, testsession_id=1234,
                                       LOG_LEVEL='0', PREVIEW_IMAGE_PATH=None):
    if key == 'WRONG_FACE_FACESCAN':
       return []
    if PREVIEW_IMAGE_PATH is None:
        preview_imgs_tmp_path = tempfile.gettempdir() + '/video_incident_previews/' + str(testsession_id) + '/'
    else:
        preview_imgs_tmp_path = PREVIEW_IMAGE_PATH
    preview_imgs_incident_folder = preview_imgs_tmp_path + 'temp_images' + '/'
    preview_images_list = []
    # if enable_person_segmenter:
    #     # fps = 9
    #     fps = 1
    # else:
    #     # fps = 3
    #     fps = 1

    fps =1 

    for inci in timestamp:
        t1 = int(inci[0])
        t2 = int(inci[1])
        video_piece_id = int(inci[0] / 1200)

        # Equally distributing frames
        n = 8
        if n < (t2 - t1):
            step = (t2 - t1) / float(n - 1)
            inci_image = set(int(round(t1 + x * step)) for x in range(n))
        elif (t2 - t1) == 0:
            inci_image = [t1]
        else:
            inci_image = range(t1, t2)

        # Remove files in temporary directory to generate preview image
        os.system('rm -rf {}*'.format(preview_imgs_incident_folder))

        not_found_frames = []
        for i in inci_image:
            video_piece_id = int(i / 1200)
            incident_time_index = int((i % 1200) * fps) + 1
            extracted_image_path = os.path.join(tmp_path, str(video_piece_id) + '_' + str(incident_time_index) + '.jpg')
            if os.path.exists(extracted_image_path):
                os.system('cp -R {} {}'.format(extracted_image_path,
                                               preview_imgs_incident_folder))
            else:
                not_found_frames.append(i)

        if not_found_frames:
            if LOG_LEVEL == '1':
                logging.info(
                    "TestSession: {0} | {1} Extracted image not found for time {2}."
                        .format(testsession_id, key, not_found_frames))

        preview_img_path = preview_imgs_tmp_path + key + '_' + str(video_piece_id) + '_' + \
                           str(t1) + '_' + str(t2) + '.jpg'
        if os.path.exists(preview_img_path):
            os.system('rm -rf {}'.format(preview_img_path))

        if generate_preview_image_from_images(preview_imgs_incident_folder, preview_img_path):
            if LOG_LEVEL == '1':
                logging.info("TestSession: {0} | Preview image created for {1} with {2}.".format(
                    testsession_id, key, inci))
            preview_images_list.append(preview_img_path)
        else:
            if LOG_LEVEL == '1':
                logging.info("TestSession: {0} | Preview image not created for {1} with {2}.".format(
                    testsession_id, key, inci))
            preview_images_list.append(None)

    return preview_images_list



