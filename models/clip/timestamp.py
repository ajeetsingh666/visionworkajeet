import os
import logging


# class TimeStamp:
#     def __init__():
#         pass


incidents = {}

def convert_to_time(list, fps=1):
    time_list = []

    # add False to starting point and ending point of list
    newlist = [False] + list + [False]
    for frame_id in range(0, newlist.__len__() - 1):
        if newlist[frame_id] == False and newlist[frame_id + 1] == True:
            time_start = frame_id
        elif newlist[frame_id] == True and newlist[frame_id + 1] == False:
            time_list.append((time_start / float(fps), (frame_id - 1) / float(fps)))
    return time_list


import shlex
from subprocess import call


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

def get_result(result_dict, certainty_dict, fps, settings, incident_types, create_preview_image=False):
    """
    This function filters the initial results and discards much shorter incidents.
    :return: Filtered incidents, certainty measures, and optional preview images.
    """
    if None in result_dict:
        return None

    # result_dict ={}
    # result_dict['0'] = 
    # result_dict['NO_PERSON'] = 

     
    incidents = {}
    certainty = {}
    all_result = []
    all_certainty = []
    all_video_piece_list = []
    all_video_length = 0
    type = ['NO_PERSON', 'MULTIPLE_PERSONS']

    settings = {}
    settings['PREVIEW_IMAGE_PATH'] = "/tmp/video_incident_previews_ajeet/2529909_-2/"

    all_result.extend(result_dict)
    # all_certainty.extend(certainty_dict)
    all_certainty.extend([None])
    preview_images_dict = {}
    preview_imgs_tmp_path = settings['PREVIEW_IMAGE_PATH']
    preview_imgs_incident_folder = preview_imgs_tmp_path + 'temp_images' + '/'

    if create_preview_image:
        call(shlex.split('mkdir -p ' + preview_imgs_tmp_path), shell=False)
        call(shlex.split('mkdir -p ' + preview_imgs_incident_folder), shell=False)

    for incident in type:
        if incident in all_result[0]:
            incident_list = [True] * all_video_length
            time_stamps = []
            confidence_measures = []
            for video_id in range(len(all_video_piece_list)):
                starting = 0
                offset = all_video_piece_list[video_id][0]
                piece_length = all_video_piece_list[video_id][1]
                incident_list[offset:offset + piece_length] = [True] * piece_length
                for idx, time_period in enumerate(all_result[video_id][incident][0]):
                    ending = int(time_period[0])
                    incident_list[starting + offset:ending + offset] = [False] * (ending - starting)
                    start_inci = (time_period[0] + offset) / fps
                    end_inci = (time_period[1] + offset) / fps
                    time_stamps.append((start_inci, end_inci))
                    try:
                        confidence_measures.append(all_result[video_id][incident][1][idx])
                    except:
                        confidence_measures.append(0)
                    starting = int(time_period[1] + 1)
                incident_list[starting + offset:offset + piece_length] = [False] * (piece_length - starting)

            if incident in settings['FILTER']:
                incident_list = frame_filter(incident_list, settings['FILTER'][incident][0], settings['FILTER'][incident][1])
            else:
                incident_list = frame_filter(incident_list, settings['FILTER']['DEFAULT'][0], settings['FILTER']['DEFAULT'][1])

            incidents[incident] = convert_to_time(incident_list, fps)
            certainty[incident] = compare_incidents(incidents[incident], time_stamps, confidence_measures)

            if incident in settings['FILTER']:
                frame_filter2(incidents[incident], certainty[incident], settings['FILTER'][incident][2], fps)
                remove_shorter_than(incidents[incident], certainty[incident], settings['FILTER'][incident][3])
            else:
                frame_filter2(incidents[incident], certainty[incident], settings['FILTER']['DEFAULT'][2], fps)
                remove_shorter_than(incidents[incident], certainty[incident], settings['FILTER']['DEFAULT'][3])

            if create_preview_image:
                preview_images_dict[incident] = create_preview_image_from_incident(
                    incident, incidents[incident], settings['ALL_TEMP_PATH'],
                    settings['ENABLE_PERSON_SEGMENTER'], settings['TESTSESSION_ID'], 
                    settings['LOG_LEVEL'], preview_imgs_tmp_path)

    # if settings.get('COMPUTE_HEAD_ANGLES', False):
    #     incidents['FSLA'], certainty['FSLA'] = generate_violations_pose_index()
    #     if create_preview_image:
    #         preview_images_dict['FSLA'] = create_preview_image_from_incident(
    #             'FSLA', incidents['FSLA'], settings['ALL_TEMP_PATH'],
    #             settings['ENABLE_PERSON_SEGMENTER'], settings['TESTSESSION_ID'],
    #             settings['LOG_LEVEL'], preview_imgs_tmp_path)
    #         call(shlex.split('rm -rf ' + preview_imgs_incident_folder), shell=False)

    return incidents, certainty, preview_images_dict


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
    if enable_person_segmenter:
        # fps = 9
        fps = 1
    else:
        # fps = 3
        fps = 1

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


















# fps = 2  # frames per second
# violation_list = [True, False, True, True, True, False, True, True, True, False, True]

# incident_list = convert_to_time(violation_list, fps)

# print(incident_list)
# my_list = []
# my_list.append()