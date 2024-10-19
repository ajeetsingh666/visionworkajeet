"""
Usage: python visiontasks/benchmark_video_tasks.py [v|a] preprod_21_1_22.csv benchmark_dataset_13_1_20
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from visiontasks.video_tasks_utils_local import process_video
from combine_timestamp import process_video
# from visiontasks.micros import NON_BM_MP
import datetime
import glob
import csv
import time
import logging as logger
import traceback
import pickle
from subprocess import call
import shlex
import sys
import collections
import pandas as pd
from datetime import timedelta
import logging

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

current_path = os.getcwd()


def compute_overlap(practical_incidents, labeled_incidents):
    """
    This function computes how well the practical and labeled incidents overlap.
    :param practical_incidents: Results of visiontasks algorithms
    :param labeled_incidents: Labeled incidents.
    :return: performance metrics. (TP, FP, FN)
    """
    # Since, each incident is a tuple like (start_time, end_time). In order to compute the overlap of incidents
    # we create a list which contains all the time stamps which lie within the incident.
    p_time_stamps = []
    l_time_stamps = []
    for p_i in practical_incidents:
        p_time_stamps.extend(range(int(p_i[0]), int(p_i[1]) + 1))
    for l_i in labeled_incidents:
        l_time_stamps.extend(range(int(l_i[0]), int(l_i[1]) + 1))

    # Now compute the overlap and remainders in the two lists
    p_multiset = collections.Counter(p_time_stamps)
    l_multiset = collections.Counter(l_time_stamps)

    overlap = list((p_multiset & l_multiset).elements())
    p_remainder = list((p_multiset - l_multiset).elements())
    l_remainder = list((l_multiset - p_multiset).elements())

    '''
    # Now compute various metrics like DSC, Sensitivity, Selectivity or (Precision, Recall)
    sensitivity = float(len(overlap)) / (len(overlap) + len(l_remainder))  # TPR, sensitivity, recall
    selectivity = float(len(overlap)) / (len(overlap) + len(p_remainder))  # selectivity, precision
    dsc = 2 * selectivity * sensitivity / (selectivity + sensitivity)  # 2*Precision*Recall/(Precision+Recall)
    '''

    return len(overlap), len(p_remainder), len(l_remainder)  # return TP, FP, FN


def check_video_id_in_ground_truth(csv_file_path,
                                   video_id):  # checks for the csv file in the directory that contains the given video
    ls_files = {}
    unprocessed_ids = []
    list_gt = glob.glob(csv_file_path + '/*.csv')
    for files in list_gt:
        with open(files, 'r') as csv1:
            reader = csv.reader(csv1)
            for row in reader:
                if video_id == row[0] and (row[5] == 'y' or row[5] == 'Y'):
                    ls_files[video_id] = files
                    break
    return ls_files


def write_to_csv(filename, data):  # writes the given data to the csv filename provided
    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
    csvfile.close()


def incidents_counter(practical_inc, labelled_inc):
    tp, tn, fp = 0, 0, 0
    li = []
    for l_i in labelled_inc:
        li.extend(range(int(l_i[0]), int(l_i[1] + 1)))
    for p_i in practical_inc:
        temp = []
        temp.extend(range(int(p_i[0]), int(p_i[1] + 1)))
        similar_count = set(temp).intersection(set(li))
        # false_positive = set(temp).difference((set(li)))
        # print(temp,similar_count)
        if not similar_count:
            fp += 1

    pi = []
    for p_i in practical_inc:
        pi.extend((range(int(p_i[0]), int(p_i[1] + 1))))
    for l_i in labelled_inc:
        li = []
        li.extend(range(int(l_i[0]), int(l_i[1] + 1)))
        similar_count = set(li).intersection(set(pi))
        if not similar_count:
            tn += 1
        else:
            tp += 1

    return tp, fp, tn


def generate_preview_image(video_file_path, start_time, end_time, preview_image_name):
    if start_time < 0:
        start_time = 0
    tile_width_img = 4
    tile_height_img = 2  # create 4x2 tile
    num_frames_to_tile = tile_width_img * tile_height_img
    if end_time - start_time < num_frames_to_tile:  # if too short duration
        end_time = start_time + num_frames_to_tile
    frame_rate = float(num_frames_to_tile) / (end_time - start_time)
    preview_img_command = "ffmpeg -loglevel panic -i {0} -frames 1 -q:v 1 -vf {1}select='between(t\,{2}\,{3})'," \
                          "fps={4},scale=iw/2:-1,tile={5}x{6}:margin=2:padding=2{7} -y {8}".format(video_file_path,
                                                                                                   "\"",
                                                                                                   start_time,
                                                                                                   end_time,
                                                                                                   frame_rate,
                                                                                                   tile_width_img,
                                                                                                   tile_height_img,
                                                                                                   "\"",
                                                                                                   preview_image_name)
    # check the image file name and end-time - start-time
    call(shlex.split(preview_img_command), shell=False)
    if os.path.exists(preview_image_name):
        return preview_image_name  # return image path if it exists
    else:
        return False


def generate_result(file_name):
    df = pd.read_csv(file_name)
    keys = pd.Series(df['violation_type']).unique()
    keys = [x for x in keys if str(x) != 'nan']
    # print('Keys: ', keys)
    total_sessions = pd.Series(df['id']).unique()
    total_sessions = [x for x in total_sessions if str(x) != 'nan']
    benchmark = {'total_sessions': len(total_sessions)}
    for k in keys:
        if k == 'Time_taken_by_process_video':
            benchmark[k] = round(df[df['violation_type'] == k]['processed_time'].sum(), 2)
            continue
        tp = df[df['violation_type'] == k]['TP'].sum()
        fp = df[df['violation_type'] == k]['FP'].sum()
        fn = df[df['violation_type'] == k]['TN'].sum()

        precision = round(tp / (tp + fp), 2)
        recall = round(tp / (tp + fn), 2)
        f1_score = round(2 * precision * recall / (precision + recall), 2)
        benchmark[k] = {'TP': tp, 'FP': fp, 'FN': fn, 'precision': precision, 'recall': recall, 'f1_score': f1_score}
    # print(benchmark)
    df_s = pd.DataFrame(benchmark)
    df_s.fillna(0)
    df_s.to_csv('{}/Result_{}'.format(os.path.dirname(file_name), os.path.basename(file_name)))
    df_l = pd.DataFrame(benchmark)
    df_l.drop(['total_sessions'], axis=1, inplace=True)
    df_l.drop(['TP', 'FP', 'FN'], axis=0, inplace=True)
    df_l.fillna(0, inplace=True)
    if 'Time_taken_by_process_video' in keys:
        df_l.drop(['Time_taken_by_process_video'], axis=1, inplace=True)
    ax = df_l.plot()
    fig = ax.get_figure()
    fig_name = '{}/Graph_{}'.format(os.path.dirname(file_name), os.path.basename(file_name).replace('csv', 'png'))
    fig.savefig(fig_name)
    return benchmark


def plot_graph(precision, recall, f1_score, image_name):
    print(precision, recall, f1_score)
    import matplotlib.pyplot as plt

    # x-coordinates of left sides of bars
    p = precision.values()
    r = recall.values()
    f1 = f1_score.values()

    # labels for bars
    tick_label = list(precision.keys()) * 3

    # plotting a bar chart
    plt.bar(range(0, 100, 10), list(p) + list(r) + list(f1), tick_label=tick_label,
            width=0.8, color=['blue', 'green', 'red'])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('Summary')
    print("[INFO] ", image_name)
    # function to show the plot
    plt.savefig(image_name)


def labelled_time_stamps(filename, video_id, violation):
    '''
	Pass the following arguments to the function:-
	1)filename of the csv file with ground truths
	2) video_id to be checked
	3) the violation to be processed
	'''
    with open(filename, 'r') as csvfile1:
        reader = csv.reader(csvfile1)
        Sessions = {}
        violation_map = {'WRONG_FACE_FACESCAN': 'Imposter_in_facescan', 'NO_FACE': 'LS', 'MULTIPLE_FACES': 'MP',
                         'WRONG_FACE': 'Imposter', 'BACKGROUND_MOTION': 'BM', 'FSLA': 'FSLA',
                         'LEFT_SCREEN': 'left screen', 'BA': 'BA'}
        # import ipdb;ipdb.set_trace()
        for row in reader:

            if video_id in row[0]:
                if row[0] not in Sessions.keys():
                    Sessions[row[0]] = {}

                if row[1] == '' or (row[5] != 'y' and row[5] != 'Y') or row[5] == '':
                    continue

                if row[1] not in Sessions[row[0]].keys():
                    Sessions[row[0]][row[1]] = []  # Violation

                Sessions[row[0]][row[1]].append(([str(row[3]), str(row[4])]))

        if violation_map[violation] in Sessions[video_id].keys():
            # print (violation_map[violation])

            return Sessions[video_id][violation_map[violation]]
        # print (Sessions[video_id][violation_map[violation]])

        else:
            return []


def minutes_second(l):
    # print len(l)
    l1 = l.split(".")
    # print l1
    if len(l1) == 1:
        l = l + '.0'
        l1 = l.split(".")
    # print l

    if l1[0] != '0':
        l11 = [int(i) for i in l1]
        # print l11
        if len(l1[1]) == 1:
            l12 = (l11[0] * 60) + (l11[1] * 10)
        else:
            l12 = (l11[0] * 60) + l11[1]
    else:
        # print len(l1[1])
        l11 = [int(i) for i in l1]
        if len(l1[1]) == 1:
            l12 = (l11[1] * 10)
        else:
            l12 = l11[1]
    # print l12

    return l12


def generated_time_stamps(labelled_file, file_to_write, dir_path, violations=[], default_dataset=True):
    '''
	Pass the following arguments to the function:-
	1) csv_file_path :- the directory with the ground truth csv files
	2) file_to_write :- the csv filename to write generated overlaps
	3) dir_path :- path of the directory with video's in it.
	'''
    prev_processed_id = set()
    dir_n = file_to_write.split('.')[0]
    os.system('mkdir -p   ' + dir_n)
    print('[Info-Bench] Creating : ', os.path.join(dir_n, file_to_write))
    logger.info(f'[Info-Bench] Creating : {os.path.join(dir_n, file_to_write)}')
    short_overlap_file = os.path.join(dir_n, 'short_overlap_' + file_to_write)
    file_to_write = os.path.join(dir_n, file_to_write)

    if default_dataset:
        infile = open('Benchmark_default_114_sessions', 'rb')
        list_dir = pickle.load(infile)
        infile.close()
    else:
        list_dir = os.listdir(dir_path)
    # list_dir = list(set(list_dir).difference(NON_BM_MP))
    print('[Info-Bench] Total videos: ', len(list_dir))
    logger.info(f'[Info-Bench] Total videos: {len(list_dir)}')

    a = os.path.isfile(file_to_write)
    if a is False:
        with open(file_to_write, 'w') as fw:
            fw = csv.writer(fw)
            col_names = ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN', 'TP_short',
                         'FP_short', 'TN_short', 'processed_time']
            fw.writerow(col_names)

    b = os.path.isfile(short_overlap_file)
    if b is False:
        with open(short_overlap_file, 'w') as fw:
            fw = csv.writer(fw)
            col_names = ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN', 'TP_short',
                         'FP_short', 'TN_short', 'processed_time']
            fw.writerow(col_names)

    with open(file_to_write, 'r') as csvfile3:
        reader = csv.reader(csvfile3)
        for row in reader:
            if row[0] != 'id':
                prev_processed_id.add(row[0])

    for no_videos, test_id in enumerate(list_dir):
        test_id = test_id.split('.')[0]
        file_path = check_video_id_in_ground_truth(labelled_file, test_id)
        if not file_path:
            print('[Info-Bench] {} {} annotation not found...'.format(no_videos, test_id))
            logger.info(f'[Info-Bench] {no_videos} {test_id} annotation not found...')
            continue

        if str(test_id) in prev_processed_id:  # checks if the given session id is already processed or not
            print('[Info-Bench] {} {} Already Benchmarked...'.format(no_videos, test_id))
            logger.info(f'[Info-Bench] {no_videos} {test_id} Already Benchmarked...')
            continue
        else:
            print('[Info-Bench] TestSession {0} | Benchmarking for video no {1}'.format(test_id, no_videos))
            logger.info(f'[Info-Bench] TestSession {test_id} | Benchmarking for video no {no_videos}')
        try:
            video_path = dir_path + "/{}/{}".format(test_id, test_id)
            if not os.path.exists(video_path):
                video_path = glob.glob(dir_path + "/{}/{}.*".format(test_id, test_id))
                if video_path:
                    video_path = video_path[0]
                    if not os.path.exists(video_path):
                        print('[Info-Bench] TestSession {0} | Video not exist.'.format(test_id))
                        logger.info(f'[Info-Bench] TestSession {test_id} | Video not exist.')
                        continue
                else:
                    continue
            imposter_bench = True
            if os.path.exists(dir_path + "/{}/facescan".format(test_id)):
                image_path = glob.glob(dir_path + "/{}/facescan/*.jpg".format(test_id))
            else:
                image_path = glob.glob(dir_path + "/{}/*.jpg".format(test_id))
            onboarding_imgs = glob.glob(dir_path + "/{}/onboarding/*.jpg".format(test_id))
            if not image_path:
                image_path = glob.glob('visiontasks/aligner/*.jpg')
                imposter_bench = False
            print('[Info-Bench] Video Path: {} Image Path: {}'.format(video_path, image_path))
            logger.info(f'[Info-Bench] Video Path: {video_path} Image Path: {image_path}')
            t0 = time.time()
            practical_ts = process_video([video_path], image_path, onboarding_imgs, violations)
            t1 = time.time()
            total = t1 - t0
            print('[Info-Bench] Visiontask Result: ', practical_ts)
            logger.info(f'[Info-Bench] Visiontask Result: {practical_ts}')
            print('[Info-Bench] Time taken by video task is {} secs'.format(total))
            logger.info(f'[Info-Bench] Time taken by video task is {total} secs')

            for key, value in practical_ts[0].items():  # calculates overlaps for rest of the violations
                lts_list = labelled_time_stamps(file_path[test_id], test_id, key)
                lts = []
                if key == 'WRONG_FACE' and not imposter_bench: continue
                for i in range(len(lts_list)):
                    if lts_list[i]:
                        lts_list1 = lts_list[i]
                        l1 = minutes_second(lts_list1[0])
                        l2 = minutes_second(lts_list1[1])
                        lts_ = (l1, l2)
                        lts.append(lts_)
                    else:
                        lts.append(lts_list[i])

                if not value and not lts:
                    print("Type :", key, " Value : ", value, " Labels : ", lts)
                    logger.info(f'Type: {key} Value: {value} Labels: {lts}')
                    cal = (0, 0, 0)
                    cal_new = (0, 0, 0)
                else:
                    print("Type :", key, " Value : ", value, " Labels : ", lts)
                    logger.info(f'Type: {key} Value: {value} Labels: {lts}')
                    cal = compute_overlap(value, lts)
                    cal_new = incidents_counter(value, lts)
                print('[Info-Bench] Overlap for : ', key, cal)
                logger.info(f'[Info-Bench] Overlap for {key}: {cal}')
                print('[Info-Bench] Incident Overlap: ', key, cal_new)
                logger.info(f'[Info-Bench] Incident Overlap for {key}: {cal_new}')
                final_list = [test_id, key + 'TOTAL', value, lts, cal[0], cal[1], cal[2], cal_new[0], cal_new[1],
                              cal_new[2]]
                final_incident_counter = [test_id, key + 'TOTAL', value, lts, cal_new[0], cal_new[1], cal_new[2]]
                write_to_csv(file_to_write, final_list)
                write_to_csv(short_overlap_file, final_incident_counter)

            final_list = [test_id, 'Time_taken_by_process_video', '', '', '', '', '', '', '', '', total]
            write_to_csv(file_to_write, final_list)
        except Exception as e:
            logger.exception("error at #{0} {1}".format(test_id, traceback.format_exc()))
            pass

    print('[Info-Bench] Results are stored in {} .'.format(file_to_write))
    logger.info(f'[Info-Bench] Results are stored in {file_to_write}.')
    b1 = generate_result(file_to_write)
    b2 = generate_result(short_overlap_file)
    violation_types = {'NO_FACETOTAL': 'LS', 'MULTIPLE_FACESTOTAL': 'MP', 'WRONG_FACETOTAL': 'Imposter',
                       'BACKGROUND_MOTIONTOTAL': 'BM', 'FSLATOTAL': 'FSLA'}
    b3 = {}
    for k, v in violation_types.items():
        b3[v] = round((b1[k]['f1_score'] + b2[k]['f1_score']) / 2, 2)
    bm_mp_f1_score = compute_BM_MP_accuracy(file_to_write)
    b3.update(bm_mp_f1_score)
    df = pd.DataFrame({'Violation_type': b3.keys(), 'F1_Score': b3.values()})
    ax = df.plot()
    fig = ax.get_figure()
    fig_name = '{}/Graph_F1-Score_{}'.format(os.path.dirname(file_to_write), os.path.basename(file_to_write).replace(
        'csv', 'png'))
    fig.savefig(fig_name)
    df.to_csv('{}/Total_F1-Score_{}'.format(os.path.dirname(file_to_write), os.path.basename(file_to_write)))
    print(df)
    logger.info(df)


def convert_to_hh_mm_ss(incident_list):
    new_incident_list = []
    for inci in incident_list:
        new_incident_list.append((timedelta(seconds=inci[0]), timedelta(seconds=inci[1])))
    return new_incident_list


def convert_mm_to_secs(mm_ss):
    x = datetime.datetime.strptime(mm_ss, '%M.%S')
    time = int(x.minute * 60 + x.second + x.microsecond / 1000000)
    return time


def convert_labels_in_seconds(labelled_incidents):
    converted_time = []
    for value in labelled_incidents:
        start_at, end_at = value
        converted_time.append((minutes_second(start_at), minutes_second(end_at)))
    return converted_time


def benchmark_audio_task(file_to_write, dir_path, labelled_file):
    from visiontasks.audio.audio_voxseg import audio_analyze
    if not os.path.exists(file_to_write):
        colomns = ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'FN', 'TP_short',
                         'FP_short', 'FN_short', 'processed_time']
        write_to_csv(file_to_write, colomns)
        processed_ids = []
    else:
        df = pd.read_csv(file_to_write)
        processed_ids = pd.Series(df['id']).unique()
        print('[Info] Processed Sessions are: ', processed_ids)

    if os.path.exists(labelled_file):
        df = pd.read_csv(labelled_file)
        labelled_ids = pd.Series(df['id']).unique()
    else:
        print('[Info] Labelled file not found..!')
        return

    list_dir = os.listdir(dir_path)
    list_dir = set(list_dir).difference(processed_ids)
    key = 'BA'
    print('[Info] Sessions : ', len(list_dir))
    for test_id in list_dir:
        if test_id not in labelled_ids:
            print('[Info] {} not labelled with audio..!'.format(test_id))
            continue
        video_path = dir_path + "/{}".format(test_id)
        st = time.time()
        practical_ts = audio_analyze(video_path, testsession_id=test_id, LOG_LEVEL='1')[0]
        time_taken = round(time.time() - st, 2)
        print('[Info-video-tasks-utils] Time taken : {} | Audio task result: '.format(time_taken), practical_ts)
        try:
            lts_list = labelled_time_stamps(labelled_file, test_id, key)
            lts = convert_labels_in_seconds(lts_list)
        except Exception as e:
            print('[Info] Error while reading labels : {}'.format(e))
            break
        value = [i[:2] for i in practical_ts]

        if not value and not lts:
            print("Type :", key, " Value : ", value, " Labels : ", lts)
            cal = (0, 0, 0)
            cal_new = (0, 0, 0)
        else:
            print("Type :", key, " Value : ", value, " Labels : ", lts)
            cal = compute_overlap(value, lts)
            cal_new = incidents_counter(value, lts)
        data = [test_id, key, practical_ts, lts] + list(cal) + list(cal_new) + [time_taken]
        print("[Info] Data: ", data)
        write_to_csv(file_to_write, data)

    result = {'TP': 0, 'FP': 0, 'FN': 0, 'TP_short': 0, 'FP_short': 0, 'FN_short': 0}
    df = pd.read_csv(file_to_write)
    for column in result:
        result[column] = pd.Series(df[column]).sum()
    precision = round(((result['TP']/(result['TP'] + result['FP'])) + (
            result['TP_short']/(result['TP_short'] + result['FP_short'])))/2, 2)

    recall = round(((result['TP'] / (result['TP'] + result['FN'])) + (
            result['TP_short'] / (result['TP_short'] + result['FN_short']))) / 2, 2)

    f1_score = round((2 * precision * recall) / (precision + recall), 2)
    print('[Info] Precision {} | Recall {} | F1 Score {}'.format(precision, recall, f1_score))


def compute_BM_MP_accuracy(file_to_write):
    # BM_MP calculation
    result = {'TP': 0, 'FP': 0, 'FN': 0}
    result2 = {'TP': 0, 'FP': 0, 'FN': 0}
    os.system('rm -rf {}/BM_MP.csv'.format(os.path.dirname(file_to_write)))
    os.system('rm -rf {}/Short_BM_MP.csv'.format(os.path.dirname(file_to_write)))
    write_to_csv('{}/BM_MP.csv'.format(os.path.dirname(file_to_write)),
                 ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN', 'TP_short',
                  'FP_short', 'TN_short'])
    write_to_csv('{}/Short_BM_MP.csv'.format(os.path.dirname(file_to_write)),
                 ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN'])
    df = pd.read_csv(file_to_write)

    for i in df['id'].unique():
        df1 = df[df['id'] == i]
        df2 = df1[df1['violation_type'] == 'MULTIPLE_FACESTOTAL']['processed_video']
        df3 = df1[df1['violation_type'] == 'BACKGROUND_MOTIONTOTAL']['processed_video']
        pt = eval(df2.tolist()[0]) + eval(df3.tolist()[0])
        df2 = df1[df1['violation_type'] == 'MULTIPLE_FACESTOTAL']['labelled_incident']
        df3 = df1[df1['violation_type'] == 'BACKGROUND_MOTIONTOTAL']['labelled_incident']
        li = eval(df2.tolist()[0]) + eval(df3.tolist()[0])
        r1 = compute_overlap(pt, li)
        r2 = incidents_counter(pt, li)
        result['TP'] += r1[0]
        result['FP'] += r1[1]
        result['FN'] += r1[2]
        result2['TP'] += r2[0]
        result2['FP'] += r2[1]
        result2['FN'] += r2[2]
        data = [i, 'BM_MP', pt, li, r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]]
        write_to_csv('{}/BM_MP.csv'.format(os.path.dirname(file_to_write)), data)
        data2 = [i, 'BM_MP', pt, li, r2[0], r2[1], r2[2]]
        write_to_csv('{}/Short_BM_MP.csv'.format(os.path.dirname(file_to_write)), data2)

    b1 = generate_result('{}/BM_MP.csv'.format(os.path.dirname(file_to_write)))
    b2 = generate_result('{}/Short_BM_MP.csv'.format(os.path.dirname(file_to_write)))
    violation_types = {'BM_MP': 'BM_MP'}
    b3 = {}
    for k, v in violation_types.items():
        b3[v] = round((b1[k]['f1_score'] + b2[k]['f1_score']) / 2, 2)

    df = pd.DataFrame({'Violation_type': b3.keys(), 'F1_Score': b3.values()})
    df.to_csv('{}/BM_MP_Total_F1-Score.csv'.format(os.path.dirname(file_to_write)))
    print(df)
    return b3


if __name__ == '__main__':
    # benchmark_file_name = 'Benchmark_' + str(sys.argv[2])
    benchmark_file_name = "abc"
    # dataset_path = sys.argv[3]
    # dataset_path = "/home/ajeet/codework/ujjawal_github/Dataset/benchmark_dataset_13_1_20"
    dataset_path = "/home/ajeet/codework/phone_dataset/test/"
    # dataset_path = "/home/ajeet/codework/daaset-download/Dataset/benchmark_dataset_13_1_20"
    # if sys.argv[1] in ['v', 'V', 'Video']:
    if 'v' in ['v', 'V', 'Video']:
        # label_file_path = os.path.join(os.getcwd(), 'visiontasks/annotations')
        label_file_path = "/home/ajeet/codework/ujjawal_github/visionwork/annotations"
        # print(os.path.exists(label_file_path), label_file_path)
        # for i in glob.glob(dataset_path+'/*'):
        #     generated_time_stamps(label_file_path, benchmark_file_name, i, default_dataset=False)

        generated_time_stamps(label_file_path, benchmark_file_name, dataset_path, default_dataset=False)
    else:
        label_file_path = 'visiontasks/annotations/BA/BA_Labelling_27_04_2023.csv'
        benchmark_audio_task(benchmark_file_name, dataset_path, labelled_file=label_file_path)

