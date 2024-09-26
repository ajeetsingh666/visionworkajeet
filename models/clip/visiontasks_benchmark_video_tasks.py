"""
Usage: python visiontasks/benchmark_video_tasks.py preprod_21_1_22.csv benchmark_dataset_13_1_20
"""

# from visiontasks.video_tasks_utils_local import process_video
from combine_timestamp import process_video
import glob
import os
import csv
import time
import logging as logger
import traceback
import pickle
from subprocess import call
import shlex
import sys
import collections

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
    TP = []
    FP = []
    TN = []
    fsla_tp = []
    fsla_fp = []
    fsla_tn = []

    viol = {'FSLA': [], 'LS': [], 'BM': [], 'MP': [], 'imposter': []}

    ls_tp = []
    ls_fp = []
    ls_tn = []

    bm_tp = []
    bm_fp = []
    bm_tn = []

    mp_tp = []
    mp_fp = []
    mp_tn = []

    imp_tp = []
    imp_fp = []
    imp_tn = []

    process_time = []
    fsla_videos = []
    total_videos = []
    ignore = []
    b_file_name = os.path.basename(file_name).split('.')[0]
    dir_name = os.path.dirname(file_name)
    files = os.path.join(dir_name, b_file_name + '.csv')
    result_file = os.path.join(dir_name, 'result_' + b_file_name + '.csv')

    with open(files, 'r') as csv1:
        reader = csv.reader(csv1)
        for row in reader:
            if row[4] != '' and row[4] != 'TP':
                TP.append(row[4])
                FP.append(row[5])
                if row[1] == 'FSLATOTAL' and row[6] != 'ignore':
                    fsla_tp.append(row[4])
                    fsla_fp.append(row[5])
                    fsla_tn.append(row[6])
                if row[1] == 'NO_FACETOTAL':
                    ls_tp.append(row[4])
                    ls_fp.append(row[5])
                    ls_tn.append(row[6])
                if row[1] == 'BACKGROUND_MOTIONTOTAL':
                    bm_tp.append(row[4])
                    bm_fp.append(row[5])
                    bm_tn.append(row[6])

                if row[1] == 'MULTIPLE_FACESTOTAL':
                    mp_tp.append(row[4])
                    mp_fp.append(row[5])
                    mp_tn.append(row[6])

                if row[1] == 'WRONG_FACETOTAL':
                    imp_tp.append(row[4])
                    imp_fp.append(row[5])
                    imp_tn.append(row[6])

            if row[6] == 'ignore':
                ignore.append(row[6])

            if row[0] not in total_videos and row[0] != 'id' and row[0] != '':
                total_videos.append(row[0])

            if row[6] != '' and row[6] != 'ignore' and row[6] != 'TN':
                TN.append(row[6])

            try:
                if row[1] == 'Time_taken_by_process_video' and row[0] != '':
                    r = float(row[7])
                    process_time.append(r)

            except ValueError:
                pass

    # print TP
    fsla_tp = [int(tp) for tp in fsla_tp]
    fsla_fp = [int(fp) for fp in fsla_fp]
    fsla_tn = [int(tn) for tn in fsla_tn]

    ls_tp = [int(ls) for ls in ls_tp]
    ls_fp = [int(fp) for fp in ls_fp]
    ls_tn = [int(tn) for tn in ls_tn]

    bm_tp = [int(tp) for tp in bm_tp]
    bm_fp = [int(fp) for fp in bm_fp]
    bm_tn = [int(tn) for tn in bm_tn]

    mp_tp = [int(tp) for tp in mp_tp]
    mp_fp = [int(fp) for fp in mp_fp]
    mp_tn = [int(tn) for tn in mp_tn]

    imp_tp = [int(tp) for tp in imp_tp]
    imp_fp = [int(fp) for fp in imp_fp]
    imp_tn = [int(tn) for tn in imp_tn]

    TP = [int(tp) for tp in TP]
    FP = [int(fp) for fp in FP]
    TN = [int(tn) for tn in TN]

    # res = [sum(fsla_tp), sum(fsla_fp), sum(fsla_tn), sum(ls_tp),sum(ls_fp),sum(ls_tn),sum(bm_tp),sum(bm_fp),sum(bm_tn),sum(mp_tp),sum(mp_fp),sum(mp_tn)]
    print(sum(fsla_tp), sum(fsla_fp), sum(fsla_tn), sum(ls_tp), sum(ls_fp), sum(ls_tn), sum(bm_tp), sum(bm_fp), sum(
        bm_tn), sum(mp_tp), sum(mp_fp), sum(mp_tn))
    print(sum(TP), sum(FP), sum(TN))
    print(len(total_videos), len(ignore), len(total_videos) - len(ignore))

    viol['FSLA'].append([sum(fsla_tp), sum(fsla_fp), sum(fsla_tn)])
    viol['BM'].append([sum(bm_tp), sum(bm_fp), sum(bm_tn)])
    viol['LS'].append([sum(ls_tp), sum(ls_fp), sum(ls_tn)])
    viol['MP'].append([sum(mp_tp), sum(mp_fp), sum(mp_tn)])
    viol['imposter'].append([sum(imp_tp), sum(imp_fp), sum(imp_tn)])

    temp = []
    precision = {}
    recall = {}
    f1_score = {}
    # with open('/home/admin-pc/workspace/Benchmarking/result.csv', 'a') as f:
    with open(result_file, 'w') as csv_file:
        # writer = csv.writer(csv_file, delimiter= ' ')
        writer = csv.writer(csv_file)
        writer.writerow(['Type', 'No_videos', 'TP', 'FP', 'FN', 'Precision', 'Accuracy', 'F1 Score', 'Process_time'])

        for key, value in viol.items():
            # val = map(str,value[0])
            tp, fp, fn = value[0]
            if tp == 0 or fp == 0:
                precision[key] = 0
                f1_score[key] = 0
            else:
                precision[key] = round(tp / (tp + fp), 2)

            if tp == 0 or fn == 0:
                recall[key] = 0
            else:
                recall[key] = round(tp / (tp + fn), 2)
            if precision[key] == 0 or recall[key] == 0:
                f1_score[key] = 0
            else:
                f1_score[key] = round((2 * precision[key] * recall[key]) / (precision[key] + recall[key]), 2)

            if key == 'FSLA':
                writer.writerow([key, len(total_videos) - len(ignore)] + list(value[0]) +
                                [precision[key], recall[key], f1_score[key]])
            else:
                writer.writerow([key, len(total_videos)] + list(value[0]) + [precision[key], recall[key], f1_score[key]])
            temp.append(viol[key][0])

    add = [sum(x) for x in zip(temp[0], temp[1], temp[2], temp[3], temp[4])]
    with open(result_file, 'a') as csv_file:
        # writer = csv.writer(csv_file, delimiter= ' ')
        writer = csv.writer(csv_file)
        data = ['Total', ''] + add + [sum(list(precision.values())) / 1, sum(list(recall.values())),
                                           sum(list(f1_score.values())), sum(process_time)]
        print(data)
        writer.writerow(data)

    # plot_graph(precision, recall, f1_score, os.path.basename(result_file).split('.')[0] + '.jpg')
    print(viol)


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
    # type: (object, object, object) -> object
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
                         'LEFT_SCREEN': 'left screen'}
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
    short_overlap_file = os.path.join(dir_n, 'short_overlap_' + file_to_write)
    file_to_write = os.path.join(dir_n, file_to_write)

    if default_dataset:
        infile = open('Benchmark_default_114_sessions', 'rb')
        list_dir = pickle.load(infile)
        infile.close()
    else:
        list_dir = os.listdir(dir_path)

    print('[Info-Bench] Total videos: ', len(list_dir))

    a = os.path.isfile(file_to_write)
    if a is False:
        with open(file_to_write, 'w') as fw:
            fw = csv.writer(fw)
            col_names = ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN',
                         'processed_time']
            fw.writerow(col_names)

    b = os.path.isfile(short_overlap_file)
    if b is False:
        with open(short_overlap_file, 'w') as fw:
            fw = csv.writer(fw)
            col_names = ['id', 'violation_type', 'processed_video', 'labelled_incident', 'TP', 'FP', 'TN',
                         'processed_time']
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
            continue

        if str(test_id) in prev_processed_id:  # checks if the given session id is already processed or not
            print('[Info-Bench] {} {} Already Benchmarked...'.format(no_videos, test_id))
            continue
        else:
            print('[Info-Bench] TestSession {0} | Benchmarking for video no {1}'.format(test_id, no_videos))
        try:
            video_path = dir_path + "/{}/{}".format(test_id, test_id)
            if not os.path.exists(video_path):
                video_path = dir_path + "/{}/{}.mp4".format(test_id, test_id)
                if not os.path.exists(video_path):
                    print('[Info-Bench] TestSession {0} | Video not exist.'.format(test_id))
                    continue
            imposter_bench = True
            if os.path.exists(dir_path + "/{}/facescan".format(test_id)):
                image_path = glob.glob(dir_path + "/{}/facescan/*.jpg".format(test_id))
            else:
                image_path = glob.glob(dir_path + "/{}/*.jpg".format(test_id))
            onboarding_imgs = glob.glob(dir_path + "/{}/onboarding/*.jpg".format(test_id))
            if not image_path:
                image_path = glob.glob(current_path + '../aligner/*.jpg')
                imposter_bench = False
            print('[Info-Bench] Video Path: {} Image Path: {}'.format(video_path, image_path))
            t0 = time.time()
            practical_ts = process_video([video_path], image_path, onboarding_imgs, violations)
            t1 = time.time()
            total = t1 - t0
            print('[Info-Bench] Visiontask Result: ', practical_ts)
            print('[Info-Bench] Time taken by video task is {} secs'.format(total))

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
                    cal = (0, 0, 0)
                    cal_new = (0, 0, 0)
                else:
                    print("Type :", key, " Value : ", value, " Labels : ", lts)
                    cal = compute_overlap(value, lts)
                    cal_new = incidents_counter(value, lts)
                print('[Info-Bench] Overlap: ', cal)
                print('[Info-Bench] Incident Overlap: ', cal_new)
                final_list = [test_id, key + 'TOTAL', value, lts, cal[0], cal[1], cal[2]]
                final_incident_counter = [test_id, key + 'TOTAL', value, lts, cal_new[0], cal_new[1], cal_new[2]]
                write_to_csv(file_to_write, final_list)
                write_to_csv(short_overlap_file, final_incident_counter)

            final_list = [test_id, 'Time_taken_by_process_video', '', '', '', '', '', total]
            write_to_csv(file_to_write, final_list)
        except Exception as e:
            logger.exception("error at #{0} {1}".format(test_id, traceback.format_exc()))
            pass

    print('[Info-Bench] Results are stored in {} .'.format(file_to_write))
    generate_result(file_to_write)
    generate_result(short_overlap_file)


if __name__ == '__main__':
    # benchmark_file_name = 'Benchmark_' + str(sys.argv[1])
    benchmark_file_name = "abc"
    # dataset_path = sys.argv[2]
    dataset_path = "/home/ajeet/codework/debug_dataset/"
    # label_file_path = os.path.join(os.getcwd(), 'visiontasks/annotations')
    label_file_path = "/home/ajeet/codework/visionworkajeet/annotations"
    # print(os.path.exists(label_file_path), label_file_path)
    for i in glob.glob(dataset_path+'/*'):
        generated_time_stamps(label_file_path, benchmark_file_name, i, default_dataset=False)
