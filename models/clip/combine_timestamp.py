import os
import glob
import shlex
import tempfile
import time
import sys
from subprocess import call
import urllib as ur
import logging
from logging_config import setup_logging
import shutil
# from old_offline import convert_to_time
from old_offline import Offline, OfflineWorker


setup_logging()
logger = logging.getLogger(__name__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_video(video_path, image_path, onboarding_imgs, violations):
    start_time = time.time()

    # video_file_path_list = ["/home/ajeet/codework/daaset-download/Dataset/benchmark_dataset_13_1_20/2529909/2529909"]
    # video_file_path = "/home/ajeet/codework/daaset-download/Dataset/benchmark_dataset_13_1_20/2529909/2529909"

    # video_file_path_list = [video_path]
    video_file_path = video_path[0]

    # folder_name = os.path.splitext(os.path.basename(video_file_path_list[0]))[0]
    folder_name = os.path.splitext(os.path.basename(video_file_path))[0]
    tmp_path = tempfile.gettempdir() + '/video_incidents_ajeet/' + folder_name + '/'

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
        print(f"{tmp_path} has been deleted")
    else:
        print(f"{tmp_path} does not exist")

    call(shlex.split('mkdir -p ' + tmp_path), shell=False)
    workflow_num = -2

    fps = 1
    # violation_types = ["No_Person", "Multiple_Person"]
    violation_types = ["NO_FACE", "MULTIPLE_FACES", "WRONG_FACE", "FSLA", "BACKGROUND_MOTION"]
    video_preview_tmp_path = tempfile.gettempdir() + '/video_incident_previews_ajeet/' +\
                                        str(folder_name) + '_' + \
                                        str(-2 if workflow_num is None else workflow_num) + '/'

    if os.path.exists(video_preview_tmp_path):
        shutil.rmtree(video_preview_tmp_path)
        print(f"{video_preview_tmp_path} has been deleted")
    else:
        print(f"{video_preview_tmp_path} does not exist")

    offline_settings = {'ALL_TEMP_PATH': tmp_path, 
                        'TESTSESSION_ID': folder_name, 
                        'LOG_LEVEL': '1',
                        'PREVIEW_IMAGE_PATH': video_preview_tmp_path}


    offline_e = Offline(video_file_path, fps=fps, incident_type=violation_types,
                        settings=offline_settings)


    n_tasks = offline_e.get_tasks()
    for i in range(n_tasks):
        offline_e.run(i)


    incidents, certainity, preview_image_dict = offline_e.get_result(create_preview_image=True)

    print('[Info-video-tasks-utils] Video tasks results :\n Timestamp: {} \n Confidence: {} '.format(incidents,
                                                                                        certainity))

    end_time = time.time() - start_time
    logger.info(f"Total Time taken: {end_time}")

    return incidents, certainity, [], [], preview_image_dict

if __name__ == "__main__":
    process_video(["/home/ajeet/codework/daaset-download/Dataset/benchmark_dataset_13_1_20/2578685/2578685"], [], [], [])
