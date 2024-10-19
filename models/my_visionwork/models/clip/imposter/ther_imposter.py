# import streamlit as st
import cv2
from models.clip.phone.modeling_clip import CLIP
from modeling_dandelin import VQADandelin
# from modeling_blip import VQABlip
# from frame_extraction2 import FrameExtractor
import os
import time
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import logging
from logging_config import setup_logging
# from old_offline import convert_to_time
from modeling_yolov8 import YOLOv8PersonDetector
from modeling_blip import VQABlip
# import modeling_yolov8
from modeling_yunet import FaceDetectorYuNet
from modeling_yolotiny import YolosPersonDetector
import imposter_detection


setup_logging()
logger = logging.getLogger(__name__)

print("Ajeet Singh")
print("all_three_multiple_persons.py")

# Initialize components
clip_model = CLIP(device="cpu")
vqa_model = VQADandelin()
yolov8_persondetector = YOLOv8PersonDetector()
vqa_blip = VQABlip()
face_detector = FaceDetectorYuNet(input_size=(320, 180))
yolo_tiny_detector = YolosPersonDetector()

def prediction(video_id, frames):

    frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    # Lists to hold frames based on classification labels

    # frames = frames[:10]



    results = []
    no_person_frames = []
    single_person_frames = []
    multiple_person_frames = []

    logger.info(f"Total Frames in {video_id} video_id:  {len(frames)}")

    overall_start_time = time.time()
    # Get embeddings for all frames
    # _, image_embeddings = clip_model.get_image_embeddings(frames)
    # print(image_embeddings.nbytes)
    print("----")
    # time_taken_by_embeddings = time.time() - start_time
    # # st.write(f"Time taken to extract embeddings: {time_taken_by_embeddings:.2f} seconds")
    # logger.info(f"Time taken to extract embeddings: {time_taken_by_embeddings:.2f} seconds")

    # precomputed_embeddings = clip_model.precompute_prompt_embeddings(text_prompts)

    clip_count = 0
    classified_frames = []
    window_size = 3


    # Step 1: Pass all frames to YOLO in a batch
    yolo_start_time = time.time()
    yolo_classifications = yolov8_persondetector.classify_batch(frames, conf_threshold=0.50)
    # yolo_classifications, results = yolov8_persondetector.dummy_classify_batch(frames, conf_threshold=0.55)
    time_taken_by_yolo = time.time() - yolo_start_time
    # st.write(f"Time taken by combined clip and vqa: {time_taken_by_clip:.2f} seconds")
    logger.info(f"time_taken_by_yolo: {time_taken_by_yolo:.2f} seconds")

    frames_for_clip = []  # Frames that YOLO marked as "no_person"
    classified_frames = []  # To store final results
    yolo_count = 0
    vqa_count = 0
    vqa_culprit_for_ls = 0
    vqa_uncertain = 0
    ls_fp_stopped_by_yolo = 0
    ls_fp_stopped_by_yolo_single_person = 0
    mp_corrected_by_blip = 0
    blip_uncertain = 0

    for frame_path, (yolo_label, yolo_prob) in zip(frames, yolo_classifications):
        if yolo_label == "no_person" or yolo_label == "multiple_persons":  # If YOLO detects no person
            # Mark frame for further processing with CLIP
            yolo_count = yolo_count + 1
            frames_for_clip.append(frame_path)
        else:
            classified_frames.append({
            "frame_path": frame_path,
            "final_label": yolo_label,
            "final_prob": yolo_prob
            })

    logger.info(f"frames_for_clip: {yolo_count}")
    if frames_for_clip:
        vqa_start_time = time.time()
        print("using yolo and vqa only")
        for i, frame_path in enumerate(frames_for_clip):
            vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are there?")
            final_label = vqa_label
            final_prob = vqa_prob

            if vqa_label == "no_person":
                vqa_culprit_for_ls = vqa_culprit_for_ls + 1

            if vqa_label == "single_person":
                yolo_classifications_in_vqa = yolov8_persondetector.classify_batch([frame_path], conf_threshold=0.001)
                yolo_label, yolo_prob = yolo_classifications_in_vqa[0]
                if yolo_label == "no_person":
                    ls_fp_stopped_by_yolo = ls_fp_stopped_by_yolo + 1
                    final_label = "no_person"
                    print(f"frame_path: {frame_path}")

            if vqa_label == "multiple_persons":
                print(frame_path)
                mp_corrected_by_blip = mp_corrected_by_blip + 1


                blib_start_time = time.time()
                blib_label, _ = vqa_blip.classify(frame_path, "how many fully visible human faces are clearly present in the picture?")

                if blib_label == "Uncertain":
                    blip_uncertain = blip_uncertain + 1
                    final_label = "multiple_persons"
                    final_prob = 1
                else:
                    final_label = blib_label
                    final_prob = 1
                    total_time_blip = time.time() - blib_start_time
                    logger.info(f"time_taken_by_blib_per_image: {total_time_blip}")
                
            if vqa_label == "Uncertain":
                vqa_uncertain = vqa_uncertain + 1
                final_label = "single_person"
                final_prob = 1

            # Step 6: Store the final results for the frame
            classified_frames.append({
                "frame_path": frame_path,
                "final_label": final_label,
                "final_prob": final_prob
            })
        time_taken_by_vqa = time.time() - vqa_start_time
        logger.info(f"time_taken_by_vqa: {time_taken_by_vqa:.2f} seconds")


    # logger.info(f"clip_count: {clip_count}")
    # logger.info(f"vqa_count: {vqa_count}")
    logger.info(f"vqa_culprit_for_ls: {vqa_culprit_for_ls}")
    logger.info(f"vqa_uncertain: {vqa_uncertain}")
    logger.info(f"ls_fp_stopped_by_yolo: {ls_fp_stopped_by_yolo}")
    logger.info(f"ls_fp_stopped_by_yolo_single_person: {ls_fp_stopped_by_yolo_single_person}")
    logger.info(f"mp_corrected_by_blip: {mp_corrected_by_blip}")
    logger.info(f"blip_uncertain: {blip_uncertain}")

    frame_window = deque(maxlen=window_size)
    print(len(classified_frames))
    corrected_frames = []
    classified_frames = sorted(classified_frames, key=lambda x: int(os.path.splitext(os.path.basename(x['frame_path']))[0].split('_')[1]))
    corrected_frames.append(classified_frames[0])

    shifted_by_window = []
    for frame in classified_frames:
        frame_window.append(frame)

        # Process the middle frame when window is full (size 3)
        if len(frame_window) == window_size:
            prev_frame, curr_frame, next_frame = frame_window

            prev_label = prev_frame["final_label"]
            curr_label = curr_frame["final_label"]
            next_label = next_frame["final_label"]

            # Round probabilities to 4 decimal places
            prev_prob = round(prev_frame["final_prob"], 4)
            curr_prob = round(curr_frame["final_prob"], 4)
            next_prob = round(next_frame["final_prob"], 4)

            # If the current label is inconsistent with previous and next frames
            if curr_label != prev_label and curr_label != next_label:
                
                # Case 1: If previous and next labels are the same, use their label as the majority
                if prev_label == next_label:
                    majority_label = prev_label
                    curr_frame["final_label"] = majority_label
                    curr_frame["final_prob"] = max(prev_prob, next_prob)  # Probability rounded to 4 decimal places
                    print(f"Outlier detected in {curr_frame['frame_path']}. Corrected label: {majority_label}")
                    # st.write(f"Outlier detected in {curr_frame['frame_path']}. Corrected label: {majority_label}")
                    shifted_by_window.append((frame["frame_path"], frame["final_prob"]))

                # Case 2: If all three labels are different, choose the label with the highest rounded probability
                else:
                    # Create a list of probabilities and labels
                    probs = [prev_prob, curr_prob, next_prob]
                    labels = [prev_label, curr_label, next_label]
                    
                    # Find the label corresponding to the highest probability
                    max_prob = max(probs)
                    majority_label = labels[probs.index(max_prob)]

                    # Update the current frame's label and rounded probability
                    curr_frame["final_label"] = majority_label
                    curr_frame["final_prob"] = max_prob  # Already rounded to 4 decimal places
                    print(f"All labels differ for {curr_frame['frame_path']}. Chose label: {majority_label} with probability: {max_prob}")
                    # st.write(f"All labels differ for {curr_frame['frame_path']}. Chose label: {majority_label} with probability: {max_prob}")

            # Append the corrected (or unchanged) current frame to corrected_frames
            corrected_frames.append(curr_frame)


    _, _ , to_add = frame_window
    corrected_frames.append(to_add)

    # Final categorized frames
    no_person_frames = []
    single_person_frames = []
    multiple_person_frames = []

    for frame in corrected_frames:
        if frame["final_label"] == "no_person":
            no_person_frames.append((frame["frame_path"], frame["final_prob"]))
        elif frame["final_label"] == "single_person":
            single_person_frames.append((frame["frame_path"], frame["final_prob"]))
        elif frame["final_label"] == "multiple_persons":
            multiple_person_frames.append((frame["frame_path"], frame["final_prob"]))

    logger.info(f"Total no_person_frames: {len(no_person_frames)}")
    logger.info(f"Total single_person Frames: {len(single_person_frames)}")
    logger.info(f"Total multiple_persons Frames: {len(multiple_person_frames)}")

    no_person_frames_file_names = []
    for path_tuple in no_person_frames:
        file_path = path_tuple[0] 
        file_name = os.path.basename(file_path) 
        no_person_frames_file_names.append(file_name) 

    single_person_frames_file_names = []
    for path_tuple in single_person_frames:
        file_path = path_tuple[0] 
        file_name = os.path.basename(file_path) 
        single_person_frames_file_names.append(file_name)


    multiple_person_frames_file_names = []
    for path_tuple in multiple_person_frames:
        file_path = path_tuple[0] 
        file_name = os.path.basename(file_path) 
        multiple_person_frames_file_names.append(file_name)

    logger.info(f"no_person_frames frame ids: {no_person_frames_file_names}")
    # logger.info(f"no_person_frames frame ids: {single_person_frames_file_names}")
    logger.info(f"multiple_person_frames frame ids: {multiple_person_frames_file_names}")

    incidents = time_stamp_conversion(corrected_frames)

    time_taken_by_clip = time.time() - overall_start_time
    # st.write(f"Time taken by combined clip and vqa: {time_taken_by_clip:.2f} seconds")
    logger.info(f"Time taken by all three: {time_taken_by_clip:.2f} seconds")


    _, single_person_list, _ = convert_to_boolean_list(corrected_frames)
    single_person_list_incident_list = convert_to_time(single_person_list, fps=1)



    authorized_embedding = clip_model.get_image_embeddings([frames[0]])

    for idx , single_person_timestamp in enumerate(single_person_list_incident_list):
        start_sec, end_sec = single_person_timestamp
        dir_name = os.path.dirname(frames[0])
        video_piece_id = os.path.basename(frames[0]).split('_')[0]
        for sec in range(start_sec, end_sec + 1):
            frame_paths = []
            frame_filename = f"{video_piece_id}_{sec}.jpg"
            full_path = os.path.join(dir_name, frame_filename)
            frame_paths.append(full_path)

            _, slot_image_embeddings = clip_model.get_image_embeddings(frame_paths)

            imposter_detection.imposter_detection_across_slots(authorized_embedding, slot_image_embeddings)


    return incidents

def 

def apply_striding(image, stride):
    """Apply striding to the image and return the sampled pixels."""
    # Sample the image with the given stride
    sampled_pixels = image[::stride, ::stride]
    return sampled_pixels

def apply_padding(image, padding):
    """Apply padding to the image and return the padded image."""
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)


def convert_to_boolean_list(corrected_frames):
    
    no_person_list = [False] * (len(corrected_frames))
    single_person_list = [False] * (len(corrected_frames))
    multiple_person_list = [False] * (len(corrected_frames))


    for frame in corrected_frames:
        
        frame_name = frame["frame_path"].split('/')[-1]  # Get the last part of the path
        frame_id = int(frame_name.split('_')[1].split('.')[0])

        frame_id = frame_id - 1

        if frame["final_label"] == "no_person":
            no_person_list[frame_id] = True
        elif frame["final_label"] == "single_person":
            single_person_list[frame_id] = True
        elif frame["final_label"] == "multiple_persons":
            multiple_person_list[frame_id] = True

    return no_person_list, single_person_list, multiple_person_list

    

def time_stamp_conversion(corrected_frames):
    incidents = {}

    no_person_list, single_person_list, multiple_person_list = convert_to_boolean_list(corrected_frames)

    # violation_types = ["No_Person", "Multiple_Person"]

    confidence_measures = []
    incident_list = convert_to_time(no_person_list, fps=1)
    # incidents["No_Person"] = (incident_list, confidence_measures)
    incidents["NO_FACE"] = (incident_list, confidence_measures)

    confidence_measures = []
    incident_list = convert_to_time(multiple_person_list, fps=1)
    # incidents["Multiple_Person"] = (incident_list, confidence_measures)
    incidents["MULTIPLE_FACES"] = (incident_list, confidence_measures)

    incidents["WRONG_FACE"] = ([], [])
    incidents["BACKGROUND_MOTION"] = ([], [])
    incidents["FSLA"] = ([], [])

    return incidents

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
