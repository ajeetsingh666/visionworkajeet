# import streamlit as st
from modeling_clip import CLIP
from modeling_dandelin import VQADandelin
import os
import time
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import logging
from logging_config import setup_logging
from modeling_yolov8 import YOLOv8PersonDetector
# from old_offline import convert_to_time


setup_logging()
logger = logging.getLogger(__name__)

print("Ajeet Singh")
print("Only yolo")

yolov8_persondetector = YOLOv8PersonDetector()

def prediction(video_id, frames):
    # frames = frames[:10]
    results = []
    no_person_frames = []
    single_person_frames = []
    multiple_person_frames = []

    logger.info(f"Total Frames in {video_id} video_id:  {len(frames)}")

    classified_frames = []
    window_size = 3

    start_time = time.time()
    frames = sorted(frames, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    # classifications = yolov8_persondetector.classify_batch(frames, conf_threshold=0.50)
    # for frame, (classification, confidence) in zip(frames, classifications):
    #     # print(f"Batch image {frame} - Classification: {classification}, Confidence: {confidence}")

    #     classified_frames.append({
    #         "frame_path": frame,
    #         "final_label": classification,
    #         "final_prob": 1
    #     })

    batch_size = 500
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        classifications = yolov8_persondetector.classify_batch(batch, conf_threshold=0.50)

        for frame, (classification, confidence) in zip(batch, classifications):
            classified_frames.append({
                "frame_path": frame,
                "final_label": classification,
                "final_prob": 1
            })

    frame_window = deque(maxlen=window_size)
    print(len(classified_frames))
    corrected_frames = classified_frames
    # corrected_frames = []
    # classified_frames = sorted(classified_frames, key=lambda x: int(os.path.splitext(os.path.basename(x['frame_path']))[0].split('_')[1]))
    # corrected_frames.append(classified_frames[0])

    # shifted_by_window = []
    # for frame in classified_frames:
    #     frame_window.append(frame)

    #     # Process the middle frame when window is full (size 3)
    #     if len(frame_window) == window_size:
    #         prev_frame, curr_frame, next_frame = frame_window

    #         prev_label = prev_frame["final_label"]
    #         curr_label = curr_frame["final_label"]
    #         next_label = next_frame["final_label"]

    #         # Round probabilities to 4 decimal places
    #         prev_prob = round(prev_frame["final_prob"], 4)
    #         curr_prob = round(curr_frame["final_prob"], 4)
    #         next_prob = round(next_frame["final_prob"], 4)

    #         # If the current label is inconsistent with previous and next frames
    #         if curr_label != prev_label and curr_label != next_label:
                
    #             # Case 1: If previous and next labels are the same, use their label as the majority
    #             if prev_label == next_label:
    #                 majority_label = prev_label
    #                 curr_frame["final_label"] = majority_label
    #                 curr_frame["final_prob"] = max(prev_prob, next_prob)  # Probability rounded to 4 decimal places
    #                 print(f"Outlier detected in {curr_frame['frame_path']}. Corrected label: {majority_label}")
    #                 # st.write(f"Outlier detected in {curr_frame['frame_path']}. Corrected label: {majority_label}")
    #                 shifted_by_window.append((frame["frame_path"], frame["final_prob"]))

    #             # Case 2: If all three labels are different, choose the label with the highest rounded probability
    #             else:
    #                 # Create a list of probabilities and labels
    #                 probs = [prev_prob, curr_prob, next_prob]
    #                 labels = [prev_label, curr_label, next_label]
                    
    #                 # Find the label corresponding to the highest probability
    #                 max_prob = max(probs)
    #                 majority_label = labels[probs.index(max_prob)]

    #                 # Update the current frame's label and rounded probability
    #                 curr_frame["final_label"] = majority_label
    #                 curr_frame["final_prob"] = max_prob  # Already rounded to 4 decimal places
    #                 print(f"All labels differ for {curr_frame['frame_path']}. Chose label: {majority_label} with probability: {max_prob}")
    #                 # st.write(f"All labels differ for {curr_frame['frame_path']}. Chose label: {majority_label} with probability: {max_prob}")

    #         # Append the corrected (or unchanged) current frame to corrected_frames
    #         corrected_frames.append(curr_frame)


    # _, _ , to_add = frame_window
    # corrected_frames.append(to_add)

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

    time_taken_by_yolo = time.time() - start_time
    # st.write(f"Time taken by combined clip and vqa: {time_taken_by_clip:.2f} seconds")
    logger.info(f"Time taken by only yolo: {time_taken_by_yolo:.2f} seconds")

    return incidents




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


    # WRONG_FACE_FACESCAN = [False] * (len(corrected_frames))
    # BACKGROUND_MOTION = [False] * (len(corrected_frames))
    # FSLA = [False] * (len(corrected_frames))

    # confidence_measures = []
    # incident_list = convert_to_time(WRONG_FACE_FACESCAN, fps=1)
    # # incidents["Multiple_Person"] = (incident_list, confidence_measures)
    # incidents["WRONG_FACE_FACESCAN"] = (incident_list, confidence_measures)

    # confidence_measures = []
    # incident_list = convert_to_time(BACKGROUND_MOTION, fps=1)
    # # incidents["Multiple_Person"] = (incident_list, confidence_measures)
    # incidents["BACKGROUND_MOTION"] = (incident_list, confidence_measures)

    # confidence_measures = []
    # incident_list = convert_to_time(FSLA, fps=1)
    # # incidents["Multiple_Person"] = (incident_list, confidence_measures)
    # incidents["FSLA"] = (incident_list, confidence_measures)

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
























    # def create_big_image(frames, grid_size=(10, 10), thumbnail_size=(100, 100), padding=10):
    #     """Create a big image by arranging smaller images in a grid with padding."""
    #     big_image_width = grid_size[0] * thumbnail_size[0] + (grid_size[0] - 1) * padding
    #     big_image_height = grid_size[1] * thumbnail_size[1] + (grid_size[1] - 1) * padding
    #     big_image = Image.new("RGB", (big_image_width, big_image_height))

    #     for i, (frame_path, prob) in enumerate(frames):
    #         if i >= grid_size[0] * grid_size[1]:  # Limit to the grid size
    #             break
    #         small_image = Image.open(frame_path).resize(thumbnail_size)

    #         frame_id = os.path.splitext(os.path.basename(frame_path))[0]
    #         draw = ImageDraw.Draw(small_image)
    #         # font = ImageFont.truetype("arial.ttf", size=10)
    #         # draw.text((5, 5), f"{prob:.4f}", fill="red")
    #         draw.text((5, 5), f"{frame_id}", fill="red")  # Adjust position and color as needed


    #         # Calculate the position with padding
    #         x = (i % grid_size[0]) * (thumbnail_size[0] + padding)
    #         y = (i // grid_size[0]) * (thumbnail_size[1] + padding)
    #         big_image.paste(small_image, (x, y))

    #     return big_image

    # def display_frames(label, frames):
    #     st.write(f"### {label.capitalize()} Frames (Count: {len(frames)})")

    #     # Calculate how many big images we can create
    #     num_big_images = (len(frames) + 100 - 1) // 100  # 100 small images per big image

    #     for page in range(num_big_images):
    #         start_idx = page * 100
    #         end_idx = min(start_idx + 100, len(frames))

    #         # Create and display the big image for the current batch
    #         big_image = create_big_image(frames[start_idx:end_idx], grid_size=(10, 10), thumbnail_size=(100, 100))
    #         st.image(big_image, caption=f"Big Image (Batch {page + 1}/{num_big_images})", use_column_width=True)


    # # Display the frames grouped by their labels
    # if no_person_frames:
    #     no_person_frames = sorted(no_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    #     display_frames("no person", no_person_frames)
    # if single_person_frames:
    #     single_person_frames = sorted(single_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    #     display_frames("single person", single_person_frames)
    # if multiple_person_frames:
    #     multiple_person_frames = sorted(multiple_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    #     display_frames("multiple persons", multiple_person_frames)

    # if shifted_frames_by_vqa:
    #     shifted_frames_by_vqa = sorted(shifted_frames_by_vqa, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    #     display_frames("shifted_frames_by_vqa", shifted_frames_by_vqa)

    # if shifted_frames_by_vqa_ther2:
    #     shifted_frames_by_vqa_ther2 = sorted(shifted_frames_by_vqa_ther2, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    #     display_frames("shifted_frames_by_vqa_ther2", shifted_frames_by_vqa_ther2)

    # total_frames = len(no_person_frames) + len(single_person_frames) + len(multiple_person_frames)
    # st.write(f"Total frames classified: {total_frames}")
    # st.write(f"clip_count: {clip_count}")