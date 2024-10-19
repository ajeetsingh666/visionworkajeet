# import streamlit as st
from modeling_clip import CLIP
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
# import modeling_yolov8


setup_logging()
logger = logging.getLogger(__name__)

print("Ajeet Singh")
print("All three models")

# Initialize components
clip_model = CLIP(device="cpu")
vqa_model = VQADandelin()
yolov8_persondetector = YOLOv8PersonDetector()

def prediction(video_id, frames):

    text_prompts = {
    "person": [

        "a photo of a person.",
        "a picture of a person.",
        # "a photo with a person visible.",
        "a photo of a person's face.",
        "a photo of a person working",


        # "a photo of multiple people.",
        # "a photo of people interacting with each other.",
        # "a photo of multiple people communicating.",
        # "a photo of multiple people working.",
        # "a photo of multiple people speaking.",
        # "a photo of multiple people discussing.",
        # "a photo of multiple people interacting.",
        # "a photo of multiple people talking.",

        # "a photo of several people working.",
        # "a photo of several people speaking.",
        # "a photo of several people discussing.",
        # "a photo of several people interacting.",
        # "a photo of several people talking",

        # "a photo of several people engaged in conversation.",
        # "a photo of a group of people.",
        # "a photo of numerous persons.",
        # "a photo of a collective of individuals working together.",

        # "a photo with multiple people visible.",
        # "a photo where several people are visible.",


],
"no_person": [

        "a photo of an empty scene.",
        "a photo with no one in it.",
        "a photo of an empty room.",
        "a photo of a room",
        "a photo of a room with no one in it.",

        # "a photo without any individuals.",
        # # "a photo with no humans.", 
        # "an image showing an empty space.",
        # "an image devoid of persons.",
        # "a photo of a location with no one around.", 
        # "a photo of a scene without individuals.",
        # # "a photo that has no one present.",
        # "an image that does not contain any individuals.",
        # "a picture where no humans are visible.",
        # "a photo of a room that is empty.", 
        # "a photo without any face",
        # "vacant",
        # "empty",
        # "void",
        # "no person",

        # "a photo without person.",
        # "a photo without any person.",
        # "a photo with no person.", 
        # # "an image with no humans in it.",
        # "an image devoid of person.",
        # "a photo of a place with no one around.", 
        # # "a photo of a place where nobody is present.",
        # "a photo of an room with no person visible.",
        # "an image that does not contain any person.",
        # "a picture where no humans are visible.",
]
}

    # text_prompts = {
    # "person": [
    #     "A photo of a person working on a project.",
    #     "A photo of an individual engaged in a task.",
    #     "A photo of a person attending a meeting.",
    #     "A photo of multiple individuals collaborating on a project.",
    #     "A photo of several people engaged in conversation.",
    #     "A photo of a group of people working together.",
    #     "A photo of numerous persons attending a meeting.",
    #     "A photo of a collective of individuals working together.",
    #     "A photo capturing an individual face.",
    #     "An image featuring one person's face.",
    #     "A photo of a person alone indoors."
    # ],
    # "no_person": [
    #     "A photo of a place where nobody is present.",
    #     "A photo depicting a scene with nobody around.",
    #     "A photo of an empty room.",
    #     "A photo capturing a moment with no one in view.",
    #     "A photo of a scene devoid of individuals.",
    #     "A photo depicting a space absent of people."
    # ]
    # }

    # text_prompts = {
    # "single_person": [

    #     "A photo of a person working on a project.",
    #     "A photo of a person attending a meeting.",
    #     "A photo of a individual participating in an activity.",
    #     "A photo of a person working.",
    #     "A photo of a person in a busy workspace",

    #     "An image featuring person's face.", 
    #     "a picture showing a  person indoors.",
    #     "a photo of a person.",

    #     "a photo of a persons face.",
    #     "a photo of a person standing.",
    #     "a photo of a person sitting.",
    #     "a photo of a person taking a selfie",

    #     "a photo of a person working.",

    #     "A photo of multiple individuals working together.",
    #     "A photo of several people engaged in conversation.",
    #     "A photo of a group of people collaborating on a project.",
    #     "A photo of numerous persons attending a meeting.",
    #     "A photo of various individuals interacting",
    #     "A photo of several persons working on a project.",
    #     "A photo of numerous individuals working",
    #     "A photo of many people collaborating on a project.",
    #     "A photo of many individuals engaged in an activity.",
    #     "A photo of numerous people interacting",
    #     "A photo of many persons involved in a brainstorming session.",
    #     "A photo of many persons in a busy co-working space.", 
    #     "A photo of a person engaging in a conversation with another individual.",
    #     "A photo capturing a lively discussion among a group of people.",
    #     "A photo capturing a discussion among several individuals.",
    #     "A photo of a person speaking with another person.",
    #     "A photo of multiple individuals connecting in a meeting.",
    #     "A photo of multiple people communicating over a task.",
    #     "A photo of multiple people working together on a project.",
    #     "A photo of multiple people speaking",
    #     "A photo of multiple people discussing",
    #     "A photo of multiple people interacting",
    #     "A photo of multiple people talking"
    #     "A photo of multiple people talking in a brainstorming session.",
    #     "A photo of multiple individuals getting together for a collaboration.",
    #     "A photo of multiple individuals coordinating on a group effort.",
    #     "A photo of a group of people connecting over a shared task.",
    #     "A photo of a collective of individuals connecting in a meeting.",
    #     "A photo of several people communicating over a project.",
    #     "A photo of multiple people working together",
    #     "A photo of multiple individuals talking in a group setting.",
    #     "A photo of many people getting together for a collaborative task.",
    #     "A photo of multiple individuals getting together for a meeting.",
    #     "A photo of some people getting together for a project discussion.",
    #     "A photo of many individuals discussing a project.",
    #     "A photo featuring multiple faces.",
    #     "A photo of a group of faces.",
    #     "A photo of multiple people indoors.",
    #     "A picture showing a group of people indoors.",
    #     "A scene of multiple persons indoors.",
    #     "a photo of multiple people.",
    #     "a photo of people interacting with each other.",
    #     "a photo of multiple faces.",
    #     "a photo of multiple people standing.",
    #     "a photo of multiple people sitting.",
    #     "a photo of group of people standing.",
    #     "a photo of group of people sitting.",
    #     "a photo of a group taking a selfie.",
    #     "a photo of friends posing for a selfie."

    # ],
    # "no_person": [

    #     "A photo of a place where nobody is present.",
    #     "A photo depicting a scene with nobody around.",
    #     # "A photo showing nobody in the frame.",
    #     "A photo that captures an area with nobody there.",
        
    #     "A photo of an empty room.",
    #     "A photo of an empty room where no one is present.",
    #     # "A photo showcasing a setting with no one around.",
    #     "A photo depicting a scene with no one in it.",
    #     "A photo capturing a moment with no one in view.",
        
    #     "A photo with no individuals present.",
    #     "A photo depicting a space absent of people.",
    #     "A photo of a scene that is absent of people.",
    #     "A photo capturing a moment that is absent of people.",
        
    #     "A photo depicting a scene devoid of individuals.",
    #     "A photo capturing a moment that is devoid of individuals.",
        
    #     # "A photo revealing an area empty of people.",
        
    #     "A photo of a scene lacking human presence.",
    #     "A photo depicting an area lacking human presence.",
    #     "A photo capturing a moment that is lacking human presence.",

    #     "A photo of nothing but an empty space.",
    #     "A photo of a still room without people.",
    #     "A photo devoid of people",
    #     "A photo that has no one in it",
    #     "A photo where no humans are visible",
    #     "A photo of a room with no people",

    #     "a photo of no visible faces.",
    #     "A photo with no faces present.",
    #     "A photo without any visible faces.",
    #     "A photo with no visible faces.",
    #     "A photo devoid of any human faces.",
    #     "A photo where no faces are visible.",
    #     "A photo capturing an area devoid of faces.",
    #     "A photo featuring a space without faces.",
    #     "A photo of a setting where no faces are visible.",

    #     "A photo of an empty indoor space.",
    #     "An image of a vacant indoor area.",
    #     "A picture showing an unoccupied indoor space.",
    #     "A scene of an empty indoor area.",
    #     "A depiction of an empty indoor space.",


    #     # "A photo of no people.",
    #     # "A photo of an empty room with no person visible",

    #     # "a photo of no visible faces.",

    #     # "A photo with no people",
    #     # "A photo that has no one in it",

    #     "An image that does not contain any people",
    #     "An image devoid of people",

    #     # "A picture showing an empty space",

    #     "A picture where no humans are visible",
    #     # "A photo of an empty room with no people visible",
    #     # "An image showing an empty room with no people visible",

    #     # "A photo of a room with no people",
    #     # "An image showing a room with no person",

    # ]}

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
    yolo_classifications = yolov8_persondetector.classify_batch(frames, conf_threshold=0.60)
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
    # Step 2: Process YOLO results
    for frame_path, (yolo_label, yolo_prob) in zip(frames, yolo_classifications):
        if yolo_label == "no_person":  # If YOLO detects no person
            # Mark frame for further processing with CLIP
            yolo_count = yolo_count + 1
            frames_for_clip.append(frame_path)
        else:
            # If YOLO detected a person, use YOLO's results
            classified_frames.append({
                "frame_path": frame_path,
                "final_label": yolo_label,
                "final_prob": yolo_prob
            })


    # logger.info(f"frames_for_vqa: {yolo_count}")
    # frames_for_single_person_clip = []
    # if frames_for_clip:
    #     vqa_start_time = time.time()
    #     print("using yolo and vqa only")
    #     for i, frame_path in enumerate(frames_for_clip):
    #         vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are there?")
    #         final_label = vqa_label
    #         final_prob = vqa_prob

    #         if vqa_label == "no_person":
    #             vqa_culprit_for_ls = vqa_culprit_for_ls + 1
    #             # print(f"frame_path: {frame_path}")

    #         if vqa_label == "single_person":
    #             frames_for_single_person_clip.append(frame_path)
    #         else:  
    #             if vqa_label == "Uncertain":
    #                 vqa_uncertain = vqa_uncertain + 1
    #                 final_label = "single_person"
    #                 final_prob = 1

    #             # Step 6: Store the final results for the frame
    #             classified_frames.append({
    #                 "frame_path": frame_path,
    #                 "final_label": final_label,
    #                 "final_prob": final_prob
    #             })
    #     time_taken_by_vqa = time.time() - vqa_start_time
    #     logger.info(f"time_taken_by_vqa: {time_taken_by_vqa:.2f} seconds")

    #     logger.info(f"frames_for_clip_reprocessing: {len(frames_for_single_person_clip)}")
    #     if frames_for_single_person_clip:
    #         clip_start_time = time.time()

    #         _, image_embeddings = clip_model.get_image_embeddings(frames_for_clip)

    #         for frame_path, image_embedding in zip(frames_for_single_person_clip, image_embeddings):
    #             clip_label, _ = clip_model.classify_image(image_embedding, precomputed_embeddings)

    #             final_label = "single_person"
    #             final_prob = 1

    #             if clip_label == "no_person":  # If CLIP detects a person
    #                 # Step 5: Use VQA to count persons
    #                 clip_count = clip_count + 1
    #                 final_label = "no_person"
    #                 final_prob = 1

    #             classified_frames.append({
    #                 "frame_path": frame_path,
    #                 "final_label": final_label,
    #                 "final_prob": final_prob
    #             })

    #         time_taken_by_clip = time.time() - yolo_start_time
    #         logger.info(f"time_taken_by_clip_for_reprocessing: {time_taken_by_clip:.2f} seconds")


    # Step 3: Process frames with CLIP if YOLO detected "no_person"
    # if frames_for_clip:
    #     # Get image embeddings for all frames_for_clip in one batch call
    #     embedding_start_time = time.time()

    #     _, image_embeddings = clip_model.get_image_embeddings(frames_for_clip)

    #     time_taken_by_clip = time.time() - embedding_start_time
    #     # st.write(f"Time taken by combined clip and vqa: {time_taken_by_clip:.2f} seconds")
    #     logger.info(f"time_taken to extrct embeddings: {time_taken_by_clip:.2f} seconds")

    #     # Step 4: Process CLIP results
    #     for frame_path, image_embedding in zip(frames_for_clip, image_embeddings):
    #         clip_label, clip_prob = clip_model.classify_image(image_embedding, precomputed_embeddings)

    #         if clip_label == "no_person":  # If CLIP detects a person
    #             # Step 5: Use VQA to count persons
    #             clip_count = clip_count + 1
    #             final_label = "no_person"
    #             final_prob = 1

    #         else:
    #             vqa_count = vqa_count + 1
    #             vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many persons are there?")
    #             final_label = vqa_label
    #             final_prob = vqa_prob

    #             if vqa_label == "no_person":
    #                 vqa_culprit_for_ls = vqa_culprit_for_ls + 1
    #                 print(f"frame_path: {frame_path}")
    #             if vqa_label == "Uncertain":
    #                 vqa_uncertain = vqa_uncertain + 1
    #                 final_label = "single_person"
    #                 final_prob = 1

    #         # Step 6: Store the final results for the frame
    #         classified_frames.append({
    #             "frame_path": frame_path,
    #             "final_label": final_label,
    #             "final_prob": final_prob
    #         })

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

                # yolo_classifications_in_vqa_no_person = yolov8_persondetector.classify_batch([frame_path], conf_threshold=0.99)
                # yolo_label, yolo_prob = yolo_classifications_in_vqa_no_person[0]
                # if yolo_label == "single_person":
                #     ls_fp_stopped_by_yolo_single_person = ls_fp_stopped_by_yolo_single_person + 1
                #     final_label = "single_person"
                #     # print(f"frame_path: {frame_path}")

            if vqa_label == "single_person":
                yolo_classifications_in_vqa = yolov8_persondetector.classify_batch([frame_path], conf_threshold=0.001)
                yolo_label, yolo_prob = yolo_classifications_in_vqa[0]
                if yolo_label == "no_person":
                    ls_fp_stopped_by_yolo = ls_fp_stopped_by_yolo + 1
                    final_label = "no_person"
                    # print(f"frame_path: {frame_path}")
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
