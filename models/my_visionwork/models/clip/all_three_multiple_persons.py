# import streamlit as st
import cv2
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
from modeling_blip import VQABlip
# import modeling_yolov8
from modeling_yunet import FaceDetectorYuNet
from modeling_yolotiny import YolosPersonDetector


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
    # Step 2: Process YOLO results
    # for frame_path, (yolo_label, yolo_prob) in zip(frames, yolo_classifications):
    #     if yolo_label == "no_person" or yolo_label == "multiple_persons":  # If YOLO detects no person
    #         # Mark frame for further processing with CLIP
    #         yolo_count = yolo_count + 1
    #         frames_for_clip.append(frame_path)
    #     else:
    #         # If YOLO detected a person, use YOLO's results
    #         classified_frames.append({
    #             "frame_path": frame_path,
    #             "final_label": yolo_label,
    #             "final_prob": yolo_prob
    #         })

    for frame_path, (yolo_label, yolo_prob) in zip(frames, yolo_classifications):
        if yolo_label == "no_person" or yolo_label == "multiple_persons":  # If YOLO detects no person
            # Mark frame for further processing with CLIP
            yolo_count = yolo_count + 1
            frames_for_clip.append(frame_path)
        else:
            # classification, _ = face_detector.classify(frame_path)

            # if classification == "single_person":
            #     final_label = yolo_label
            #     final_prob = yolo_prob
            # else:
            #     blib_start_time = time.time()
            #     blib_label, _ = vqa_blip.classify(frame_path, "how many people are in the picture?")

            #     if blib_label == "Uncertain":
            #         blip_uncertain = blip_uncertain + 1
            #         final_label = "single_person"
            #         final_prob = 1
            #     else:
            #         final_label = blib_label
            #         final_prob = 1

            # adjusted_image_path = "/tmp/video_incidents_ajeet_temp/bright.jpg"
            # img_original = cv2.imread(frame_path)

            # alpha = 2  
            # beta = 10
            # img_contrast_bright = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
            # cv2.imwrite(adjusted_image_path, img_contrast_bright)
            # yolo_classifications = yolov8_persondetector.classify_batch([adjusted_image_path], conf_threshold=0.10)

            yolo_classifications, results = yolov8_persondetector.dummy_classify_batch([frame_path], conf_threshold=0.05)
            result = results[0]
            cropped_img_path = "/tmp/video_incidents_ajeet_temp/modified_image.jpg"
            img_original = cv2.imread(frame_path)

            # Assuming boxes_temp is the result.boxes.data from YOLOv8
            # boxes_temp = result.boxes.data
            # boxes_temp = sorted(boxes_temp, key=lambda x: x[4], reverse=True)
            # # Filter out boxes that are completely inside other boxes
            # filtered_boxes = []
            # for i, box1 in enumerate(boxes_temp):
            #     x1_1, y1_1, x2_1, y2_1 = box1[:4].tolist()  # Extract coordinates of the current box
            #     is_inside = False

            #     for j, box2 in enumerate(boxes_temp):
            #         if i != j:  # Avoid comparing the box with itself
            #             x1_2, y1_2, x2_2, y2_2 = box2[:4].tolist()  # Extract coordinates of the other box

            #             # # Check if box1 is completely inside box2
            #             # if (x1_1 >= x1_2 and x2_1 <= x2_2 and y1_1 >= y1_2 and y2_1 <= y2_2):
            #             #     is_inside = True
            #             #     break
            #                     # Check if box1 intersects with box2
            #             if not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2):
            #                 is_inside = True
            #                 break

            #     # If the box is not inside any other box, keep it
            #     if not is_inside:
            #         filtered_boxes.append(box1)

            boxes_temp = result.boxes.data
            boxes_temp = sorted(boxes_temp, key=lambda x: x[4], reverse=True) 

            filtered_boxes = []

            for i, box1 in enumerate(boxes_temp):
                x1_1, y1_1, x2_1, y2_1 = box1[:4].tolist()
                should_keep = True

                for j, box2 in enumerate(filtered_boxes):
                    x1_2, y1_2, x2_2, y2_2 = box2[:4].tolist()

                    if not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2):
                        should_keep = False
                        break

                if should_keep:
                    filtered_boxes.append(box1)

            filtered_boxes_sorted = sorted(filtered_boxes, key=lambda x: x[4], reverse=True)[1:]

            for box in filtered_boxes_sorted:
                x1, y1, x2, y2 = map(int, box[:4])
                print(f"{x1}: {y1}: {x2}: {y2}: ")

                cropped_img = img_original[y1:y2, x1:x2]
                # cropped_img = cv2.resize(cropped_img, (100, 60))
                cv2.imwrite(cropped_img_path, cropped_img)

                yolo_tiny_start_time = time.time()
                # classification, probability = yolo_tiny_detector.classify(cropped_img_path)
                classification, probability = face_detector.classify(frame_path)
                yolo_tiny_end_time = time.time() - yolo_tiny_start_time
                print(f"time_taken_by_tiny:{yolo_tiny_end_time}")

                # prevoius_prob = box[4]
                # yolo_classification, _  = yolov8_persondetector.dummy_classify_batch([frame_path], conf_threshold=0.05)
                # classification, probability = yolo_classification[0]

                print(f"YOLO-tiny classification: {classification}, Probability: {probability}")
                if classification == "single_person" or classification == "multiple_persons":
                    yolo_label = "multiple_persons"
                    break
                
                try:
                    os.remove(cropped_img_path)
                    print(f"Deleted adjusted image at: {cropped_img_path}")
                except OSError as e:
                    print(f"Error deleting file: {e}")
                # cv2.rectangle(img_original, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)

            classified_frames.append({
            "frame_path": frame_path,
            "final_label": yolo_label,
            "final_prob": yolo_prob
            })

            # if result.boxes is not None and len(result.boxes) > 0:
            #     boxes_temp = result.boxes.data
            #     boxes = sorted(boxes_temp, key=lambda x: x[4], reverse=True)[1:]
            #     for idx, box in enumerate(boxes):
            #         # x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            #         x1 = int(box[0])
            #         y1 = int(box[1])
            #         x2 = int(box[2])
            #         y2 = int(box[3])
            #         print(f"{x1}: {y1}: {x2}: {y2}: ")

            #         cropped_img = img_original[y1:y2, x1:x2]
            #         # cropped_img = cv2.resize(cropped_img, (100, 60))
            #         cv2.imwrite(cropped_img_path, cropped_img)
            #         classification, probability = yolo_tiny_detector.classify(cropped_img_path)
            #         print(f"YOLO-tiny classification: {classification}, Probability: {probability}")
            #         if classification == "single_person":
            #             yolo_label = "multiple_persons"
            #             break
                    
            #         try:
            #             os.remove(cropped_img_path)
            #             print(f"Deleted adjusted image at: {cropped_img_path}")
            #         except OSError as e:
            #             print(f"Error deleting file: {e}")
            #         # cv2.rectangle(img_original, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)

            # classified_frames.append({
            # "frame_path": frame_path,
            # "final_label": yolo_label,
            # "final_prob": yolo_prob
            # })

          # Save the modified image with the white rectangle
            # modified_image_path = "/tmp/video_incidents_ajeet_temp/modified_image.jpg"
            # cv2.imwrite(modified_image_path, img_original)


            # adjusted_image_path = "/tmp/video_incidents_ajeet_temp/bright.jpg"
            # img_original = cv2.imread(modified_image_path)
            # alpha = 1.5
            # beta = 30
            # img_contrast_bright = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
            # cv2.imwrite(adjusted_image_path, img_contrast_bright)
            # second_results_classification, _ = yolov8_persondetector.dummy_classify_batch([adjusted_image_path], conf_threshold=0.20)

            # second_results_classification, _ = yolov8_persondetector.dummy_classify_batch([modified_image_path], conf_threshold=0.10)

            # second_yolo_label, second_yolo_prob = second_results_classification[0]

            # if second_yolo_label == "single_person":
            #     print(f"Changed_by_second_yolo_call: {frame_path}")
            #     yolo_label = "multiple_persons"

            # classified_frames.append({
            # "frame_path": frame_path,
            # "final_label": yolo_label,
            # "final_prob": yolo_prob
            # })

            # try:
            #     os.remove(modified_image_path)
            #     print(f"Deleted adjusted image at: {modified_image_path}")
            # except OSError as e:
            #     print(f"Error deleting file: {e}")

            # try:
            #     os.remove(adjusted_image_path)
            #     print(f"Deleted adjusted image at: {adjusted_image_path}")
            # except OSError as e:
            #     print(f"Error deleting file: {e}")



    # for frame_path in frames:
    #     classification, probability = face_detector.classify(frame_path)
    #     if classification == "no_person" or classification == "multiple_persons":  # If YOLO detects no person
    #         # Mark frame for further processing with CLIP
    #         yolo_count = yolo_count + 1
    #         frames_for_clip.append(frame_path)
    #     else:
    #         # If YOLO detected a person, use YOLO's results
    #         classified_frames.append({
    #             "frame_path": frame_path,
    #             "final_label": yolo_label,
    #             "final_prob": yolo_prob
    #         })

    logger.info(f"frames_for_clip: {yolo_count}")
    if frames_for_clip:
        vqa_start_time = time.time()
        print("using yolo and vqa only")
        for i, frame_path in enumerate(frames_for_clip):
            vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are clearly present there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many fully visible people are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="how many clearly visible people are present there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many clearly visible people are in the picture?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are clearly present in the picture?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many living humans are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many living people are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many real people are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="how many fully visible human faces are there?")
            # vqa_label, vqa_prob = vqa_model.classify(frame_path, query="How many people are clearly visible and recognizable in the picture?")
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

                # Load the original image
                # image_path = frame_path
                # image = cv2.imread(image_path)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                # # padding_value = 5
                # # padded_image = apply_padding(image, padding_value)
                # # print(padded_image.shape)
                # # plt.imshow(padded_image)

                # # Apply striding to the image
                # strided_image = apply_striding(image, 5)
                # strided_image_path = "/home/ajeet/codework/stripped_image/strided_image.jpg"
                # cv2.imwrite(strided_image_path, cv2.cvtColor(strided_image, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for saving
                # raw_image = Image.open(strided_image_path).convert('RGB')

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
                
                # if os.path.exists(strided_image_path):
                #     os.remove(strided_image_path)
                #     print(f"Deleted: {strided_image_path}")
                # else:
                #     print(f"File not found: {strided_image_path}")

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

    return incidents


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
