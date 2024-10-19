import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append("/home/ajeet/codework/ujjawal_github/visionwork/models/clip")

print("\n".join(sys.path))



import cv2
import os
import time
from PIL import Image
from modeling_yolo_finetuned import YOLOv8
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F
# from transformers import CLIPProcessor, CLIPModel
from transformers import DetrImageProcessor, DetrForObjectDetection
# from transformers import YolosImageProcessor, YolosForObjectDetection


import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


print("Ajeet Singh")
print("yolo_clip_phone_detection")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# list_of_classes = ["a person", "a cell phone"]

# you can specify the revision tag if you don't want the timm dependency
# detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# yolo_tiny_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
# yolo_tiny_image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    # Load a pre-trained RetinaNet model
# retinanet_model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
# retinanet_model.eval()  # Set the model to evaluation mode


# ssdlite_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
# ssdlite_model.eval()  # Set the model to evaluation mode

# ssd_model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
# ssd_model.eval()  # Set the model to evaluation mode


base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                    score_threshold=0.20)
detector = vision.ObjectDetector.create_from_options(options)

def prediction(images):

    yolo_thershold = 0.50
    verify_yolo_by_clip_below_ther = 0.40
    # detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx" , yolo_thershold, 0.40)

    frames_result = []
    for image in images:
        # final_label = "no_cell_phone"
        # output_image, is_phone_detected, yolo_phone_score = detection.main(image)
        # # print(f"{image}: {is_phone_detected} {yolo_phone_score} ")
        # if is_phone_detected:
        #     final_label = list_of_classes[1]

        clip_prob = -1
        # if 0.0001 < yolo_phone_score < verify_yolo_by_clip_below_ther:
        #     class_detected, clip_prob = clip_classifier(image)
        #     base_name = os.path.basename(image)
        #     if class_detected == list_of_classes[1]:
        #         final_label = list_of_classes[1]
        #         # print(f"cell_phone detected by clip for {base_name}::::: clip_result: {class_detected} {clip_prob} True")
        #     else:
        #         final_label = "no_cell_phone"
        #         # print(f"cell_phone_not detected by clip for {base_name}::::: clip_result: {class_detected} {clip_prob} False")



        # yolo_phone_score = -1 
        # class_detected, clip_prob = clip_classifier(image)
        # if class_detected == list_of_classes[1]:
        #     final_label = list_of_classes[1]
        # else:
        #     final_label = "no_cell_phone"

        # frames_result.append({
        #         "frame_path": image,
        #         "final_label": final_label,
        #         "yolo_phone_score": yolo_phone_score,
        #         "clip_prob": clip_prob
        #     })
        

        start_time = time.time()
        # detr_classify(image)
        # yolo_tiny(image)
        # retinanet(image)
        # ssdlite(image)
        # ssd(image)

        score, is_detected = efficientnet(image)
        if is_detected:
            print(f"{image} True {score}")
        else:
            print(f"{image} False {score}")

        end_time = time.time() - start_time
        # print("time_taken:", end_time)



    # print("-"*50)
    # for result in frames_result:
    #     print(result)
        

# def clip_classifier(image):
#     frame_path = image
#     image = Image.open(image)

#     inputs = processor(text=list_of_classes, images=image, return_tensors="pt", padding=True)

#     with torch.inference_mode():
#         outputs = model(**inputs)

#     logits_per_image = outputs.logits_per_image # image-text similarity score
#     probs = logits_per_image.softmax(dim=1) # label probabilities
#     max_prob_index = int(torch.argmax(probs))

#     return list_of_classes[max_prob_index], probs


# def detr_classify(image):
#     image = Image.open(image)
#     with torch.inference_mode():
#         inputs = detr_processor(images=image, return_tensors="pt")
#         outputs = detr_model(**inputs)

#     # convert outputs (bounding boxes and class logits) to COCO API
#     # let's only keep detections with score > 0.9
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(detr_model.config.id2label[label.item()])
#         # print(
#         #         f"Detected {detr_model.config.id2label[label.item()]} with confidence "
#         #         f"{round(score.item(), 3)} at location {box}"
#         # )

# def yolo_tiny(image):
#     frame_path = image
#     image = Image.open(image)
#     with torch.inference_mode():
#         inputs = yolo_tiny_image_processor(images=image, return_tensors="pt")
#         outputs = yolo_tiny_model(**inputs)

#     # model predicts bounding boxes and corresponding COCO classes
#     logits = outputs.logits
#     bboxes = outputs.pred_boxes


#     # print results
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = yolo_tiny_image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(f"{frame_path} {yolo_tiny_model.config.id2label[label.item()]}")
#         # print(
#         #     f"Detected {yolo_tiny_model.config.id2label[label.item()]} with confidence "
#         #     f"{round(score.item(), 3)} at location {box}"
#         # )


# def retinanet(image):
#     image_path = image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_tensor = F.to_tensor(image_rgb)

#     # Add batch dimension
#     image_tensor = image_tensor.unsqueeze(0)

#     # Perform object detection
#     with torch.no_grad():
#         outputs = retinanet_model(image_tensor)

#     # Post-process the results
#     scores = outputs[0]['scores'].numpy()
#     boxes = outputs[0]['boxes'].numpy()
#     labels = outputs[0]['labels'].numpy()

#     # COCO class ID for "cell phone"
#     PHONE_CLASS_ID = 77

#     # Filter boxes for the "cell phone" class with a confidence score above a threshold (e.g., 0.5)
#     threshold = 0.5
#     phone_indices = (labels == PHONE_CLASS_ID) & (scores > threshold)
#     phone_boxes = boxes[phone_indices]
#     phone_scores = scores[phone_indices]
#     print(f"{image_path} {phone_scores}")


# def ssdlite(image):
#     image_path = image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_tensor = F.to_tensor(image_rgb)

#     # Add batch dimension
#     image_tensor = image_tensor.unsqueeze(0)

#     # Perform object detection
#     with torch.no_grad():
#         outputs = ssdlite_model(image_tensor)

#     # Post-process the results
#     scores = outputs[0]['scores'].numpy()
#     boxes = outputs[0]['boxes'].numpy()
#     labels = outputs[0]['labels'].numpy()

#     # COCO class ID for "cell phone"
#     PHONE_CLASS_ID = 77

#     # Filter boxes for the "cell phone" class with a confidence score above a threshold (e.g., 0.5)
#     threshold = 0.5
#     phone_indices = (labels == PHONE_CLASS_ID) & (scores > threshold)
#     phone_boxes = boxes[phone_indices]
#     phone_scores = scores[phone_indices]
#     print(f"{image_path} {phone_scores}")

# def ssd(image):
#     image_path = image
#     image = cv2.imread(image_path)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_tensor = F.to_tensor(image_rgb)

#     # Add batch dimension
#     image_tensor = image_tensor.unsqueeze(0)

#     # Perform object detection
#     with torch.no_grad():
#         outputs = ssd_model(image_tensor)

#     # Post-process the results
#     scores = outputs[0]['scores'].numpy()
#     boxes = outputs[0]['boxes'].numpy()
#     labels = outputs[0]['labels'].numpy()

#     # COCO class ID for "cell phone"
#     PHONE_CLASS_ID = 77

#     # Filter boxes for the "cell phone" class with a confidence score above a threshold (e.g., 0.5)
#     threshold = 0.5
#     phone_indices = (labels == PHONE_CLASS_ID) & (scores > threshold)
#     phone_boxes = boxes[phone_indices]
#     phone_scores = scores[phone_indices]
#     print(f"{image_path} {phone_scores}")

def efficientnet(myimage):
    IMAGE_FILE = myimage
    img = cv2.imread(IMAGE_FILE)
    image = mp.Image.create_from_file(IMAGE_FILE)

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)



    # output_path = "/home/ajeet/codework/efficientnet_results/" + os.path.basename(myimage)
    # cv2.imwrite(output_path, cv2.cvtColor(rgb_annotated_image, cv2.COLOR_RGB2BGR))


    for detection in detection_result.detections:
        for category in detection.categories:
            # print(category)
            if category.category_name == 'cell phone':  # Threshold can be adjusted
                return category.score , True
    
    return -1, False


def filter_cell_phone_detections(detection_result):
    filtered_detections = []
    for detection in detection_result.detections:
        for category in detection.categories:
            if category.category_name == 'cell phone':  # Adjust score threshold if needed
                filtered_detections.append(detection)
    return filtered_detections



#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image




if __name__ == "__main__":
    
    start_time = time.time()

    print("hi")

    # images = [os.path.join("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged") if filename.startswith("0_")]

    # images = [os.path.join("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/ed2e9e4c-ec0e-437d-a13d-5a1b87340c44_20240911181755971334_20240911181756999046_0_merged_140332") if filename.startswith("0_")]

    # images = [os.path.join("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/9aa3bbb8-9cc2-48a9-a2a6-aa9c86acf71b_20240911182617744144_20240911182619026111_0_merged") if filename.startswith("0_")]

    images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
    for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]

    # images = [os.path.join("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged" , filename) 
    # for filename in os.listdir("/tmp/video_incidents_ajeet/b2045fbc-5af3-4b40-a166-072b90f803e5_merged") if filename.startswith("0_")]

    # images = [os.path.join("/home/ajeet/codework/Cellphone_train/train" , filename) 
    # for filename in os.listdir("/home/ajeet/codework/Cellphone_train/train")]

    images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    # images = images[:5]
    print(len(images))
    prediction(images)

    end_time = time.time() - start_time

    print(f"time_taken: {end_time}")    