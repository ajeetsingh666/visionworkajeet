
# import sys
# import os

# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
# sys.path.append("/home/ajeet/codework/ujjawal_github/visionwork/models/clip")

# print("\n".join(sys.path))


# from modeling_clip import CLIP
# import time
# import os

# clip_model = CLIP(device="cpu")

# # text_prompts = {
# # "phone": [
# #     # "a photo of a person using a cell phone.",
# #     "a cell phone is visible in the photo."


# # ],
# # "no_phone": [
# #     # "a photo of a person not using a cell phone.",
# #     "cell phone is not visible in the photo."
# # ]
# # }

# text_prompts = {
#     "phone": [
#         "a cell phone is clearly visible in the image.",
#     ],
#     "no_phone": [
#         "no cell phone is visible in the image.",
#     ]
# }


# images = [os.path.join("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged" , filename) 
# for filename in os.listdir("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged") ]

# # images = images[:10]

# images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

# start_time = time.time()
# _, image_embeddings = clip_model.get_image_embeddings(images)
# time_taken_by_embeddings = time.time() - start_time
# print(f"Time taken to extract embeddings: {time_taken_by_embeddings:.2f} seconds")

# precomputed_embeddings = clip_model.precompute_prompt_embeddings(text_prompts)

# clip_results = []
# for frame_path, image_embedding in zip(images, image_embeddings):
#     clip_label, clip_prob = clip_model.classify_image(image_embedding, precomputed_embeddings)
#     # clip_prob = 1
#     clip_results.append((frame_path, clip_label, clip_prob))  # Store frame path, label, and probability

# for result in clip_results:
#     print(result)


############################

from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import os
import time

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# images = [os.path.join("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged" , filename) 
# for filename in os.listdir("/tmp/video_incidents_ajeet/9d47d182-296b-47d3-a64f-d6a2fd06f9f3_merged") ]

images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]

# images = images[:10]

images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

# Define the list of classes 

start_time = time.time()
list_of_classes = ['human', "phone"]
clip_prob_list = []
for image in images:
    frame_path = image
    image = Image.open(image)

    inputs = processor(text=list_of_classes, images=image, return_tensors="pt", padding=True)

    with torch.inference_mode():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    # print(probs)
    max_prob_index = int(torch.argmax(probs))
    # print(list_of_classes[max_prob_index])

    clip_prob_list.append((frame_path, list_of_classes[max_prob_index], probs))

for result in clip_prob_list:
    print(result)

end_time = time.time() - start_time

print(f"time_taken: {end_time}")


##############################


# Ultralytics YOLO 🚀, AGPL-3.0 license

# import argparse

# import cv2
# import numpy as np
# import onnxruntime as ort
# import torch
# import os
# import time

# from ultralytics.utils import ASSETS, yaml_load
# from ultralytics.utils.checks import check_requirements, check_yaml


# class YOLOv8:
#     """YOLOv8 object detection model class for handling inference and visualization."""

#     def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
#         """
#         Initializes an instance of the YOLOv8 class.

#         Args:
#             onnx_model: Path to the ONNX model.
#             input_image: Path to the input image.
#             confidence_thres: Confidence threshold for filtering detections.
#             iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
#         """
#         self.onnx_model = onnx_model
#         self.input_image = input_image
#         self.confidence_thres = confidence_thres
#         self.iou_thres = iou_thres

#         # Load the class names from the COCO dataset
#         self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]

#         # Generate a color palette for the classes
#         self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

#     def draw_detections(self, img, box, score, class_id):
#         """
#         Draws bounding boxes and labels on the input image based on the detected objects.

#         Args:
#             img: The input image to draw detections on.
#             box: Detected bounding box.
#             score: Corresponding detection score.
#             class_id: Class ID for the detected object.

#         Returns:
#             None
#         """
#         # Extract the coordinates of the bounding box
#         x1, y1, w, h = box

#         # Retrieve the color for the class ID
#         color = self.color_palette[class_id]

#         # Draw the bounding box on the image
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

#         # Create the label text with class name and score
#         label = f"{self.classes[class_id]}: {score:.2f}"

#         # Calculate the dimensions of the label text
#         (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

#         # Calculate the position of the label text
#         label_x = x1
#         label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

#         # Draw a filled rectangle as the background for the label text
#         cv2.rectangle(
#             img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
#         )

#         # Draw the label text on the image
#         cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

#     def preprocess(self):
#         """
#         Preprocesses the input image before performing inference.

#         Returns:
#             image_data: Preprocessed image data ready for inference.
#         """
#         # Read the input image using OpenCV
#         self.img = cv2.imread(self.input_image)

#         # Get the height and width of the input image
#         self.img_height, self.img_width = self.img.shape[:2]

#         # Convert the image color space from BGR to RGB
#         img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

#         # Resize the image to match the input shape
#         img = cv2.resize(img, (self.input_width, self.input_height))

#         # Normalize the image data by dividing it by 255.0
#         image_data = np.array(img) / 255.0

#         # Transpose the image to have the channel dimension as the first dimension
#         image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

#         # Expand the dimensions of the image data to match the expected input shape
#         image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

#         # Return the preprocessed image data
#         return image_data

#     def postprocess(self, input_image, output):
#         """
#         Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

#         Args:
#             input_image (numpy.ndarray): The input image.
#             output (numpy.ndarray): The output of the model.

#         Returns:
#             numpy.ndarray: The input image with detections drawn on it.
#         """
#         # Transpose and squeeze the output to match the expected shape
#         outputs = np.transpose(np.squeeze(output[0]))

#         # Get the number of rows in the outputs array
#         rows = outputs.shape[0]

#         # Lists to store the bounding boxes, scores, and class IDs of the detections
#         boxes = []
#         scores = []
#         class_ids = []
#         phone_detected = False

#         # Calculate the scaling factors for the bounding box coordinates
#         x_factor = self.img_width / self.input_width
#         y_factor = self.img_height / self.input_height

#         # Iterate over each row in the outputs array
#         for i in range(rows):
#             # Extract the class scores from the current row
#             classes_scores = outputs[i][4:]
#             # print(classes_scores)

#             # num_classes = len(classes_scores)
#             # print(num_classes)

#         # # Check if the phone class (assumed ID 67) is within the range of class IDs
#         #     if 67 < num_classes:
#         #         phone_score = classes_scores[67]
#         #         if phone_score >= self.confidence_thres:
#         #             phone_detected = True

#         #             # Extract the bounding box coordinates from the current row
#         #             x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

#         #             # Calculate the scaled coordinates of the bounding box
#         #             left = int((x - w / 2) * x_factor)
#         #             top = int((y - h / 2) * y_factor)
#         #             width = int(w * x_factor)
#         #             height = int(h * y_factor)

#         #             # Add the class ID (phone), score, and box coordinates to the respective lists
#         #             class_ids.append(67)  # Phone class ID
#         #             scores.append(phone_score)
#         #         boxes.append([left, top, width, height])

#             # Find the maximum score among the class scores
#             max_score = np.amax(classes_scores)
#             # print(max_score)

#             # If the maximum score is above the confidence threshold
#             if max_score >= self.confidence_thres:
#                 # Get the class ID with the highest score
#                 class_id = np.argmax(classes_scores)
#                 # print(f"class_id: {class_id}")

#                 if class_id == 2:
#                     phone_detected = True
#                     # Extract the bounding box coordinates from the current row
#                     x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

#                     # Calculate the scaled coordinates of the bounding box
#                     left = int((x - w / 2) * x_factor)
#                     top = int((y - h / 2) * y_factor)
#                     width = int(w * x_factor)
#                     height = int(h * y_factor)

#                     # Add the class ID, score, and box coordinates to the respective lists
#                     class_ids.append(class_id)
#                     scores.append(max_score)
#                     boxes.append([left, top, width, height])

#         # Apply non-maximum suppression to filter out overlapping bounding boxes
#         indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

#         # Iterate over the selected indices after non-maximum suppression
#         for i in indices:
#             # Get the box, score, and class ID corresponding to the index
#             box = boxes[i]
#             score = scores[i]
#             class_id = class_ids[i]

#             # Draw the detection on the input image
#             self.draw_detections(input_image, box, score, class_id)

#         # Return the modified input image
#         return input_image , phone_detected

#     def main(self):
#         """
#         Performs inference using an ONNX model and returns the output image with drawn detections.

#         Returns:
#             output_img: The output image with drawn detections.
#         """
#         # Create an inference session using the ONNX model and specify execution providers
#         session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

#         # Get the model inputs
#         model_inputs = session.get_inputs()

#         # Store the shape of the input for later use
#         input_shape = model_inputs[0].shape
#         self.input_width = input_shape[2]
#         self.input_height = input_shape[3]

#         # Preprocess the image data
#         img_data = self.preprocess()

#         # Run inference using the preprocessed image data
#         outputs = session.run(None, {model_inputs[0].name: img_data})

#         # Perform post-processing on the outputs to obtain output image.
#         return self.postprocess(self.img, outputs) # output image

# def save_image(image, path, image_name):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     cv2.imwrite(os.path.join(path, image_name), image)

# if __name__ == "__main__":
#     # Create an argument parser to handle command-line arguments
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
#     # parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image.")
#     # parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
#     # parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
#     # args = parser.parse_args()

#     # Check the requirements and select the appropriate backend (CPU or GPU)
#     # check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

#     # Create an instance of the YOLOv8 class with the specified arguments
#     # detection = YOLOv8(args.model, args.img, args.conf_thres, args.iou_thres)
#     images = [os.path.join("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged" , filename) 
#     for filename in os.listdir("/tmp/video_incidents_ajeet/f044cf1d-8a5b-4703-a05c-3b57c5c14989_merged") ]

#     # images = images[:1]

#     images = sorted(images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

#     start_time = time.time()
#     for image in images:
#         detection = YOLOv8("/home/ajeet/Downloads/od_v8_nano_feb24_2.onnx",image, 0.50, 0.40)

#         # Perform object detection and obtain the output image
#         output_image, phone_detected = detection.main()
        
#         if phone_detected:
#             print(f"Phone detected in image: {image}")
#         else:
#             print(f"No phone detected in image: {image}")



#         # save_path_with_phone = "/home/ajeet/codework/finetuned_yolo"
#         # image_name = os.path.basename(detection.input_image)
#         # save_image(output_image, save_path_with_phone, image_name)

#         # Display the output image in a window
#         # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#         # cv2.imshow("Output", output_image)

#         # # Wait for a key press to exit
#         # cv2.waitKey(10000)
#         # cv2.destroyAllWindows()

#     end_time = time.time() - start_time

#     print(f"time_taken: {end_time}")