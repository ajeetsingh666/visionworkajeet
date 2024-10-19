# from transformers import DetrImageProcessor, DetrForObjectDetection
# import torch
# from PIL import Image
# import requests
# import time 

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # you can specify the revision tag if you don't want the timm dependency
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# for i in range(100):
#     start_time = time.time()
#     with torch.inference_mode():
#         inputs = processor(images=image, return_tensors="pt")
#         outputs = model(**inputs)

#     # convert outputs (bounding boxes and class logits) to COCO API
#     # let's only keep detections with score > 0.9
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         print(
#                 f"Detected {model.config.id2label[label.item()]} with confidence "
#                 f"{round(score.item(), 3)} at location {box}"
#         )

#     end_time = time.time() - start_time

#     print(f"end_time: {end_time}")

#########################################################################################

# import torch
# import torchvision
# from torchvision import models
# from torchvision import transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# import time

# model = models.detection.retinanet_resnet50_fpn(pretrained=True)

# tf_ = T.ToTensor()

# img = Image.open("/home/ajeet/testing/data/using_cell_phone.jpeg")
# transformed = tf_(img)
# batched = transformed.unsqueeze(0) # model input
# int_img = torch.tensor(transformed * 255, dtype=torch.uint8)

# model = model.eval() # Make sure to not forget this
# for i in range(100):
#     start_time = time.time()
#     with torch.inference_mode(): #speeds up inference by turning off autograd
#         outputs = model(batched)


#     # COCO class index for "person" is 1
#     person_class_idx = 1

#     # Iterate through detections and keep only "person" class detections
#     for i, (box, label, score) in enumerate(zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores'])):
#         if label.item() == person_class_idx and score.item() > 0.5:  # Set a threshold for confidence
#             print(f"Detected 'person' with confidence {score.item()} at location {box.tolist()}")

#     end_time = time.time() - start_time

#     print(f"end_time: {end_time}")

########################################################################################################

# from optimum.onnxruntime import ORTModel
# from transformers import DetrImageProcessor
# import torch
# from PIL import Image
# import requests

# # Load the image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # Load the processor
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# # Load the model using ORTModel
# model = ORTModel.from_pretrained("facebook/detr-resnet-50", revision="no_timm", from_transformers=True)

# # Prepare inputs
# inputs = processor(images=image, return_tensors="pt")

# # Run inference using the ONNX model
# outputs = model(**inputs)

# # Post-process the outputs
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# # Display results
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {model.config.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )


# import onnxruntime as ort
# from transformers import DetrImageProcessor
# import torch
# from PIL import Image
# import requests

# # Load the image
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # Load the processor
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# # Prepare inputs
# inputs = processor(images=image, return_tensors="pt")
# pixel_values = inputs['pixel_values'].numpy()  # Convert to numpy array

# # Load the ONNX model using onnxruntime
# ort_session = ort.InferenceSession("path/to/detr_resnet50.onnx")

# # Run inference
# outputs = ort_session.run(None, {"pixel_values": pixel_values})

# # Convert outputs to tensors
# logits = torch.tensor(outputs[0])
# pred_boxes = torch.tensor(outputs[1])

# # Post-process the outputs
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection({"logits": logits, "pred_boxes": pred_boxes},
#                                                   target_sizes=target_sizes, threshold=0.9)[0]

# # Display results
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 2) for i in box.tolist()]
#     print(
#         f"Detected {processor.id2label[label.item()]} with confidence "
#         f"{round(score.item(), 3)} at location {box}"
#     )


# from ultralytics import RTDETR
# import time
# from PIL import Image
# import requests
# import torch
# # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("rtdetr-l.pt")

# # Display model information (optional)
# model.info()

# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open("/home/ajeet/codework/ujjawal_github/Dataset/benchmark_dataset_13_1_20/2529597/onboarding/scans_1.jpg")
# for i in range(100):
#     start_time = time.time()
#     with torch.no_grad():
#         results = model(image)
#     end_time = time.time() - start_time

#     print(f"time: {end_time}")
# # print(results)


import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time

# Load the ONNX model
model_path = '/home/ajeet/codework/ujjawal_github/visionwork/models/clip/rtdetr-l.onnx'  # Update this path
ort_session = ort.InferenceSession(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Resize the image to 640x640
    image = image.resize((640, 640))
    # Convert the image to a numpy array
    image_array = np.array(image).astype(np.float32)
    # Normalize the image (if needed, depending on the training normalization)
    image_array /= 255.0
    # Transpose dimensions to match (1, 3, 640, 640) BCHW format
    image_array = np.transpose(image_array, (2, 0, 1))
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Preprocess the input image
image_path = '/home/ajeet/codework/ujjawal_github/Dataset/benchmark_dataset_13_1_20/2529597/onboarding/scans_1.jpg'  # Update with the path to your image
input_image = preprocess_image(image_path)

for i in range(100):
    # Run inference
    start_time = time.time()
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_image})

    # Post-process the output (parsing detection results)
    # Assuming the output has shape (1, 300, 84), with 80 classes + 4 bbox coordinates per detection
    detections = outputs[0]
    boxes = detections[0, :, :4]  # Extract bounding boxes
    scores = detections[0, :, 4:]  # Extract class scores

    # Example: print the first detection
    # print("Bounding Box (x, y, w, h):", boxes[0])
    # print("Class Scores:", scores[0])
    end_time = time.time() - start_time
    print(f"time: {end_time}")