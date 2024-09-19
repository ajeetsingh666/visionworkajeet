from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import time

# you can specify the revision tag if you don't want the timm dependency
processor_fc = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model_fc = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")


start_time = time.time()
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"

for i in range(1, 100):
    image = Image.open(f"/home/ajeet/codework/dataset_frames/2648356/0_{i}.jpg")

    with torch.no_grad():
            inputs = processor_fc(images=image, return_tensors="pt")
            outputs = model_fc(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor_fc.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model_fc.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
    print("+")

print("Time", time.time() - start_time)