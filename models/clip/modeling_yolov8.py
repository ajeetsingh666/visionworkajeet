import cv2
from ultralytics import YOLO
import torch
import time
import os

class YOLOv8PersonDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.classes_to_detect = [0]  # Class 0 for "person" in COCO dataset

    def predict(self, image_path, conf_threshold=0.5):
        results = self.model.predict(image_path, conf=conf_threshold, device=self.device, classes=self.classes_to_detect)
        return results[0]
    
    def classify(self, image_path, conf_threshold=0.5):
        results = self.predict(image_path, conf_threshold=conf_threshold)

        # Count the number of detected persons
        person_count = len(results.boxes) 
        # person_count = results[].boxes.shape[0]
        if person_count == 0:
            classification = "no_person"
        elif person_count == 1:
            classification = "single_person"
        else:
            classification = "multiple_persons"

        # Get the highest confidence score for detected persons
        # max_conf = max([box.conf.item() for box in results[0].boxes], default=0)
        max_conf = 1

        return classification, max_conf
    
    def batch_predict(self, image_paths, conf_threshold):
        # results = self.model.predict(image_paths, conf=conf_threshold, device=self.device, imgsz=320, classes=self.classes_to_detect, show_labels=False)
        results = self.model.predict(image_paths, conf=conf_threshold, device=self.device, imgsz=320, classes=self.classes_to_detect, 
                                     stream=True, show_labels=False, show_conf=False, iou=0.40)
        return results
    

    # def classify_batch(self, image_paths, conf_threshold=0.5):
    #     # Run batch prediction
    #     results = self.batch_predict(image_paths, conf_threshold=conf_threshold)

    #     classifications = []
    #     for result in results:
    #         # Extract bounding boxes for each result
    #         boxes = result.boxes

    #         # Count the number of detected persons
    #         person_count = len(boxes)

    #         if person_count == 0:
    #             classification = "no_person"
    #         elif person_count == 1:
    #             classification = "single_person"
    #         else:
    #             classification = "multiple_persons"

    #         # Get the highest confidence score for this image
    #         # max_conf = torch.max(boxes.conf).item() if person_count > 0 else 0
    #         max_conf = 1

    #         classifications.append((classification, max_conf))

    #     return classifications

    def classify_batch(self, image_paths, conf_threshold):

        # Stream prediction
        classifications = []
        for result in self.batch_predict(image_paths, conf_threshold=conf_threshold):
            boxes = result.boxes

            # Count the number of detected persons
            person_count = len(boxes)

            if person_count == 0:
                classification = "no_person"
            elif person_count == 1:
                classification = "single_person"
            else:
                classification = "multiple_persons"

            # Get the highest confidence score for this image
            max_conf = torch.max(boxes.conf).item() if person_count > 0 else 0

            classifications.append((classification, max_conf))

        return classifications


# Usage example
if __name__ == "__main__":
    detector = YOLOv8PersonDetector(model_path="yolov8n.pt", device="cpu")
    # classification, confidence = detector.classify("/home/ajeet/testing/data/dog.jpg", conf_threshold=0.5)
    # print(f"Classification: {classification}, Confidence: {confidence}")

     # Batch image classification
    start_time = time.time()
    # image_paths = ["/home/ajeet/testing/data/dog.jpg", "/home/ajeet/testing/data/dog.jpg", "/home/ajeet/testing/data/dog.jpg"]
    folder_path = "/home/ajeet/codework/dataset_frames/2529909"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

    print("Total images:", len(image_paths))
    classifications = detector.classify_batch(image_paths, conf_threshold=0.5)
    for i, (classification, confidence) in enumerate(classifications):
        print(f"Batch image {i+1} - Classification: {classification}, Confidence: {confidence}")
    total_time = time.time() - start_time
    print("total_time", total_time)
