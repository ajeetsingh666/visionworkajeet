from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2

class YolosPersonDetector:
    def __init__(self, model_name='hustvl/yolos-tiny', threshold=0.20):
        self.threshold = 0.20
        self.model = YolosForObjectDetection.from_pretrained(model_name)
        self.processor = YolosImageProcessor.from_pretrained(model_name)

    def detect(self, image_path):
        # Load the image
        image = Image.open(image_path)
        # Preprocess the image

        # Perform object detection
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

        # Post-process the outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=self.threshold, target_sizes=target_sizes)[0]

        # Extract detected objects
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()]
            detections.append({
                "label": label_name,
                "score": round(score.item(), 3),
                "box": box
            })
        
        return detections

    def classify(self, image_path):
        detections = self.detect(image_path)

        # Check if any person is detected
        person_detections = [d for d in detections if d['label'] == 'person']
        num_persons = len(person_detections)
        prob = max([d['score'] for d in person_detections], default=0)  # Highest confidence score

        classification = "Uncertain"
        if num_persons == 0:
            classification = "no_person"
        elif num_persons == 1:
            classification = "single_person"
        else:
            classification = "multiple_persons"

        return classification, prob

if __name__ == "__main__":
    yolos_detector = YolosPersonDetector()

    # Example image path
    # image_path = "/tmp/video_incidents_ajeet/2538482/0_1.jpg"
    image_path = "/home/ajeet/testing/data/dog.jpg"
    classification, probability = yolos_detector.classify(image_path)
    print(f"Classification: {classification}, Probability: {probability}")
