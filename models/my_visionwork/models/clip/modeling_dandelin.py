import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class VQADandelin:
    def __init__(self, model_name="dandelin/vilt-b32-finetuned-vqa", device="cpu"):
        self.device = device
        self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.processor = ViltProcessor.from_pretrained(model_name)
    
    def predict(self, image_path, question):
        image = Image.open(image_path)

        with torch.inference_mode():
            encoding = self.processor(image, question, return_tensors="pt")
            outputs = self.model(**encoding)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        max_prob, idx = torch.max(probs, dim=-1)

        logits = outputs.logits
        idx = logits.argmax(-1).item()

        # return self.model.config.id2label[idx], max_prob.item()

        return {
            "predicted_answer": self.model.config.id2label[idx],
            "probability": max_prob.item()
        }
    
    def classify(self, image_path, query):
        result = self.predict(image_path, query)
        answer = result["predicted_answer"].lower()

        try:
            answer_number = int(answer)
        except ValueError:
            return "Uncertain", result["probability"]

        if answer_number == 0:
            classification = "no_person"
        elif answer_number == 1:
            classification = "single_person"
        elif answer_number >= 2:
            classification = "multiple_persons"
        else:
            classification = "Uncertain"

        return classification, result["probability"]
    

    def detect_phone(self, image_path, query):
        result = self.predict(image_path, query)
        answer = result["predicted_answer"].lower()

        try:
            answer_number = int(answer)
        except ValueError:
            return "Uncertain", result["probability"]

        if answer_number == 0:
            classification = "no_person"
        elif answer_number == 1:
            classification = "single_person"
        elif answer_number >= 2:
            classification = "multiple_persons"
        else:
            classification = "Uncertain"

        return classification, result["probability"]
