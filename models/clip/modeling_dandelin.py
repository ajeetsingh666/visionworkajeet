import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class VQADandelin:
    def __init__(self, model_name="dandelin/vilt-b32-finetuned-vqa", device="cpu"):
        self.device = device
        self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.processor = ViltProcessor.from_pretrained(model_name)
    
    def predict(self, image_path, question):
        """Predict the answer to a question given an image."""
        image = Image.open(image_path)
        # Prepare inputs
        encoding = self.processor(image, question, return_tensors="pt")
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits

        # Get the predicted answer and probability
        probs = torch.softmax(logits, dim=-1)
        max_prob, idx = torch.max(probs, dim=-1)

        logits = outputs.logits
        idx = logits.argmax(-1).item()

        # return self.model.config.id2label[idx], max_prob.item()

        return {
            "predicted_answer": self.model.config.id2label[idx],
            "probability": max_prob.item()
        }
