import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

class VQABlip:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = device
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)
    
    def predict(self, image_path_or_url, question):
        """Predict the answer to a question given an image."""
        # Load image from file path or URL
        # if image_path_or_url.startswith('http'):
        #     raw_image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert('RGB')
        # else:
        #     raw_image = Image.open(image_path_or_url).convert('RGB')
        
        raw_image = Image.open(image_path_or_url).convert('RGB')

        # Prepare inputs
        inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode the generated answer
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Return the predicted answer (logits are not directly available in the BLIP model)
        return {"predicted_answer": answer.strip(), "probability": None}

    def classify(self, image_path_or_url, query):
        """Classify the image based on the number of people present."""
        result = self.predict(image_path_or_url, query)
        answer = result["predicted_answer"].lower()

        # Try to convert the answer to a number
        try:
            # Attempt to parse the answer as an integer
            answer_number = int(answer)
        except ValueError:
            # If answer isn't a number, return uncertain classification
            return "Uncertain", None

        # Classify based on the parsed number
        if answer_number == 0:
            classification = "no_person"
        elif answer_number == 1:
            classification = "single_person"
        elif answer_number >= 2:
            classification = "multiple_persons"
        else:
            classification = "Uncertain"

        return classification, None  # Probability is not provided by BLIP

# Example usage:
# vqa_blip = VQABlip(device="cpu")
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# question = "How many people are there?"

# result = vqa_blip.classify("/home/ajeet/codework/dataset_frames/2648356/0_460.jpg", question)
# print(result)
