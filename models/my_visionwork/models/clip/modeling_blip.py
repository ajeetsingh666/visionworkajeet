import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import requests

class VQABlip:
    def __init__(self, model_name="Salesforce/blip-vqa-base", device="cpu"):
        self.device = "cpu"
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

        # Forward pass
        with torch.inference_mode():
            inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)

        # Decode the generated answer
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Return the predicted answer (logits are not directly available in the BLIP model)
        return {"predicted_answer": answer.strip(), "probability": 1}

    def classify(self, image_path_or_url, query):
        """Classify the image based on the number of people present."""
        result = self.predict(image_path_or_url, query)
        answer = result["predicted_answer"].lower()

        prob = 1

        number_words = {
            "none": 0, 
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }
        # Try to convert the answer to a number
        try:
            if answer in number_words:
                print("inside_word")
                answer_number = number_words[answer]
            else:
                # Attempt to parse the answer as an integer
                answer_number = int(answer)
        except ValueError:
            # If answer isn't a number, return uncertain classification
            print("inside_exception_blip")
            return "Uncertain", prob

        # Classify based on the parsed number
        if answer_number == 0:
            classification = "no_person"
        elif answer_number == 1:
            classification = "single_person"
        elif answer_number >= 2:
            classification = "multiple_persons"
        else:
            classification = "Uncertain"

        return classification, prob  # Probability is not provided by BLIP

# Example usage:
if __name__ == "__main__":
    vqa_blip = VQABlip(device="cpu")
    question = "how many people are in the picture?"
    result = vqa_blip.classify("/tmp/video_incidents_ajeet/2538482/0_375.jpg", question)
    print(result)
