import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


class CLIP:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        # self.device = device
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_image_embeddings(self, frames, batch_size=500):
        """Get image embeddings for a list of frames."""
        embeddings_list = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            images = [Image.open(frame) for frame in batch_frames]
            # images = [Image.open(frame) for frame in frames]
            # inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt")
                input_memory = inputs['pixel_values'].element_size() * inputs['pixel_values'].nelement()
                print(f"Memory taken by input tensor: {input_memory / (1024 ** 2):.2f} MB")
                embeddings = self.model.get_image_features(**inputs)

            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings_list.append(embeddings)
        return images, torch.cat(embeddings_list)

    def get_text_embedding(self, query):
        """Get text embeddings for a query."""
        # inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)
        inputs = self.processor(text=query, return_tensors="pt")
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=1, keepdim=True)
        return text_embedding
    
    def compute_similarity(self, image_embedding, text_embedding):
        """
        Compute the cosine similarity between image and text embeddings using sklearn.
        """
        # image_embedding and text_embedding are NumPy arrays
        # return cosine_similarity(image_embedding.cpu().numpy().reshape(1, -1), text_embedding.cpu().numpy().reshape(1, -1))[0][0]

        image_embedding = image_embedding.unsqueeze(0)
        # text_embedding = text_embedding.unsqueeze(0)
        cosine_similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding)
        return cosine_similarity

    # def classify_image(self, image_embedding, text_prompts):
    #     """
    #     Classify the image by computing its similarity to different text prompts and return the label with max similarity.
    #     """
    #     max_sim = -float('inf')
    #     best_label = None
        
    #     for label, prompt in text_prompts.items():
    #         # Compute average embedding for all prompts related to this label
    #         # prompt_embeddings = np.mean([self.get_text_embedding(p) for p in prompt], axis=0)
    #         text_embeddings = torch.stack([self.get_text_embedding(p) for p in prompt])
    #         prompt_embeddings = torch.mean(text_embeddings, dim=0)  # PyTorch mean

    #         prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(dim=-1, keepdim=True)
            
    #         # Compute cosine similarity between the image and the prompt embeddings
    #         sim = self.compute_similarity(image_embedding, prompt_embeddings)
            
    #         if sim > max_sim:
    #             max_sim = sim
    #             best_label = label
        
    #     return best_label, max_sim
    
    
    def precompute_prompt_embeddings(self, text_prompts):
        """
        Precompute and return embeddings for each set of text prompts.
        """
        precomputed_prompt_embeddings = {}
        
        for label, prompt in text_prompts.items():
            text_embeddings = torch.stack([self.get_text_embedding(p) for p in prompt])
            prompt_embeddings = torch.mean(text_embeddings, dim=0)  # PyTorch mean

            # Normalize the embedding
            prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(dim=1, keepdim=True)
            
            precomputed_prompt_embeddings[label] = prompt_embeddings
        
        return precomputed_prompt_embeddings


    def classify_image(self, image_embedding, precomputed_prompt_embeddings):
        """
        Classify the image by computing its similarity to precomputed text prompt embeddings and return the label with max similarity.
        """
        max_sim = -float('inf')
        best_label = None
        
        # Iterate over the precomputed prompt embeddings
        for label, prompt_embedding in precomputed_prompt_embeddings.items():
            # Compute cosine similarity between the image and the prompt embeddings
            sim = self.compute_similarity(image_embedding, prompt_embedding)
            # print(sim)
            
            if sim > max_sim:
                max_sim = sim
                best_label = label
        # print("-------")
        return best_label, max_sim
    
    # def precompute_prompt_embeddings(self, text_prompts):
    #     """
    #     Precompute and return embeddings for each set of text prompts.
    #     """
    #     precomputed_prompt_embeddings = {}
        
    #     for label, prompt in text_prompts.items():
    #         # Compute embeddings for each prompt
    #         text_embeddings = torch.stack([self.get_text_embedding(p) for p in prompt])
            
    #         # Normalize each embedding
    #         # text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
    #         # Store normalized embeddings
    #         precomputed_prompt_embeddings[label] = text_embeddings
        
    #     return precomputed_prompt_embeddings

    # def classify_image(self, image_embedding, precomputed_prompt_embeddings):
    #     """
    #     Classify the image by computing the average cosine similarity to precomputed text prompt embeddings 
    #     and return the label with max average similarity.
    #     """
        
    #     average_similarities = {}
        
    #     # Iterate over the precomputed prompt embeddings
    #     for label, prompt_embeddings in precomputed_prompt_embeddings.items():
    #         # Calculate cosine similarities for each prompt embedding
    #         similarities = []
    #         for prompt_embedding in prompt_embeddings:
    #             sim = self.compute_similarity(image_embedding, prompt_embedding)
    #             similarities.append(sim.item())  # Convert tensor to float
            
    #         # Compute the average cosine similarity for the current label
    #         average_similarity = sum(similarities) / len(similarities)
    #         average_similarities[label] = average_similarity
        
    #     # Get the label with the maximum average similarity
    #     best_label = max(average_similarities, key=average_similarities.get)
    #     max_average_sim = average_similarities[best_label]
        
    #     return best_label, max_average_sim



    def _run_vqa(self, image_path, text_inputs):
        """Common method to process inputs and compute probabilities."""
        image = Image.open(image_path)

        # Prepare inputs for CLIP
        inputs = self.processor(text=text_inputs, images=image, return_tensors="pt", padding=True).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get the logits and probabilities
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

        return probs

    # def detect_multiple_persons(self, image_path):
    #     """Classify if the image contains a single person or multiple persons."""
    #     # text_inputs = ["A photo of multiple persons", "A photo of a single person"]
    #     text_inputs = ["A photo of multiple persons", "A photo of a single person", "A photo with no person"]
    #     probs = self._run_vqa(image_path, text_inputs)

    #     # Interpret the result
    #     # label = "multiple persons" if probs[0][0] > probs[0][1] else "single person"
    #     # probability = probs.max().item()

    #     if probs[0][0] > probs[0][1] and probs[0][0] > probs[0][2]:
    #         label = "multiple persons"
    #     elif probs[0][1] > probs[0][0] and probs[0][1] > probs[0][2]:
    #         label = "single person"
    #     elif probs[0][2] > probs[0][0] and probs[0][2] > probs[0][1]:
    #         label = "no person"
    #     probability = probs.max().item()

    #     return label, probability

    def detect_multiple_persons(self, image_path):
        """Classify if the image contains multiple persons, a single person, or no person."""
        text_inputs = ["A photo of multiple persons present", "A photo of a single person present", "A photo with no person present"]
        probs = self._run_vqa(image_path, text_inputs)
        
        # Get the index of the highest probability
        max_index = probs[0].argmax().item()
        
        # Map the index to the corresponding label
        labels = ["multiple persons", "single person", "no person"]
        label = labels[max_index]
        
        # Get the maximum probability
        probability = probs[0][max_index].item()
        
        return label, probability

    def detect_no_person(self, image_path):
        """Classify if there is no person in the image."""
        text_inputs = ["A photo with no person", "A photo with a person"]
        probs = self._run_vqa(image_path, text_inputs)

        # Interpret the result
        label = "no person" if probs[0][0] > probs[0][1] else "person present"
        probability = probs.max().item()

        return label, probability

    def detect_person_using_phone(self, image_path):
        """Classify if a person in the image is using a phone."""
        text_inputs = ["A phot of a person using a phone", "A photo of a person not using a phone"]
        probs = self._run_vqa(image_path, text_inputs)

        # Interpret the result
        label = "person using phone" if probs[0][0] > probs[0][1] else "not using phone"
        probability = probs.max().item()

        return label, probability



class TextImageRetriever:
    def __init__(self, clip_model_wrapper):
        self.clip_model_wrapper = clip_model_wrapper

    def retrieve_similar_frames(self, query, image_embeddings):
        """Find the frames most similar to the query text."""
        text_embedding = self.clip_model_wrapper.get_text_embedding(query)
        similarities = cosine_similarity(
            text_embedding.cpu().numpy(), image_embeddings.cpu().numpy()
        )
        return similarities[0]

    def get_top_matching_frames(self, frames, similarities, top_k=5):
        """Retrieve the top K matching frames based on similarity."""
        sorted_indices = np.argsort(similarities)[::-1]
        top_frames = [frames[idx] for idx in sorted_indices[:top_k]]
        top_similarities = [similarities[idx] for idx in sorted_indices[:top_k]]
        return top_frames, top_similarities





