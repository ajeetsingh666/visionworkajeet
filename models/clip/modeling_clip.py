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
    
    def get_image_embeddings(self, frames):
        """Get image embeddings for a list of frames."""
        images = [Image.open(frame) for frame in frames]
        # inputs = self.processor(images=pil_images, return_tensors="pt", padding=True).to(self.device)
        inputs = self.processor(images=images, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return images, embeddings

    def get_text_embedding(self, query):
        """Get text embeddings for a query."""
        # inputs = self.processor(text=query, return_tensors="pt", padding=True).to(self.device)
        inputs = self.processor(text=query, return_tensors="pt")
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding



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





