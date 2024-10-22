import os
import torch
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image


def load_data(data_dir):
    images = []
    labels = []

    for label, folder in enumerate(['cell_phones', 'no_cell_phones']):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(folder_path, filename))
                labels.append(label)

    return images, labels

images, labels = load_data('/home/ajeet/codework/datasets/train_clip/testing/')
dataset = {'image': images, 'label': labels}


from torch.utils.data import Dataset

class ImageLabelDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        # encoding = self.processor(images=image, return_tensors="pt")
        inputs = self.processor(text=["no cell phone", "a cell phone"], images=image, return_tensors="pt", padding=True)
        return inputs, label

custom_dataset = ImageLabelDataset(images, labels)
print(custom_dataset)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
model.train()  # Set model to training mode


from torch.utils.data import DataLoader

dataloader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 5

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()

        # Initialize the total loss for the batch
        total_loss = 0.0

        # Process each image in the batch
        for i in range(inputs['pixel_values'].shape[0]):  # Assuming 'pixel_values' holds your images
            # single_image_input = {k: v[i:i+1] for k, v in inputs.items()}  # Take a single image from the batch
            single_image_input = {k: v[i:i+1] if v.ndim == 4 else v[i] for k, v in inputs.items()}  # Ensure the correct shape
            
            label = labels[i:i+1]  # Get the corresponding label
            
            # Forward pass
            outputs = model(**single_image_input)
            logits_per_image = outputs.logits_per_image  # Similarity scores
            
            # Calculate the loss for the single image
            loss = torch.nn.functional.cross_entropy(logits_per_image, label)

            # Accumulate the loss
            total_loss += loss

        # Average the loss over the batch
        total_loss /= inputs['pixel_values'].shape[0]

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")



# for epoch in range(num_epochs):
#     for batch in dataloader:
#         inputs, labels = batch
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image  # Similarity scores
#         loss = torch.nn.functional.cross_entropy(logits_per_image, labels)

#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

