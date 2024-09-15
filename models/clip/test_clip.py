from modeling_clip import CLIP
import os
import time
from PIL import Image, ImageDraw, ImageFont

# Initialize components
clip_model = CLIP(device="cpu")

frames = [
    "/home/ajeet/codework/dataset_frames/2572904/0_4.jpg",
    "/home/ajeet/codework/dataset_frames/2572904/0_15.jpg",

    "/home/ajeet/codework/dataset_frames/2572904/0_24.jpg",
    "/home/ajeet/codework/dataset_frames/2572904/0_26.jpg",

    "/home/ajeet/codework/dataset_frames/2572904/0_41.jpg",
    "/home/ajeet/codework/dataset_frames/2572904/0_12.jpg",


]
# text_prompts = {
# "multiple persons": [
#     "a photo of 2 persons",
#     # "a photo of 2 people",
#     # "a photo of 2 or more persons",
#     # "a photo of 2 or more peoples",

#     # "a photo of 2 people's faces",
#     # "a photo of 2 persons faces",
#     # "a photo of 2 or more people's faces",
#     # "a photo of 2 or more persons faces",
# ],
# "single person": [
# "a photo of 1 person",
# "a photo of 1 people",

# "a photo of 1 person's face",
#         "A photo of a single person",
#         # "A photo of a single person facing the camera.",
#         # "An image showing one individual looking directly at the camera.",
#         "A picture of one person",
#         "A photo with just one person",
#         "A photo of a person working",
#         "A photo showing single face"
        

#         "An image showing single person",
#         "An image featuring one individual",
#         "An image of a lone individual",
#         "A picture of one person",
#         "An image showing an individual alone",
#         # "An image showing singe face",

#         "A photo of the person working",
#         "A photo containing a single person",

#         "There is a person in a room"

# ],
# "no person": [
#     "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",
#         "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",
#         "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",
#         "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",
#         "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",
#         "a photo of no people",
#     "a photo of a place without any people.",
#     "a photo of a place without any person.",

#     "a photo of 0 people's faces.",
#     "a photo of 0 persons' faces.",

# ]}

text_prompts = {
"multiple persons": [
    "a photo of multiple people",
],
"single person": [
    "a photo of a single person",

],
"no person": [
    "A photo with no person.",

]}

images, image_embeddings = clip_model.get_image_embeddings(frames)


results = []
no_person_frames = []
single_person_frames = []
multiple_person_frames = []

precomputed_embeddings = clip_model.precompute_prompt_embeddings(text_prompts)
for i, (frame_path, image_embedding) in enumerate(zip(frames, image_embeddings)):
    # Define text prompts for each classification

    # label, prob = clip_model.classify_image(image_embedding, text_prompts)
    label, prob = clip_model.classify_image(image_embedding, precomputed_embeddings)
    
    # Group frames by classification
    if label == "no person":
        no_person_frames.append((frame_path, prob))
    elif label == "single person":
        single_person_frames.append((frame_path, prob))
    elif label == "multiple persons":
        multiple_person_frames.append((frame_path, prob))

print("check")