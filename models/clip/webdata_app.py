import streamlit as st
from modeling_clip import CLIP, TextImageRetriever
from frame_extraction2 import FrameExtractor
import os
import torch
import time

# Initialize components
clip_model = CLIP(device="cuda" if torch.cuda.is_available() else "cpu")
# storage = ImageFrameStorage()
retriever = TextImageRetriever(clip_model)

# Streamlit UI
st.title("Video Frame Retrieval using CLIP")

# Upload video file
detection_type = st.selectbox(
    "Select detection type:",
    ("Multiple persons", 
     "No person present"),
    index=None,
    placeholder="Select violation to be detected...",
)
st.write("You selected:", detection_type)

if detection_type == "Multiple persons":
    detection_fn = clip_model.detect_multiple_persons
elif detection_type == "No person present":
    detection_fn = clip_model.detect_no_person

# frames = [image for image in os.listdir("../../../web_dataset/multiple_persons")]
# frames = [os.path.join("/home/ajeet/codework/web_dataset/multiple_persons/", frame) for frame in frames]

frames = [image for image in os.listdir("../../../web_dataset/single_person")]
frames = [os.path.join("/home/ajeet/codework/web_dataset/single_person/", frame) for frame in frames]

num_columns = 6  # Number of columns in a row
columns = st.columns(num_columns)
st.write(f"Running {detection_type} detection on each frame...")
results = []
no_person_frames = []
single_person_frames = []
multiple_person_frames = []

start_time = time.time()

for i, frame_path in enumerate(frames):
    label, prob = detection_fn(frame_path)
    
    if label == "no person":
        no_person_frames.append((frame_path, prob))
    elif label == "single person":
        single_person_frames.append((frame_path, prob))
    elif label == "multiple persons":
        multiple_person_frames.append((frame_path, prob))

time_taken_by_clip = time.time() - start_time
st.write(f"Time taken by clip: {time_taken_by_clip}")

def display_frames(label, frames):
    st.write(f"### {label.capitalize()} Frames (Count: {len(frames)})")
    num_columns = 8  # Adjust the number of columns
    columns = st.columns(num_columns)
    for i, (frame_path, prob) in enumerate(frames):
        columns[i % num_columns].image(frame_path, caption=f"{frame_path}: {prob:.2f}", use_column_width=True)
    # st.write(f"### {len(frames)} Frames:")

# Display the frames grouped by their labels
if no_person_frames:
    display_frames("no person", no_person_frames)
if single_person_frames:
    display_frames("single person", single_person_frames)
if multiple_person_frames:
    display_frames("multiple persons", multiple_person_frames)
