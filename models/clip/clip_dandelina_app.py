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

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "webm"])
# query = st.text_input("Enter a text query for frame retrieval:")
# print(query)


if uploaded_video is not None:
    # Save and extract frames
    video_path = os.path.join("uploaded_videos", uploaded_video.name)

    save_directory = "uploaded_videos"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.success(f"Video uploaded successfully: {video_path}")
    
    # Extract frames from the video
    st.text("Extracting frames...")
    video_processor = FrameExtractor(video_path, fps=1)
    start_time = time.time()
    frames = video_processor.extract_frames(start_frame=0, length=3600, video_piece_id=0)
    frame_extraction_time = time.time() - start_time
    st.write(f"Time taken to extract {len(frames)} frames: {frame_extraction_time}")
    
    # # Display some frames
    # st.text("Displaying a few frames...")
    # for i, frame in enumerate(frames[:5]):
    #     st.image(frame, caption=f"Frame {i+1}")

    # Compute embeddings


    if detection_type == "Multiple persons":
        detection_fn = clip_model.detect_multiple_persons
    elif detection_type == "No person present":
        detection_fn = clip_model.detect_no_person
    # elif detection_type == "Person using phone":
    #     detection_fn = clip_model.detect_person_using_phone

    num_columns = 10  # Number of columns in a row
    columns = st.columns(num_columns)
    st.write(f"Running {detection_type} detection on each frame...")
    results = []
    start_time = time.time()
    for i, frame_path in enumerate(frames):
        label, prob = detection_fn(frame_path)
        
        # Store and display the result
        results.append((i, label, prob))
        # st.image(frame_path, caption=f"Frame {i+1} - {label} Probability: {prob:.4f}", width=100)
        # st.image(frame_path, width=100)
        columns[i % num_columns].image(frame_path, caption=f"{label}|Prob:{prob:.2f}", use_column_width=True)
        # st.write(f"{label} Probability: {prob:.4f}")

    time_taken_by_clip = time.time() - start_time
    st.write(f"Time taken by clip: {time_taken_by_clip}")
