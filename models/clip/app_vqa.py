import streamlit as st
# from modeling_clip import CLIP, TextImageRetriever
from modeling_vqa import VQAViolationDetection
from frame_extraction2 import FrameExtractor
import os
import torch
import cv2

# Initialize components
vqa_violation_detection = VQAViolationDetection()
# storage = ImageFrameStorage()
# retriever = TextImageRetriever(clip_model)

# Streamlit UI
st.title("Video Frame Retrieval")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "webm"])
# query = st.text_input("Enter a text query for frame retrieval:")
# print(query)
query_options = {
    "No person": "Is there no person in the image?",
    "Multiple persons": "Are there multiple persons in the image?",
    "Person using phone": "Is there a person using a phone?"
}

selected_query_option = st.selectbox("Select a query for violation detection:", options=list(query_options.keys()))



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
    frames = video_processor.extract_frames(start_frame=0, length=3600, video_piece_id=0)
    
    
    if selected_query_option:
        # Process each frame and pass the query to the violation detection model
        query = query_options[selected_query_option]
        st.text(f"Selected Query: {query}")
        st.text("Processing frames and detecting violations...")
        for i, frame in enumerate(frames):
            # Convert the frame to a format that can be passed to the Qwen2-VL model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            violation_detected = vqa_violation_detection.predict(rgb_frame, query)
            st.text(f"Anwer: {violation_detected}")
            if violation_detected:
                st.image(frame, caption=f"Violation detected in frame {i+1}")

