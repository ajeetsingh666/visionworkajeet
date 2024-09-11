import streamlit as st
from modeling_dandelin import VQADandelin
from frame_extraction2 import FrameExtractor
import os
import torch
from PIL import Image

# Initialize components
vqa_model = VQADandelin()

# Streamlit UI
st.title("Visual Question Answering with Vilt")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "webm"])
query = st.text_input("Enter a text query for frame retrieval:")
print(query)

if uploaded_video is not None and query:
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
    
    # # Display some frames
    # st.text("Displaying a few frames...")
    # for i, frame in enumerate(frames[:5]):
    #     st.image(frame, caption=f"Frame {i+1}")

    # Compute embeddings
    st.text("Running VQA on each frame...")
    yes_count = 0
    for i, frame_path in enumerate(frames):
        image = Image.open(frame_path)
        # st.image(frame_path, caption=f"Frame {i+1}")
        st.image(image, caption=f"Frame {i+1}", width=100)
        
        # Perform VQA on the frame
        result = vqa_model.predict(frame_path, query)
        
        # Display the prediction and the probability for each frame
        # st.write(f"**Frame {i+1} - Predicted Answer:** {result['predicted_answer']}")
        # st.write(f"**Probability:** {result['probability']:.4f}")

        st.write(f"**Answer:** {result['predicted_answer']} | **Probability:** {result['probability']:.4f}")
        if result['predicted_answer'].lower() == "yes":
            yes_count += 1

    

