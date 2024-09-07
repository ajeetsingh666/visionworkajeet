import streamlit as st
from modeling_clip import CLIP, TextImageRetriever
from frame_extraction2 import FrameExtractor
import os
import torch

# Initialize components
clip_model = CLIP(device="cuda" if torch.cuda.is_available() else "cpu")
# storage = ImageFrameStorage()
retriever = TextImageRetriever(clip_model)

# Streamlit UI
st.title("Video Frame Retrieval using CLIP")

# Upload video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "webm"])
query = st.text_input("Enter a text query for frame retrieval:")
print(query)

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
    
    # # Display some frames
    # st.text("Displaying a few frames...")
    # for i, frame in enumerate(frames[:5]):
    #     st.image(frame, caption=f"Frame {i+1}")

    # Compute embeddings
    st.text("Computing embeddings...")
    images, image_embeddings = clip_model.get_image_embeddings(frames)

    print(image_embeddings.shape)
    st.text(image_embeddings.shape)
    # # Text query input
    
    if query:
        # Perform text-image retrieval
        st.text("Retrieving frames matching the query...")
        similarities = retriever.retrieve_similar_frames(query, image_embeddings)
        print(similarities)
        
        # Get and display the top frames
        top_frames, top_similarities = retriever.get_top_matching_frames(images, similarities, 10)
        for i, (frame, score) in enumerate(zip(top_frames, top_similarities)):
            st.image(frame, caption=f"Match {i+1} with similarity score {score:.4f}")
