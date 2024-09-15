import streamlit as st
from modeling_dandelin import VQADandelin
from frame_extraction2 import FrameExtractor
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import time
# Initialize components
vqa_model = VQADandelin()

# Streamlit UI
st.title("Visual Question Answering with Vilt")

# Upload video file
# uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "webm"])
# query = st.text_input("Enter a text query for frame retrieval:")
# print(query)

# frames_dir = "/home/ajeet/codework/dataset_frames/2625050"
# frames_dir = "/home/ajeet/codework/dataset_frames/2591822"
frames_dir =  "/home/ajeet/codework/dataset_frames/2602597"
# frames_dir = "/home/ajeet/codework/dataset_frames/2529909"

frames = [os.path.join(frames_dir, image) for image in os.listdir(frames_dir)]

# frames = frames[:100]

no_person_frames = []
single_person_frames = []
multiple_person_frames = []
uncertain_frames = []

query = "How many persons are there?"
st.text("Running VQA on each frame...")
start_time = time.time()
yes_count = 0
for i, frame_path in enumerate(frames):
    image = Image.open(frame_path)
    # st.image(frame_path, caption=f"Frame {i+1}")
    # st.image(image, caption=f"Frame {i+1}", width=100)
    
    # Perform VQA on the frame
    label, prob = vqa_model.classify(frame_path, query)

    if label == "no_person":
        no_person_frames.append((frame_path, prob))
    elif label == "single_person":
        single_person_frames.append((frame_path, prob))
    elif label == "multiple_persons":
        multiple_person_frames.append((frame_path, prob))
    elif label == "Uncertain":
        uncertain_frames.append((frame_path, prob))


    yes_count = yes_count + 1
    if yes_count % 100 == 0:
        st.write(f"yes_count: {yes_count}::::{time.time() - start_time}")

time_taken_by_embeddings = time.time() - start_time
st.write(f"Time taken to vqa: {time_taken_by_embeddings:.2f} seconds")





def create_big_image(frames, grid_size=(10, 10), thumbnail_size=(100, 100), padding=10):
    """Create a big image by arranging smaller images in a grid with padding."""
    big_image_width = grid_size[0] * thumbnail_size[0] + (grid_size[0] - 1) * padding
    big_image_height = grid_size[1] * thumbnail_size[1] + (grid_size[1] - 1) * padding
    big_image = Image.new("RGB", (big_image_width, big_image_height))

    for i, (frame_path, prob) in enumerate(frames):
        if i >= grid_size[0] * grid_size[1]:  # Limit to the grid size
            break
        small_image = Image.open(frame_path).resize(thumbnail_size)

        frame_id = os.path.splitext(os.path.basename(frame_path))[0]
        draw = ImageDraw.Draw(small_image)
        draw.text((5, 5), f"{prob}", fill="white")  # Adjust position and color as needed

        # Calculate the position with padding
        x = (i % grid_size[0]) * (thumbnail_size[0] + padding)
        y = (i // grid_size[0]) * (thumbnail_size[1] + padding)
        big_image.paste(small_image, (x, y))

    return big_image

def display_frames(label, frames):
    st.write(f"### {label.capitalize()} Frames (Count: {len(frames)})")

    # Calculate how many big images we can create
    num_big_images = (len(frames) + 100 - 1) // 100  # 100 small images per big image

    for page in range(num_big_images):
        start_idx = page * 100
        end_idx = min(start_idx + 100, len(frames))

        # Create and display the big image for the current batch
        big_image = create_big_image(frames[start_idx:end_idx], grid_size=(10, 10), thumbnail_size=(100, 100))
        st.image(big_image, caption=f"Big Image (Batch {page + 1}/{num_big_images})", use_column_width=True)


if no_person_frames:
    no_person_frames = sorted(no_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    display_frames("no person", no_person_frames)
if single_person_frames:
    single_person_frames = sorted(single_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    display_frames("single person", single_person_frames)
if multiple_person_frames:
    multiple_person_frames = sorted(multiple_person_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    display_frames("multiple persons", multiple_person_frames)
if uncertain_frames:
    uncertain_frames = sorted(uncertain_frames, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].split('_')[1]))
    display_frames("uncertain_frames", uncertain_frames)










# if uploaded_video is not None and query:
    # Save and extract frames
    # video_path = os.path.join("uploaded_videos", uploaded_video.name)

    # save_directory = "uploaded_videos"
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)
    # with open(video_path, "wb") as f:
    #     f.write(uploaded_video.read())
    
    # st.success(f"Video uploaded successfully: {video_path}")
    
    # # Extract frames from the video
    # st.text("Extracting frames...")
    # video_processor = FrameExtractor(video_path, fps=1)
    # frames = video_processor.extract_frames(start_frame=0, length=3600, video_piece_id=0)
    
    # # Display some frames
    # st.text("Displaying a few frames...")
    # for i, frame in enumerate(frames[:5]):
    #     st.image(frame, caption=f"Frame {i+1}")

    # Compute embeddings
    # st.text("Running VQA on each frame...")
    # start_time = time.time()
    # yes_count = 0
    # for i, frame_path in enumerate(frames):
    #     image = Image.open(frame_path)
    #     # st.image(frame_path, caption=f"Frame {i+1}")
    #     st.image(image, caption=f"Frame {i+1}", width=100)
        
    #     # Perform VQA on the frame
    #     result = vqa_model.predict(frame_path, query)
        
    #     # Display the prediction and the probability for each frame
    #     # st.write(f"**Frame {i+1} - Predicted Answer:** {result['predicted_answer']}")
    #     # st.write(f"**Probability:** {result['probability']:.4f}")

    #     st.write(f"**Answer:** {result['predicted_answer']} | **Probability:** {result['probability']:.4f}")
    #     if result['predicted_answer'].lower() == "yes":
    #         yes_count += 1

    # time_taken_by_embeddings = time.time() - start_time
    # st.write(f"Time taken to vqa: {time_taken_by_embeddings:.2f} seconds")

    

