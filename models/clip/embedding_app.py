import streamlit as st
from modeling_clip import CLIP
from frame_extraction2 import FrameExtractor
import os
import time
from PIL import Image

# Initialize components
clip_model = CLIP(device="cpu")

# Streamlit UI
st.title("Video Frame Retrieval using CLIP")

# Upload video file
detection_type = st.selectbox(
    "Select detection type:",
    ("Multiple persons", 
     "Single person",
     "No person present"),
    # index=None,
    placeholder="Select violation to be detected...",
)
st.write("You selected:", detection_type)

if detection_type != "None":

    # Define the text prompts for each detection type
    # text_prompts = {
    #     "multiple persons": ["A photo of multiple persons"],
    #     "single person": ["A photo of a single person"],
    #     "no person": ["A photo with no person"]
    # }


    text_prompts = {
    "multiple persons": [
        "A photo of multiple persons",
        "A photo with several people present",
        "A picture featuring a group of people",
        "A picture with more than one person visible",
        "A photo showing numerous people",
        "A group of people in a picture",
        "A photo with multiple faces.",
        "Two or more persons taking a selfie",

        "An image with more than one person.",
        "An image showing multiple persons",
        "An image of multiple persons",
        "An image of several people present",
        "An image of several persons interacting",
        "An image of multiple persons interacting",
        "An image showing multiple faces.",

        "A group of people",
        "A group of people working",

        # "A photo of multiple persons, with some in the background.",
        # "An image showing people in the foreground and background.",
        # "A picture of a group with some persons located in the background.",
        # "A photo with people interacting in the foreground and others in the background.",
        # "A scene capturing both foreground and background persons.",
        # "A photo featuring several individuals, with some appearing in the background.",

        # "A photo of multiple persons facing the camera.",
        # "A picture showing multiple persons with their faces towards the camera.",
        # "An image showing several people looking directly at the camera."
    ],
    "single person": [
        "A photo of a single person",
        # "A photo of a single person facing the camera.",
        # "An image showing one individual looking directly at the camera.",
        "A picture of one person",
        "A photo with just one person",
        "A photo of a person working",
        "A photo showing single face"
        

        "An image showing single person",
        "An image featuring one individual",
        "An image of a lone individual",
        "A picture of one person",
        "An image showing an individual alone",
        # "An image showing singe face",

        "A photo of the person working",
        "A photo containing a single person"
        
    ],
    "no person": [
        "A photo with no person",
        "A photo that has no one in it",

        "An image that does not contain any people",
        "An image devoid of people",

        "A picture showing an empty space",

        "A picture where no humans are visible",
        "A photo of an empty room with no person visible",
        "An image showing an empty room with no person visible",

        ]
    }


    # Load frames from a directory (replace with your actual frame extraction)
    # frames_dir = "../../../web_dataset/single_person"
    # frames = [os.path.join(frames_dir, image) for image in os.listdir(frames_dir)]

    # frames_dir = "/home/ajeet/codework/web_dataset/multiple_persons"
    # frames = [os.path.join(frames_dir, image) for image in os.listdir(frames_dir)]

    frames_dir = "/home/ajeet/codework/dataset_frames/2602597"
    frames = [os.path.join(frames_dir, image) for image in os.listdir(frames_dir)]
    # print(frames.nbytes)

    # Display layout configuration
    num_columns = 6  # Number of columns in a row for displaying images
    columns = st.columns(num_columns)
    st.write(f"Running {detection_type} detection on each frame...")

    # Lists to hold frames based on classification labels
    results = []
    no_person_frames = []
    single_person_frames = []
    multiple_person_frames = []

    start_time = time.time()

    # Get embeddings for all frames
    start_time = time.time()
    images, image_embeddings = clip_model.get_image_embeddings(frames)
    print(image_embeddings.nbytes)
    print("----")
    time_taken_by_embeddings = time.time() - start_time
    st.write(f"Time taken to extract embeddings: {time_taken_by_embeddings:.2f} seconds")

    # Classify each frame
    start_time = time.time()
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

    time_taken_by_clip = time.time() - start_time
    st.write(f"Time taken by clip: {time_taken_by_clip:.2f} seconds")

    # Function to display frames with a specific label
    # def display_frames(label, frames):
    #     st.write(f"### {label.capitalize()} Frames (Count: {len(frames)})")
    #     num_columns = 8  # Adjust the number of columns for displaying images
    #     columns = st.columns(num_columns)
    #     for i, (frame_path, prob) in enumerate(frames):
    #         columns[i % num_columns].image(frame_path, caption=f"prob: {prob:.2f}", use_column_width=True)
    #     # st.write(f"### Total {label.capitalize()} Frames: {len(frames)}")


    def create_big_image(frames, grid_size=(10, 10), thumbnail_size=(100, 100)):
        """Create a big image by arranging smaller images in a grid."""
        big_image_width = grid_size[0] * thumbnail_size[0]
        big_image_height = grid_size[1] * thumbnail_size[1]
        big_image = Image.new("RGB", (big_image_width, big_image_height))

        for i, (frame_path, prob) in enumerate(frames):
            if i >= grid_size[0] * grid_size[1]:  # Limit to the grid size
                break
            small_image = Image.open(frame_path).resize(thumbnail_size)
            x = (i % grid_size[0]) * thumbnail_size[0]
            y = (i // grid_size[0]) * thumbnail_size[1]
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


    # Display the frames grouped by their labels
    if no_person_frames:
        display_frames("no person", no_person_frames)
    if single_person_frames:
        display_frames("single person", single_person_frames)
    if multiple_person_frames:
        display_frames("multiple persons", multiple_person_frames)
