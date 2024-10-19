import numpy as np
from collections import deque
from models.clip.phone.modeling_clip import CLIP


clip = CLIP()

# Function to calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Imposter Detection Function using Dynamic Threshold Adaptation Across Slots
def imposter_detection_across_slots(authorized_embedding, slots, window_size=3, decay_factor=0.9):
    # History deque to keep track of slot-level statistics for the moving window
    slot_averages = deque(maxlen=window_size)
    slot_std_devs = deque(maxlen=window_size)

    for idx, slot_frames in enumerate(slots):
        similarities = [calculate_cosine_similarity(authorized_embedding, frame_embedding) for frame_embedding in slot_frames]
        slot_average = np.mean(similarities)
        slot_std_dev = np.std(similarities)

        # Add slot statistics to the history
        slot_averages.append(slot_average)
        slot_std_devs.append(slot_std_dev)

        # Calculate the dynamic threshold based on the historical averages and std deviations
        if len(slot_averages) >= window_size:
            historical_average = np.mean(slot_averages)
            historical_std_dev = np.mean(slot_std_devs)
            dynamic_threshold = historical_average - (historical_std_dev * decay_factor)
        else:
            # Use the current slot's average and std deviation if the history isn't full yet
            dynamic_threshold = slot_average - (slot_std_dev * decay_factor)

        # Check if the entire slot indicates an imposter
        if all(similarity < dynamic_threshold for similarity in similarities):
            print(f"Imposter detected for slot {idx + 1}!")
        else:
            print(f"Authorized user detected for slot {idx + 1}.")

# # Example usage
# authorized_embedding = np.array([0.5, 0.2, 0.1, 0.9])  # Placeholder authorized user embedding
# slot1_frames = [np.array([0.4, 0.1, 0.1, 0.8]), np.array([0.3, 0.2, 0.1, 0.85]), np.array([0.45, 0.15, 0.05, 0.8])]
# slot2_frames = [np.array([0.1, 0.05, 0.02, 0.4]), np.array([0.15, 0.1, 0.05, 0.3]), np.array([0.12, 0.08, 0.04, 0.35])]
# slot3_frames = [np.array([0.5, 0.3, 0.2, 0.7]), np.array([0.48, 0.32, 0.25, 0.75]), np.array([0.46, 0.29, 0.21, 0.68])]

# # List of slots
# slots = [slot1_frames, slot2_frames, slot3_frames]

clip.get_image_embeddings()

# Detect imposter across slots
imposter_detection_across_slots(authorized_embedding, slots)
