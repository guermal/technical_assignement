import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Parameters
num_classes = 10
train_videos_per_class = 100
test_videos_per_class = 5
frames = 16
height, width = 64, 64
save_root = Path("synthetic_videos")  # NEW FOLDER for .mp4 videos

# Draw shape
def draw_shape(frame, shape, color, position):
    if shape == "circle":
        cv2.circle(frame, position, 6, color, -1)
    elif shape == "square":
        cv2.rectangle(frame, (position[0]-5, position[1]-5), (position[0]+5, position[1]+5), color, -1)
    elif shape == "triangle":
        pts = np.array([[position[0], position[1]-6],
                        [position[0]-5, position[1]+5],
                        [position[0]+5, position[1]+5]], np.int32)
        cv2.fillPoly(frame, [pts], color)

# Define class patterns (shape, color (BGR), motion (dx, dy))
class_patterns = {
    0: ("circle", (0, 0, 255), (3, 0)),
    1: ("circle", (255, 0, 0), (-3, 0)),
    2: ("square", (0, 255, 0), (0, 3)),
    3: ("square", (0, 255, 255), (0, -3)),
    4: ("triangle", (255, 255, 255), (2, 2)),
    5: ("triangle", (255, 0, 255), (-2, -2)),
    6: ("circle", (255, 255, 0), (0, 0)),
    7: ("square", (0, 0, 255), (np.random.randint(-2, 3), np.random.randint(-2, 3))),
    8: ("triangle", (0, 255, 0), (1, 1)),
    9: ("circle", (0, 255, 255), (6, 0)),
}

# Save videos as .mp4
def generate_and_save_videos(is_train=True):
    mode = "train" if is_train else "test"
    for class_id in tqdm(range(num_classes), desc=f"Saving {mode} videos"):
        shape, color, motion = class_patterns[class_id]
        class_dir = save_root / mode / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        num_videos = train_videos_per_class if is_train else test_videos_per_class

        for vid_idx in range(num_videos):
            x, y = 10, 10
            dx, dy = motion
            video_path = class_dir / f"video_{vid_idx}.mp4"

            # Define video writer (MP4V codec)
            out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

            for t in range(frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                draw_shape(frame, shape, color, (int(x), int(y)))
                x = np.clip(x + dx, 5, width - 5)
                y = np.clip(y + dy, 5, height - 5)
                out.write(frame)

            out.release()

# Call both train and test
def save_dataset_as_mp4():
    generate_and_save_videos(is_train=True)
    generate_and_save_videos(is_train=False)

save_dataset_as_mp4()
