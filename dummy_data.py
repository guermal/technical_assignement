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
save_root = Path("synthetic_dataset")

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

def generate_video(class_id, is_train=True):
    shape, color, motion = class_patterns[class_id]
    videos = []

    for _ in range(train_videos_per_class if is_train else test_videos_per_class):
        clip = []

        # Start pos
        x, y = 10, 10
        dx, dy = motion

        for t in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            draw_shape(frame, shape, color, (int(x), int(y)))
            x += dx
            y += dy
            x = np.clip(x, 5, width - 5)
            y = np.clip(y, 5, height - 5)
            clip.append(torch.from_numpy(frame).permute(2, 0, 1))  # [C, H, W]

        video_tensor = torch.stack(clip)  # [T, C, H, W]
        videos.append(video_tensor)

    return videos

# Define class patterns (shape, color (BGR), motion (dx, dy))
class_patterns = {
    0: ("circle", (0, 0, 255), (3, 0)),     # red circle, left→right
    1: ("circle", (255, 0, 0), (-3, 0)),    # blue circle, right→left
    2: ("square", (0, 255, 0), (0, 3)),     # Green square, up→down
    3: ("square", (0, 255, 255), (0, -3)),  # Yellow square, down→up
    4: ("triangle", (255, 255, 255), (2, 2)),   # White triangle, diag ↘
    5: ("triangle", (255, 0, 255), (-2, -2)),  # Magenta triangle, diag ↖
    6: ("circle", (255, 255, 0), (0, 0)),   # Yellow circle, static
    7: ("square", (0, 0, 255), (np.random.randint(-2, 3), np.random.randint(-2, 3))),  # blue square, random
    8: ("triangle", (0, 255, 0), (1, 1)),   # Green triangle, slow motion
    9: ("circle", (0, 255, 255), (6, 0)),   # Fast motion yellow circle
}

# Save all videos (as .pt files)
def save_dataset():
    for mode in ["train", "test"]:
        for class_id in tqdm(range(num_classes)):
            videos = generate_video(class_id, is_train=(mode == "train"))
            class_dir = save_root / mode / f"class_{class_id}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for i, video in enumerate(videos):
                torch.save(video, class_dir / f"video_{i}.pt")

save_dataset()
