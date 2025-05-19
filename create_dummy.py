import os
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image

import os
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image

'''def save_single_video_and_frames(tensor, output_dir, video_name="video", fps=8):
    """
    tensor: [T, C, H, W]
    output_dir: directory to save video and frames
    """
    os.makedirs(output_dir, exist_ok=True)
    T, C, H, W = tensor.shape

    video_path = os.path.join(output_dir, f"{video_name}.mp4")
    frame_dir = os.path.join(output_dir, video_name)
    os.makedirs(frame_dir, exist_ok=True)

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

    for t in range(T):
        frame = tensor[t]  # [C, H, W]
        frame = (frame * 255).clamp(0, 255).byte().cpu()
        img = to_pil_image(frame)
        img.save(os.path.join(frame_dir, f"frame_{t:03d}.jpg"))

        # Convert PIL to OpenCV format
        frame_cv2 = np.array(img)[:, :, ::-1]  # RGB to BGR
        writer.write(frame_cv2)

    writer.release()
    print(f"Saved video to {video_path} and frames to {frame_dir}")


video_tensor = torch.load("./synthetic_dataset/test/class_0/video_1.pt")  # Shape [B, T, C, H, W]
save_single_video_and_frames(video_tensor, output_dir="output_videos")'''


def save_pt_as_video(pt_path, out_path="video_output.mp4", fps=8):
    video_tensor = torch.load(pt_path)  # [T, C, H, W]
    frames = video_tensor.permute(0, 2, 3, 1).numpy()  # [T, H, W, C]
    
    # Convert to uint8 if needed
    if frames.dtype != 'uint8':
        frames = (frames * 255).clip(0, 255).astype('uint8')

    h, w = frames.shape[1:3]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame in frames:
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Saved video to {out_path}")

def save_pt_as_frames(pt_path, output_dir="frames_see/"):
    os.makedirs(output_dir, exist_ok=True)
    video_tensor = torch.load(pt_path)  # [T, C, H, W]
    frames = video_tensor.permute(0, 2, 3, 1).numpy()  # [T, H, W, C]
    
    if frames.dtype != 'uint8':
        frames = (frames * 255).clip(0, 255).astype('uint8')

    for i, frame in enumerate(frames):
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)

save_pt_as_frames(pt_path='./synthetic_dataset/test/class_3/video_2.pt')

#save_pt_as_video(pt_path='./synthetic_dataset/test/class_0/video_0.pt')