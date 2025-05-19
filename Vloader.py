import json
import os.path as osp
#from bisect import bisect_right
import os
import random
import torch
import torch.utils.data as data
import numpy as np
#import pandas as pd
#import h5py
#from .datasets import DATA_LAYERS as registry
#from rekognition_online_action_detection.utils.ek_utils import (action_to_noun_map, action_to_verb_map)
import pickle
import math
import cv2
#from torchvision import datasets, transforms
from PIL import Image
#from transforms.spatial_transforms import Compose, Normalize,  MultiScaleCornerCrop, RandomHorizontalFlip, Scale ,MultiScaleRandomCrop, MultiScaleRandomCropMultigrid, ToTensor, CenterCrop, CenterCropScaled
#from transforms.temporal_transforms import TemporalRandomCrop
from torchvision.transforms.functional import to_pil_image

from transformers import AutoImageProcessor
import torchvision.transforms as transforms
        

import random



CLASS_DESCRIPTIONS = {
    "class_0": "a red circle moving from left to right",
    "class_1": "a blue circle moving from right to left",
    "class_2": "a green square moving downward",
    "class_3": "a yellow square moving upward",
    "class_4": "a white triangle moving diagonally down and right",
    "class_5": "a magenta triangle moving diagonally up and left",
    "class_6": "a yellow circle that is static",
    "class_7": "a red square moving in a random direction",
    "class_8": "a green triangle moving slowly",
    "class_9": "a cyan circle moving fast horizontally",
}






def generate_text_prompt(label,captioning=False):
    """
    Generate a textual prompt for a given label.
    This avoids label leakage by using indirect or noisy cues.
    """

    label_parts = CLASS_DESCRIPTIONS[label]  # e.g., "green_ball_moving_right"

    # Extract visual features
    colors = ["green", "red", "blue", "yellow"]
    shapes = ["square", "circle", "triangle"]
    motions = [
    "moving from left to right", "moving from right to left",
    "moving downward", "moving upward",
    "moving diagonally down and right", "moving diagonally up and left",
    "static", "moving in a random direction",
    "moving slowly", "moving fast horizontally"
]

    color = next((c for c in colors if c in label_parts), "object")
    shape = next((s for s in shapes if s in label_parts), "object")
    motion = next((m for m in motions if m in label_parts), None)

    prompt_type = random.choice(["partial", "neutral", "ambiguous", "noisy", "question"])
    if captioning:
        return CLASS_DESCRIPTIONS[label]
    if prompt_type == "partial":
        return f"A {color} {shape} appears in the scene."
    elif prompt_type == "neutral":
        return "An object is present in the video."
    elif prompt_type == "ambiguous":
        return f"The video might contain a {random.choice(colors)} {random.choice(shapes)}."
    elif prompt_type == "noisy":
        return f"A {random.choice(colors)} {random.choice(shapes)} is doing something."
    elif prompt_type == "question":
        return f"What is the {shape} doing?"
    else:
        return "This is a visual input."














class base_loader(data.Dataset):
    def __init__(self,gt_folder,seq_len,num_classes,captioning=False,spatial_transforms=None,temporal_transforms=None,mode='train'):
        
        #self.obsrv_len = obsrv_len
        self.gt_folder = gt_folder
        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms
        self.mode = mode
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.captioning = captioning
        #self. processor = AutoImageProcessor.from_pretrained(modelconf)
        self._make_dataset()

    
    def _make_dataset(self):
        
        
        self.dataset = []
        split_set = os.path.join(self.gt_folder,self.mode)
        for clss in os.listdir(split_set):
            label = clss.split('_')[1]
            txt = generate_text_prompt(clss,self.captioning)
            for clip in os.listdir(os.path.join(split_set,clss)):
                clip_folder = os.path.join(os.path.join(split_set,clss),clip)
                self.dataset.append([clip_folder,label,txt])

        
        
            
    def __getitem__(self,idx):
        clip_folder,label,txt = self.dataset[idx]
        video_input = torch.load(clip_folder)
        frame_indices = np.arange(0,len(video_input))
        if self.temporal_transforms is not None:
            frame_indices = self.temporal_transforms(frame_indices)

        video_input = video_input[frame_indices]
        
        if self.spatial_transforms is not None:
            self.spatial_transforms.randomize_parameters()
            video_input = [to_pil_image(f) for f in video_input]
            
            video_input = [self.spatial_transforms(f) for f in video_input]
            
            
        video_input = torch.stack(video_input)  # Shape: [T, C, H, W]
        #video_input = self.processor(video_input, return_tensors="pt")
        #video_input = {k: v.squeeze(0) for k, v in video_input.items()}  # remove batch dim
        


            
        # the bellow commented folder is if we want to control the number of frames
        #for sake of simplicity we use the full frames since our dummy dataset produces only 16 frame
        '''num_frames = len(video_input)
        if num_frames > self.seq_len:
            start_frame = np.random.randint(0,num_frames-self.seq_len)
        else:
            start_frame = 0
        indices = np.linspace(start_frame, start_frame+numFrame, self.seqLen, endpoint=False)
        
        video_input = video_input[indices]'''
        
        return video_input,int(label),txt

    
    def __len__(self):
        return len(self.dataset)




    
def load_video(path,resize=224,max_frames=64):
    assert os.path.isfile(path), f"{path} is not a valid file."

    cap = cv2.VideoCapture(path)
    frames = []
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(frame)
        tensor_frame = transform(image)  # Shape: [C, H, W]
        frames.append(tensor_frame)
        count += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
    return video_tensor

def load_frames(path,resize=224,max_frames=64):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    contents = sorted(os.listdir(path))
    count = 0
    frames = []
    for img in contents:
        frame = cv2.imread(os.path.join(path,img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        tensor_frame = transform(image)  # Shape: [C, H, W]
        frames.append(tensor_frame)
        count += 1
        if count >= max_frames:
            break
    return torch.stack(frames)


def read_video(path,resize=224):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in video_extensions:
            video_input = load_video(path,resize)

            return video_input
        else:
            return "unknown file"
    
    elif os.path.isdir(path):
        
        video_input = load_frames(path,resize)
        if len(video_input) <= 0:
            raise ValueError(f"No frames read from {path}")
        else:
            return video_input

    else:
        raise ValueError(f"invalid path {path}")
        

            
    
    





