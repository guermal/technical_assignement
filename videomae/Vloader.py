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
#from transformers import video_utils
#from transformers.video_utils import load_video
from huggingface_hub import hf_hub_download
import random
import av


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





def read_video_pyav(container):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    indices = np.linspace(0, 16, 16, endpoint=False)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])








class base_loader(data.Dataset):
    def __init__(self,gt_folder,num_classes,processor,device,mode='train'):
        
        #self.obsrv_len = obsrv_len
        self.gt_folder = gt_folder
        self.mode = mode
        self.num_classes = num_classes
        self.processor = processor
        #self. processor = AutoImageProcessor.from_pretrained(modelconf)
        self._make_dataset()
        self.device = device

    
    def _make_dataset(self):
        
        
        self.dataset = []
        split_set = os.path.join(self.gt_folder,self.mode)
        for clss in os.listdir(split_set):
            label = clss.split('_')[1]
            txt = generate_text_prompt(clss,captioning=False)
            for clip in os.listdir(os.path.join(split_set,clss)):
                '''file_path = hf_hub_download(
                    repo_id=os.listdir(os.path.join(split_set,clss)), filename=clip, repo_type="dataset"
                    )'''
                clip_folder = os.path.join(os.path.join(split_set,clss),clip)
                self.dataset.append([clip_folder,label,txt])

        
        
            
    def __getitem__(self,idx):
        clip_folder,label,txt = self.dataset[idx]
        
        container = av.open(clip_folder)
        video = read_video_pyav(container)
        #video_input = load_video(clip_folder)
        inputs = self.processor(list(video), return_tensors="pt",device='cpu')
        #video_input = self.processor(video_input,return_tensors="pt")
        
        
        return inputs,int(label)

    
    def __len__(self):
        return len(self.dataset)




    

            
    
    





