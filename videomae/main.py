








from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor,Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb
import tqdm
from Vloader import *
#from spatial_transforms import *
#from temporal_transforms import *
import numpy as np
import logging
import sys
import argparse
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from collections import OrderedDict
import argparse
import json
import yaml
import os
from distutils.util import strtobool
from utils import compute_captioning_metrics,compute_top_k
from transformers import Qwen2VLForConditionalGeneration,AutoModelForVision2Seq,Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration#,Qwen2_5VLProcessor
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
#from qwen_vl_utils import process_vision_info
from transformers import VideoMAEConfig, VideoMAEModel, AutoImageProcessor
import copy
from torch.nn.utils import clip_grad_norm_
# Function to load the configuration from a JSON or YAML file
def load_config(config_file):
    file_extension = os.path.splitext(config_file)[1].lower()
    
    if file_extension == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    elif file_extension == '.yaml' or file_extension == '.yml':
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml")
    
    return config

import os




def parse_args():
    parser = argparse.ArgumentParser(description="MLLM video classification task")
    parser.add_argument('--config', type=str, default='config_file.yaml', help="Path to the config file")
    parser.add_argument('--batch_size', type=int,  help="Batch size for training")
    parser.add_argument('--seq_len', type=int,  help="temporal input length")
    parser.add_argument('--gt_folder', type=str,  help="dataset folder")
    parser.add_argument('--lora', type=lambda x: bool(strtobool(x)), default=False,  help="if True uses lora")
    parser.add_argument('--inference', type=lambda x: bool(strtobool(x)), default=False,  help="if True run inference")
    parser.add_argument('--train', type=lambda x: bool(strtobool(x)), default=False,  help="if True run training")
    parser.add_argument('--video_inference', type=str,  help="path to video for inference")
    #parser.add_argument('--captioning', type=lambda x: bool(strtobool(x)), default=False,  help="if True use train for captioning not classification")



    
    args = parser.parse_args()
    return args












def main():
    



    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')
    args = parse_args()
    config = load_config(args.config)


    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.gt_folder:
        config['gt_folder'] = args.gt_folder
    if args.seq_len:
        config['seq_len'] = args.seq_len
    if args.lora:
        config['lora'] = args.lora
    if args.inference:
        config['inference'] = args.inference
    if args.video_inference:
        config['video_inference'] = args.video_inference
    if args.train:
        config['train'] = args.train



    def eval2(model, val_loader,epoch):
        model.eval()
        predictions = []
        references = []
        
        model.train(False)
        with torch.set_grad_enabled(False):
            loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}")
            i = 0
            for batch in loop:
                

                inputs,labels = batch
                #inputs.pixel_values = inputs.pixel_values.squeeze(1)
                inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items() if k == "pixel_values"}
                #inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor) and k != "labels"}
                
                #inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                #outputs = model(**inputs)#, labels=labels)
               
            

                predictions.append(outputs.detach().cpu().numpy())
                references.append(labels.detach().cpu().numpy())
            acc = compute_top_k(predictions,references)
            
        model.train(True)
        return acc
    
    


    def train(model, train_loader, val_loader, num_epochs=3):
        model.train()
        best_acc = 0
        for epoch in range(num_epochs):
            total_loss = 0
            loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
            i = 0
            for batch in loop:
                

                inputs,labels = batch
                #inputs.pixel_values = inputs.pixel_values.squeeze(1)
                inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items() if k == "pixel_values"}
                #inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor) and k != "labels"}
                
                #inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                #outputs = model(**inputs)#, labels=labels)
                
            
                loss = criterion(outputs,labels)

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                i+=1
                
                

            print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")
            #evaluate(model, val_loader, tokenizer)
            acc =  eval2(model, val_loader,epoch)
            print(f"Epoch {epoch+1} accuracy: {acc:.4f}")
            if acc > best_acc :
                best_acc = acc
                # Assuming 'model' is wrapped in DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    model_save = model.module  # Access the underlying model (without DataParallel wrapper)
                else:
                    model_save = model
                model_save = copy.deepcopy(model_save)
                # Move the model to CPU
                model_save.to('cpu')

                # Save the model's state_dict
                torch.save(model_save.state_dict(), os.path.join('./checkpoints','videomae.pth'))
            


    
    

    
    
    class VideoClassificationModel(torch.nn.Module):
        def __init__(self, num_classes, freeze_backbone=False):
            super().__init__()

            # Load backbone encoder only (no classification head)
            
            self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")#, attn_implementation="sdpa", torch_dtype=torch.float16)

            # Get hidden size from config
            hidden_size = getattr(self.backbone.config, 'hidden_size', getattr(self.backbone.config, 'hidden_dim', None))
            assert hidden_size is not None, "Backbone config must define hidden_size or hidden_dim"

            # Custom classifier head
            self.classifier = nn.Linear(hidden_size, num_classes)
            self.hidden_size = hidden_size
            # Optionally freeze backbone (can be overridden later if using LoRA)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

        def forward(self, inputs):
            outputs = self.backbone(**inputs)
            
            outputs = outputs.last_hidden_state.mean(dim=1)
            return self.classifier(outputs)


    from transformers import VideoMAEForVideoClassification
    
    
    model = VideoClassificationModel(num_classes=10)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    #processor = AutoVideoProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", device=device)
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")#,use_fast=False)#,device=device)
    
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #processor = AutoProcessor.from_pretrained(model_name)
    #tokenizer = processor.tokenizer
    
    



    



    



    vid_seq_train = base_loader(config['gt_folder'],num_classes=10,processor=processor,device=device,mode='train')


    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=config['batch_size'],shuffle=True, num_workers=4, pin_memory=True)#,collate_fn=collate_fn)

    


    vid_seq_test = base_loader(config['gt_folder'],num_classes=10,processor=processor,device=device,mode='test')
    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1, shuffle=False, num_workers=4,
                                                pin_memory=True)#,collate_fn=lambda x: x)#,collate_fn=collate_fn)


    num_epochs = 5
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f" Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    

    '''if config['inference']:
        ckpt = torch.load('./checkpoints/videomae.pth')
        model.load_state_dict(ckpt,strict=False)
        model = model.to(device)
        generate_caption_from_video(config['video_inference'], model, processor, device)'''
    model = model.to(device)
    if config['train']:
        train(model,train_loader,test_loader,num_epochs=5)
    
    




if __name__ == "__main__":
    main()