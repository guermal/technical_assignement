








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
from spatial_transforms import *
from temporal_transforms import *
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
from utils import compute_captioning_metrics
from transformers import Qwen2VLForConditionalGeneration,AutoModelForVision2Seq,Qwen2VLProcessor,Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from qwen_vl_utils import process_vision_info

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



    def eval2(model, val_loader, tokenizer):
        model.eval()
        predictions = []
        references = []
        max_new_tokens = 1024
        model.train(False)
        with torch.set_grad_enabled(False):
            for sample in tqdm.tqdm(val_loader, desc="Evaluating"):
                #print(sample)
                sample = sample[0]  # because batch_size=1
                # Prepare the input without the label
                user_only = sample[1:2]  # get just the user message
                prompt = processor.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)

                _,video_tensor = process_vision_info(sample)

                model_inputs = processor(
                    text=[prompt],
                    videos=[video_tensor],
                    return_tensors="pt",
                    padding=True
                ).to(device)

                generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
                pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Get the reference label from sample
                ref = sample[2]["content"][0]["text"]

                print(f"Prediction: {pred}")
                print(f"Reference : {ref}")

                predictions.append(pred)
                references.append(ref)
            capt_metric = compute_captioning_metrics(predictions,references)
            bleu,rouge,meteor,em = capt_metric['BLEU'],capt_metric['ROUGE-L'],capt_metric['METEOR'],capt_metric['Exact Match']
            #compute_captioning_metrics(predictions, references)
        model.train(True)
        return bleu,rouge,meteor,em
    
    


    def train(model, train_loader, val_loader, tokenizer, num_epochs=3):
        model.train()
        best_acc = 0
        for epoch in range(num_epochs):
            total_loss = 0
            loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")
            i = 0
            for batch in loop:
                #inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model(**inputs, labels=labels)
                
                loss = outputs.loss

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                

            print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")
            #evaluate(model, val_loader, tokenizer)
            bleu,rouge,meteor,em = eval2(model, val_loader, tokenizer)
            if bleu > best_acc :
                best_acc = bleu
                # Assuming 'model' is wrapped in DataParallel
                if isinstance(model, torch.nn.DataParallel):
                    model_save = model.module  # Access the underlying model (without DataParallel wrapper)
                else:
                    model_save = model
                model_save = copy.deepcopy(model_save)
                # Move the model to CPU
                model_save.to('cpu')

                # Save the model's state_dict
                torch.save(model_save.state_dict(), os.path.join('./checkpoints','qwen.pth'))



    def generate_caption_from_video(video_path, model, processor, device):
        # Prepare the structured sample in chat format (like your dataset)
        sample = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that describes what is happening in videos."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": "What is the object doing in this video?"}
                ]
            }
        ]

        # Build the input prompt
        prompt = processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)

        # Process the video using your helper (returns tensor)
        _, video_tensor = process_vision_info(sample)

        # Encode inputs with processor
        model_inputs = processor(
            text=[prompt],
            videos=[video_tensor],
            return_tensors="pt",
            padding=True
        ).to(device)

        # Generate prediction
        generated_ids = model.generate(**model_inputs, max_new_tokens=64)

        # Decode result
        caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f" Generated Caption: {caption}")
        return caption





    def collate_fn(examples):

        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False) for example in examples
        ]  # Prepare texts for processing
        
        #video_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs
        _,video_inputs = process_vision_info(examples) #for example in examples]  # Process the images to extract inputs

        








        # Tokenize the texts and process the images
        batch = processor(
            text=texts,videos=video_inputs,return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
        

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655,151656]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.video_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        #batch.to(device)
        

        return batch  # Return the prepared batch
    

    
    







    #model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    #model_name = "Salesforce/blip-image-captioning-base"
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    from transformers import BitsAndBytesConfig

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,  torch_dtype=torch.bfloat16, quantization_config=bnb_config
    )
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    
    
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer
    if config['inference']:
        ckpt = torch.load('./checkpoints/qwen.pth')
        model.load_state_dict(ckpt,strict=False)
        model = model.to(device)
        generate_caption_from_video(config['video_inference'], model, processor, device)
    model = model.to(device)
    



    imagenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    scales = [1.0, 0.875, 0.75]
    crop_size = 224
    gamma_tau = 1
    num_frames = 8



    



    vid_seq_train = VideoCaptionDataset(config['gt_folder'],mode='train')


    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=config['batch_size'],shuffle=True, num_workers=4, pin_memory=True,collate_fn=collate_fn)

    validation_transforms = {
        'spatial':  Compose([CenterCropScaled(crop_size), #CenterCrop(crop_size),
                                #ToTensor(255),
                                #Normalize(imagenet_mean, imagenet_std)
                                ]),
        'temporal': TemporalRandomCrop(num_frames, gamma_tau)
    }


    vid_seq_test = VideoCaptionDataset(config['gt_folder'],mode='test')
    test_loader = torch.utils.data.DataLoader(vid_seq_test, batch_size=1, shuffle=False, num_workers=4,
                                                pin_memory=True,collate_fn=lambda x: x)#,collate_fn=collate_fn)


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
    print(model_name)
    if config['lora']:
        peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        #target_modules=None
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Adapt for Qwen layers if needed
    )
        model = get_peft_model(model, peft_config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total}")
    print(f" Trainable parameters: {trainable} / {total} ({100 * trainable / total:.2f}%)")
    print(model_name)


    if config['train']:
        train(model,train_loader,test_loader,tokenizer,num_epochs=5)
    
    '''training_args = TrainingArguments(
        output_dir="./qwen-vl-finetune",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vid_seq_train,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()'''





if __name__ == "__main__":
    main()