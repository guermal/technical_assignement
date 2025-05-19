import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from build_model import *
from train_code import *
from distutils.util import strtobool

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
    parser.add_argument('--learning_rate', type=float,  help="init Learning rate for optimization")
    parser.add_argument('--dropout', type=float,  help="dropout for optimization")
    parser.add_argument('--vision_model', type=str,  help="which vision model to use")
    parser.add_argument('--language_model', type=str,  help="which language model to use")
    parser.add_argument('--seq_len', type=int,  help="temporal input length")
    parser.add_argument('--num_classes', type=int,  help="num_classes")
    parser.add_argument('--gt_folder', type=str,  help="dataset folder")
    parser.add_argument('--freeze_visionbackbone', type=lambda x: bool(strtobool(x)), default=False,  help="if True freezes backbone")
    parser.add_argument('--use_lora_vision', type=lambda x: bool(strtobool(x)), default=False,  help="if True uses lora")
    parser.add_argument('--freeze_llmbackbone', type=lambda x: bool(strtobool(x)), default=False,  help="if True freezes backbone")
    parser.add_argument('--use_lora_llm', type=lambda x: bool(strtobool(x)), default=False,  help="if True uses lora")
    parser.add_argument('--inference', type=lambda x: bool(strtobool(x)), default=False,  help="if True run inference")
    parser.add_argument('--use_vlm', type=lambda x: bool(strtobool(x)), default=False,  help="if True run uses a vlm model not only a vision transformer model")
    parser.add_argument('--use_text', type=lambda x: bool(strtobool(x)), default=False,  help="if True use also text input")
    parser.add_argument('--captioning', type=lambda x: bool(strtobool(x)), default=False,  help="if True use train for captioning not classification")
    parser.add_argument('--video_dir', type=str,  help="dir to video input")
    parser.add_argument('--text_input', type=str,  help="text input")
    

    



    
    args = parser.parse_args()
    return args





def main():
    torch.autograd.set_detect_anomaly(True)



    args = parse_args()
    config = load_config(args.config)


    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.dropout:
        config['dropout'] = args.dropout
    if args.vision_model:
        config['vision_model'] = args.vision_model
    if args.language_model:
        config['language_model'] = args.language_model
    if args.gt_folder:
        config['gt_folder'] = args.gt_folder
    if args.seq_len:
        config['seq_len'] = args.seq_len
    if args.num_classes:
        config['num_classes'] = args.num_classes
    if args.freeze_visionbackbone:
        config['freeze_visionbackbone'] = args.freeze_visionbackbone
    if args.freeze_llmbackbone:
        config['freeze_llmbackbone'] = args.freeze_llmbackbone
    if args.use_lora_vision:
        config['use_lora_vision'] = args.use_lora_vision
    if args.use_lora_llm:
        config['use_lora_llm'] = args.use_lora_llm
    if args.use_vlm:
        config['use_vlm'] = args.use_vlm
    if args.use_text:
        config['use_text'] = args.use_text
    if args.inference:
        config['inference'] = args.inference
    if args.captioning:
        config['captioning'] = args.captioning
    if args.video_dir:
        config['video_dir'] = args.video_dir
    if args.text_input:
        config['text_input'] = args.text_input
    
    
    seed=42
    #torch.manual_seed(42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    CLASS_DESCRIPTIONS = {
    "class_0": "a blue circle moving from left to right",
    "class_1": "a red circle moving from right to left",
    "class_2": "a green square moving downward",
    "class_3": "a yellow square moving upward",
    "class_4": "a white triangle moving diagonally down and right",
    "class_5": "a magenta triangle moving diagonally up and left",
    "class_6": "a yellow circle that is static",
    "class_7": "a red square moving in a random direction",
    "class_8": "a green triangle moving slowly",
    "class_9": "a cyan circle moving fast horizontally",
}
    video_input = read_video(config['video_dir'])
    


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    video_transformers = {
    "videomae": ["MCG-NJU/videomae-base",["query", "key", "value", "output.dense"]],
    "vivit": ["google/vivit-b-16x2-kinetics400",["attention.query", "attention.value"]],
    "timesformer": ["facebook/timesformer-base-finetuned-k400",["attention.attention.qkv","attention.output.dense", "temporal_attention.attention.qkv","temporal_attention.output.dense"]]
    
}

    language_models = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-base",
    "distilbert": "distilbert-base-uncased",
    "albert": "albert-base-v2",
    "electra": "google/electra-base-discriminator"
    
}
    causal_language_models = {
   "gpt2": "gpt2",
   "gpt2-medium": "gpt2-medium",
   "gpt-neo": "EleutherAI/gpt-neo-1.3B",
   "gpt-j": "EleutherAI/gpt-j-6B",
   "gpt4all": "nomic-ai/gpt4all-j",
   "qwen-1b": "Qwen/Qwen-1B",
   "qwen-7b": "Qwen/Qwen-7B",
   "phi": "microsoft/phi-1_5",
   "mistral": "mistralai/Mistral-7B-v0.1"
}

    if args.use_vlm:

        if args.captioning:
            if config['use_lora_llm']:
                ckpt = torch.load('./checkpoints/new_timesformer_gpt2_fvbb_lora.pth')
            else:
                ckpt = torch.load('./checkpoints/new_timesformer_gpt2_fvbb.pth')

            model = build_vlm_model(video_transformers[config['vision_model']][0],causal_language_models[config['language_model']],config['num_classes'],video_transformers[config['vision_model']][1],llm_target_modules=None,captioning=config['captioning'],use_text=config['use_text'],use_lora_vison=config['use_lora_vision'],use_lora_llm=config['use_lora_llm'],freeze_visionbackbone=config['freeze_visionbackbone'],freeze_llmbackbone=config['freeze_llmbackbone'])
            model_name = config['vision_model']+'_'+config['language_model']
            model.load_state_dict(ckpt,strict=False)
            _,output_text = model(torch.unsqueeze(video_input,0),text_inputs=None)
            print(output_text)
        else:
            if config['use_text']:
                if config['freeze_llmbackbone']:
                    ckpt = torch.load('./checkpoints/new_timesformer_bert_fvbb_fvllm.pth')
                elif config['use_lora_llm']:
                    ckpt = torch.load('./checkpoints/new_timesformer_bert_fvbb_lorallm.pth')

            else:
                ckpt = torch.load('./checkpoints/new_timesformer_bert_fvbb.pth')
            
            model = build_vlm_model(video_transformers[config['vision_model']][0],language_models[config['language_model']],config['num_classes'],video_transformers[config['vision_model']][1],llm_target_modules=None,captioning=config['captioning'],use_text=config['use_text'],use_lora_vison=config['use_lora_vision'],use_lora_llm=config['use_lora_llm'],freeze_visionbackbone=config['freeze_visionbackbone'],freeze_llmbackbone=config['freeze_llmbackbone'])
            model_name = config['vision_model']+'_'+config['language_model']
            model.load_state_dict(ckpt,strict=False)
            if config['use_text']:
                if config['text_input']:
                    output = model(torch.unsqueeze(video_input,0),[config['text_input']])
                    pclass = torch.argmax(output, dim=1).item()
                    pclass = 'class_'+str(pclass)
                    print('predicted class: ',CLASS_DESCRIPTIONS[pclass])
                else:
                    print('you should provide a text input for this model')
                    exit()
            else:
                output = model(torch.unsqueeze(video_input,0),text_inputs = None)
                pclass = torch.argmax(output, dim=1).item()
                pclass = 'class_'+str(pclass)
                print('predicted class: ',CLASS_DESCRIPTIONS[pclass])

    else:
        if config['freeze_visionbackbone']:
            ckpt = torch.load('./checkpoints/new_timesformer_fbb.pth')
        elif config['use_lora_vision']:
            ckpt = torch.load('./checkpoints/new_timesformer_lora.pth')
        else:
            ckpt = torch.load('./checkpoints/new_timesformer_fft.pth')
        model,_ = build_vision_model(video_transformers[config['vision_model']][0],video_transformers[config['vision_model']][1], config['num_classes'], use_lora=config['use_lora_vision'], freeze_backbone=config['freeze_visionbackbone'])
        model_name = config['vision_model']
        model.load_state_dict(ckpt,strict=False)
        output,_ = model(torch.unsqueeze(video_input,0))
        pclass = torch.argmax(output, dim=1).item()
        pclass = 'class_'+str(pclass)
        print('predicted class: ',CLASS_DESCRIPTIONS[pclass])


if __name__ == "__main__":
    main()