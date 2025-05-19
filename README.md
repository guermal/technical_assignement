# # 🧠 Technical Assignment

This repository contains the complete solution for a technical assignment involving dummy video dataset generation, vision and vision-language models, and training/inference pipelines.

---

## 📦 Data Creation

To generate the dummy dataset, run:

```bash
python dummy_data.py
```

This creates a folder named `synthetic_dataset/` with the following structure:

```text
synthetic_dataset/
├── train/
│   ├── class_0/
│   │   ├── video_0.pt
│   │   └── ...
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```

Each `.pt` file is a PyTorch tensor of shape `[T, C, H, W]` representing a synthetic video.

---

## 🏗️ Model Building

The `build_model.py` script defines three types of models:

### 🔹 Vision Transformer (ViT)

A fully vision-based transformer model (e.g., TimeSformer, VideoMAE) loaded from Hugging Face and followed by a classification head.

### 🔹 Vision-Language Models (VLMs)

Two types:
- **Classification**: Vision encoder + LLM (e.g., BERT)
- **Captioning**: Vision encoder + Causal LLM (e.g., GPT-2)

---

## 🏋️‍♂️ Model Training

### ▶️ Video Captioning

```bash
python main.py \
  --vision_model timesformer \
  --use_vlm True \
  --language_model gpt2 \
  --use_text True \
  --captioning True \
  --freeze_visionbackbone True \
  --use_lora_llm True
```

📌 Notes:
- `--use_lora_llm` is optional (default is `False` in `config_file.yaml`)
- `--freeze_visionbackbone` is recommended for single GPU training

---

### ▶️ Video Classification (Vision Only)

```bash
python main.py --vision_model timesformer
```

Optional flags:
- `--freeze_visionbackbone True` → train only the classification head
- `--use_lora_vision True` → apply LoRA for lightweight fine-tuning

---

### ▶️ Video Classification (Vision + Language)

```bash
python main.py \
  --vision_model timesformer \
  --language_model bert \
  --use_vlm True \
  --freeze_visionbackbone True
```

Use `--use_text True` to combine vision and text embeddings.

📌 All models save checkpoints to the `./checkpoints/` directory.

---

## 🔎 Inference

To run inference, use the same CLI args as training, but replace `main.py` with `inference.py`.

### Example: Captioning Inference

```bash
python inference.py \
  --vision_model timesformer \
  --use_vlm True \
  --language_model gpt2 \
  --use_text False \
  --captioning True \
  --freeze_visionbackbone True \
  --use_lora_llm True \
  --video_dir ./frames_see/
```

📌 You must specify `--video_dir` with frames for inference.  
📌 Checkpoints must be saved in `./checkpoints/`.

---

## 🛠️ Frame Generation for Inference

To convert dummy test tensors into image frames for inference:

```bash
python create_dummy.py
```

This will:
- Load the test videos from `dummy_data.py`
- Convert them into frames
- Save the frames to `./frames_see/`

---

## ✅ Summary

| Task                | Script            | Description                          |
|---------------------|-------------------|--------------------------------------|
| Dataset Generation  | `dummy_data.py`   | Creates dummy video tensors          |
| Model Definitions   | `build_model.py`  | ViT, VLM, classification, captioning |
| Training            | `main.py`         | Flexible CLI with transformer options|
| Inference           | `inference.py`    | Matches CLI used in training         |
| Frame Conversion    | `create_dummy.py` | Converts tensor → image sequences    |

---
