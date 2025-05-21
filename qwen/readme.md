# Qwen Video Captioning

This folder contains code for direct loading, training, and inference of the Qwen2.5-VL model for video captioning tasks.

## 📦 What This Includes

- A `main.py` script for training and inference
- A `dummy_video.py` script to generate synthetic video data
- LoRA-based fine-tuning support via the `--use_lora` flag
- Inference from `.mp4` video files using the `--video_inference` argument

## 🧪 Generate Dummy Video Data

To create synthetic training data, run:

```bash
python dummy_video.py
