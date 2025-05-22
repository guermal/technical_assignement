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
```

--

```text
synthetic_videos/
├── train/
│   ├── class_0/
│   │   ├── video_0.mp4
│   │   ├── video_1.mp4
│   │   └── ...
│   ├── class_1/
│   └── ...
└── test/
    ├── class_0/
    ├── class_1/
    └── ...
```
Each video shows a shape (circle, square, triangle) moving in a distinct pattern and color, representing a unique class.

## 🏗️ Model: Qwen2.5-VL

This project uses the Qwen2.5-VL-3B-Instruct model from Alibaba, which can process both vision and language inputs to generate captions.

## 🏋️‍♂️ Training

To train the model using LoRA adapters:
```bash
python main.py --use_lora True
```
This command will load the model, apply LoRA for parameter-efficient fine-tuning, and begin training on the synthetic videos.

📌 Checkpoints and training logs will be saved automatically.

## 🔎 Inference

To run inference and generate a caption for a .mp4 video:
```bash
python main.py --use_lora True --inference True --video_inference path/to/your_video.mp4
```
📌 Replace path/to/your_video.mp4 with your actual video path.

📌 The model will return a natural-language caption describing the motion and shape in the video.

## 📂 File Handling

The following files and folders are excluded from version control:

*.mp4
*.pth
synthetic_videos/
checkpoints/
Make sure these patterns are included in your .gitignore.

---

## ✅ Summary

Task	Script	Description
Generate Dataset	dummy_video.py	Creates synthetic .mp4 videos by class
Train Model	main.py	Fine-tunes Qwen2.5-VL using LoRA
Run Inference	main.py	Generates a caption from a video



| Task                | Script                         | Description                          |
|---------------------|--------------------------------|--------------------------------------|
| Dataset Generation  | `dummy_video.py`               | Creates dummy video tensors          |
| Train Model         | `main.py (with params)`        | Fine-tunes Qwen2.5-VL using LoRA     |
| Run Inference       | `main.py (with params)`        | Generates a caption from a video     |


---


## ✨ Credits

This work builds upon the Qwen2.5-VL-3B-Instruct model released by Alibaba, and leverages Hugging Face's Transformers and PEFT libraries.

usefull websites : 
 [Qwen2.5-VL-3B-Instruct]([https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl#4-fine-tune-the-model-using-trl)

To go deeper in QWEN code (data_preparation, model_init...)
 [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL/tree/d2240f11656bfe404b9ba56db4e51cd09f522ff1)

For full manipulation of hugging-face transformers refer to [API](https://huggingface.co/docs/transformers/index)

