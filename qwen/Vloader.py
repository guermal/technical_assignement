from pathlib import Path
from torch.utils.data import Dataset
import random





def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": sample["label"][0]}],  # label becomes system message
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",   # Use "video" instead of "image"
                    "video": sample["video"],  # Path to video file
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]













class VideoCaptionDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.video_paths = list(Path(root_dir, mode).rglob("*.mp4"))
        self.class_names = {
            0: "red circle moving to the right",
            1: "blue circle moving to the left",
            2: "green square moving downward",
            3: "yellow square moving upward",
            4: "white triangle moving diagonally down-right",
            5: "magenta triangle moving diagonally up-left",
            6: "yellow circle not moving",
            7: "blue square moving randomly",
            8: "green triangle moving slowly",
            9: "yellow circle moving very fast to the right"
        }

        self.query_templates = [
            "What is the object doing?",
            "Describe the object's motion.",
            "What kind of movement is shown?",
            "How is the object moving in the video?",
            "Can you describe the motion in the scene?"
        ]


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        
        class_id = int(path.parent.name.split("_")[1])  # e.g., 'class_0' → 0

        sample = {
            "video": str(path),  # Pass path; processor can load it later
            "query": random.choice(self.query_templates),
            "label": [self.class_names[class_id]],
            "human_or_machine": 0
        }

        return format_data(sample)
