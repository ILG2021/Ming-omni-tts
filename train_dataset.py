import json
import torch
from torch.utils.data import Dataset
import torchaudio

class MingOmniTTSDataset(Dataset):
    def __init__(self, jsonl_file, target_sample_rate=16000):
        super().__init__()
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        latent_path = item.get("latent_path")
        frame_nums = item.get("frame_nums")
        
        if latent_path is None or frame_nums is None:
            raise ValueError(f"Missing 'latent_path' or 'frame_nums' in dataset item. Please run preprocess_latents.py first. Item: {item}")
            
        latents = torch.load(latent_path, map_location='cpu') # [T, D]
        
        return {
            "text": text,
            "latents": latents,
            "frame_nums": frame_nums
        }

class MingOmniTTSDataCollator:
    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        latents_list = [item["latents"] for item in batch]
        frame_nums_list = [item["frame_nums"] for item in batch]
        
        # Pad latents [T, D]
        lengths = [l.shape[0] for l in latents_list]
        max_len = max(lengths)
        dim = latents_list[0].shape[1]
        
        padded_latents = torch.zeros((len(batch), max_len, dim), dtype=latents_list[0].dtype)
        frame_nums = torch.tensor(frame_nums_list, dtype=torch.long)
        
        for i, l in enumerate(latents_list):
            padded_latents[i, :lengths[i], :] = l
            
        return {
            "texts": texts,
            "latents": padded_latents,
            "frame_nums": frame_nums
        }
