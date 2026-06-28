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
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        audio_path = item.get("audio_path", "")
        
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
            
        return {
            "text": text,
            "waveform": waveform.squeeze(0)  # [T]
        }

class MingOmniTTSDataCollator:
    def __call__(self, batch):
        texts = [item["text"] for item in batch]
        waveforms = [item["waveform"] for item in batch]
        
        # Pad waveforms
        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths)
        
        padded_waveforms = torch.zeros((len(batch), max_len))
        waveform_lengths = torch.tensor(lengths, dtype=torch.long)
        
        for i, w in enumerate(waveforms):
            padded_waveforms[i, :lengths[i]] = w
            
        return {
            "texts": texts,
            "waveforms": padded_waveforms,
            "waveform_lengths": waveform_lengths
        }
