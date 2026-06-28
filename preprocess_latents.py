import os
import json
import argparse
import torch
import torchaudio
from tqdm import tqdm
from modeling_bailingmm_training import BailingMMForFineTuning

def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files to latent features.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model containing AudioVAE")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to output JSONL dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the .pt latent files")
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_name_or_path}...")
    model = BailingMMForFineTuning.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True
    ).to(device)
    
    # We only need the audio VAE
    model.eval()

    data = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Processing {len(data)} items...")
    
    out_file = open(args.output_jsonl, 'w', encoding='utf-8')

    with torch.no_grad():
        for i, item in enumerate(tqdm(data)):
            audio_path = item.get("audio_path", "")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                continue

            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            if sr != args.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, args.target_sample_rate)
                waveform = resampler(waveform)

            # waveform: [1, T]
            waveform = waveform.squeeze(0).unsqueeze(0) 
            waveform_lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
            
            # We must convert to bfloat16 because the model was loaded in bfloat16
            waveform = waveform.to(device).to(torch.bfloat16)
            waveform_lengths = waveform_lengths.to(device)

            latents, frame_nums = model.audio.encode_latent(waveform, waveform_lengths)
            
            # latents shape: [1, max_frame, D], we squeeze to [max_frame, D]
            latents = latents.squeeze(0).cpu()
            frame_nums_val = frame_nums.item()
            
            latent_filename = f"latent_{i}.pt"
            latent_path = os.path.join(args.output_dir, latent_filename)
            
            # Save latent tensor
            torch.save(latents, latent_path)
            
            # Create new JSONL item
            new_item = {
                "text": item.get("text", ""),
                "latent_path": os.path.abspath(latent_path),
                "frame_nums": frame_nums_val
            }
            out_file.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    out_file.close()
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
