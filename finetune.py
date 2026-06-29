import os
import argparse
from transformers import TrainingArguments, Trainer, AutoTokenizer
from train_dataset import MingOmniTTSDataset, MingOmniTTSDataCollator
from modeling_bailingmm_training import BailingMMForFineTuning
import torch

def main():
    parser = argparse.ArgumentParser(description="Full-Parameter Fine-tuning for Ming-omni-tts-0.5B")
    parser.add_argument("--model_name_or_path", type=str, default="inclusionAI/Ming-omni-tts-0.5B", help="Path to pretrained model")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to train jsonl dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    args = parser.parse_args()

    print(f"Loading model from {args.model_name_or_path}...")
    # Load with bfloat16 to save memory and accelerate training
    model = BailingMMForFineTuning.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", # Restored to SDPA since we added _supports_sdpa=True
        trust_remote_code=True
    )
    
    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, config=model.config, trust_remote_code=True)
    model.tokenizer = tokenizer
    
    # We are doing full-parameter fine-tuning, so we do not freeze any parameters 
    # but some components like Audio VAE might not need training.
    # Usually the audio VAE (tokenizer) is frozen in Audio-LLM fine-tuning:
    print("Freezing Audio VAE...")
    for param in model.audio.parameters():
        param.requires_grad = False
        
    model.train()
    
    print(f"Preparing dataset from {args.train_data_path}...")
    train_dataset = MingOmniTTSDataset(
        jsonl_file=args.train_data_path,
        target_sample_rate=model.audio.config.sample_rate if hasattr(model.audio.config, "sample_rate") else 16000
    )
    
    data_collator = MingOmniTTSDataCollator()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True, # Recommended for A100/H100/RTX30xx/40xx
        remove_unused_columns=False, # Important: prevent Trainer from removing our custom arguments
        dataloader_num_workers=0, # Changed to 0 for native Windows compatibility
        gradient_checkpointing=True, # Enables activation checkpointing to drastically save VRAM
        report_to="tensorboard" # Or tensorboard/wandb
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    print("Starting full-parameter fine-tuning...")
    trainer.train()
    
    print(f"Saving final model to {args.output_dir}...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
