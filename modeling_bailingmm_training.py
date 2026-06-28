import torch
import torch.nn as nn
from modeling_bailingmm import BailingMMNativeForConditionalGeneration

class BailingMMForFineTuning(BailingMMNativeForConditionalGeneration):
    _supports_sdpa = True
    
    def forward(self, texts, latents=None, frame_nums=None, waveforms=None, waveform_lengths=None, **kwargs):
        device = self.device
        bsz = len(texts)
        
        if latents is None or frame_nums is None:
            raise ValueError("Precomputed 'latents' and 'frame_nums' are required for training. Please run preprocess_latents.py and use the updated dataset.")
            
        latents = latents.to(device)
        frame_nums = frame_nums.to(device)
        
        patch_size = self.patch_size
        T_frame = latents.size(1)
        T_patch = T_frame // patch_size
        valid_frames = T_patch * patch_size
        
        latents = latents[:, :valid_frames, :] # [B, T_patch * patch_size, D]
        latents_patched = latents.reshape(bsz, T_patch, patch_size, self.latent_dim)
        
        # 3. Project patched latents to LLM input dimension
        audio_embeds = self.linear_proj_audio(latents_patched.reshape(-1, patch_size, self.latent_dim))
        audio_embeds = audio_embeds.reshape(bsz, T_patch, -1) # [B, T_patch, H]
        
        inputs_embeds_list = []
        
        for i in range(bsz):
            text = texts[i]
            # Match inference prompt EXACTLY:
            # prompt = "Please generate speech based on the following description.\n"
            # prompt2 = " Text input:\n"
            prompt_str = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "Please generate speech based on the following description.\n"
                " Text input:\n"
                f"{text}<|im_end|>\n"
                "<|im_start|>assistant\n<audio>"
            )
            prompt_ids = self.tokenizer.encode(prompt_str)
            prompt_embeds = self.model.get_input_embeddings()(torch.tensor(prompt_ids, device=device))
            
            # Use actual length for this item
            T_patch_i = min(frame_nums[i].item(), T_patch)
            
            # Combine
            full_embeds = torch.cat([prompt_embeds, audio_embeds[i, :T_patch_i]], dim=0)
            inputs_embeds_list.append((full_embeds, len(prompt_ids)))
            
        # Pad inputs_embeds
        max_seq_len = max(emb[0].size(0) for emb in inputs_embeds_list)
        batch_inputs_embeds = torch.zeros(bsz, max_seq_len, self.model.config.hidden_size, device=device, dtype=audio_embeds.dtype)
        attention_mask = torch.zeros(bsz, max_seq_len, device=device, dtype=torch.long)
        
        prompt_lengths = []
        for i, (emb, prompt_len) in enumerate(inputs_embeds_list):
            seq_len = emb.size(0)
            batch_inputs_embeds[i, :seq_len] = emb
            attention_mask[i, :seq_len] = 1
            prompt_lengths.append(prompt_len)
            
        # 5. Forward LLM
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill((attention_mask == 0), 1)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=batch_inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1] # [B, S, H]
        
        # 6. Compute Flow Matching Loss
        loss = 0
        valid_items = 0
        
        for i in range(bsz):
            T_patch_i = min(frame_nums[i].item(), T_patch)
            if T_patch_i <= 0:
                continue
                
            prompt_len = prompt_lengths[i]
            # The token that predicts patch 0 is the <audio> token, which is at index prompt_len - 1
            audio_start = prompt_len
            
            cond = hidden_states[i, audio_start-1 : audio_start+T_patch_i-1, :] # [T_patch_i, H]
            target = latents_patched[i, :T_patch_i] # [T_patch_i, patch_size, D]
            
            # Create latent_history
            history = torch.zeros(T_patch_i, self.history_patch_size, self.latent_dim, device=device, dtype=latents.dtype)
            for t in range(T_patch_i):
                start_idx = t * patch_size - self.history_patch_size
                if start_idx < 0:
                    valid_len = t * patch_size
                    if valid_len > 0:
                        history[t, -valid_len:, :] = latents[i, :valid_len, :]
                else:
                    history[t, :, :] = latents[i, start_idx:start_idx+self.history_patch_size, :]
                    
            cond = cond.unsqueeze(1) # [T_patch_i, 1, H]
            mask = torch.ones(T_patch_i, 1, device=device)
            item_loss = self.flowloss(cond=cond, target=target, latent_history=history, mask=mask, patch_size=patch_size)
            loss += item_loss
            valid_items += 1
            
        if valid_items > 0:
            loss = loss / valid_items
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        return {"loss": loss}
