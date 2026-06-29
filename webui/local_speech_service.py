import os
import sys
import torch
from loguru import logger

# Ensure we can import MingAudio from cookbooks
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from cookbooks.test import MingAudio

class LocalSpeechService:
    def __init__(self, default_model_path="inclusionAI/Ming-omni-tts-0.5B"):
        self.model_path = None
        self.model = None
        self.sample_rate = 16000
        # Automatically load the default model
        # self.load_model(default_model_path)
    
    def load_model(self, model_path):
        # Resolve latest checkpoint if model_path is a directory without config.json
        if os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by checkpoint number
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                latest_checkpoint = checkpoints[-1]
                resolved_model_path = os.path.join(model_path, latest_checkpoint)
                logger.info(f"Resolved latest checkpoint: {resolved_model_path} from {model_path}")
                model_path = resolved_model_path
                
        if self.model_path == model_path and self.model is not None:
            logger.info(f"Model {model_path} is already loaded.")
            return True, f"Model {model_path} is already loaded."
            
        logger.info(f"Loading model from {model_path}...")
        try:
            # Unload previous model to free VRAM
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
                
            self.model = MingAudio(model_path)
            self.model_path = model_path
            self.sample_rate = self.model.sample_rate
            logger.info(f"Successfully loaded model from {model_path}")
            return True, f"成功加载模型: {model_path}"
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.model = None
            self.model_path = None
            return False, f"加载模型失败: {e}"
            
    def get_status(self) -> str:
        if self.model is None:
            return "⚠️ 模型未加载"
        return f"✅ 已加载: {self.model_path}"

    def speech_generation(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("模型未加载，请先点击『重新加载模型』按鈕。")
        return self.model.speech_generation(*args, **kwargs)
