# Ming-omni-tts-0.5B 全参微调完整指南

这份文档将带您从零开始，跑通 Ming-omni-tts-0.5B 模型的环境搭建、数据准备、模型全参微调，以及微调后的推理测试全流程。

---

## 1. 环境搭建

强烈建议您使用 Anaconda 创建一个隔离的虚拟环境。

```bash
# 1. 创建并激活虚拟环境 (推荐 Python 3.10+)
conda create -n ming_tts python=3.10 -y
conda activate ming_tts

# 2. 安装 PyTorch (请根据您的 CUDA 版本修改，此处以 CUDA 12.1 为例)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安装项目原始依赖
pip install -r requirements.txt

# 4. 安装微调核心依赖 (Hugging Face 生态)
pip install transformers accelerate datasets
```

**显存优化提醒**：如果您在全参微调时显存不足（例如出现 Out-Of-Memory），建议安装 DeepSpeed 插件来极大地降低显存占用：
`pip install deepspeed`

---

## 2. 数据准备

微调所需的数据格式极其简单。您只需要准备一个包含若干条数据的 `train.jsonl` 文件。
如果您手里有的是标准的 CSV（两列，包含文件名和文本），可以直接使用我们提供的 `convert_ljspeech.py` 脚本来生成 JSONL：

```bash
python convert_ljspeech.py -i 您的原始标注.csv -w ./您的wav文件夹路径 -o train.jsonl
```

### 最终的 `train.jsonl` 格式说明：
每一行是一个合法的 JSON 字符串，包含两项必须的 Key：`text` 和 `audio_path`。
*(注意：音频长度建议在 2 秒到 15 秒之间，过长可能导致 OOM)*

```json
{"text": "你好，这是第一句微调训练的测试语音。", "audio_path": "/绝对路径或相对路径/your_dataset/wavs/sample_1.wav"}
{"text": "Ming-omni-tts 的全参微调其实非常简单。", "audio_path": "/绝对路径或相对路径/your_dataset/wavs/sample_2.wav"}
```

---

## 3. 开始全参微调

在项目根目录下，直接运行 `finetune.py` 启动脚本即可开始全参微调。

```bash
python finetune.py \
    --model_name_or_path "inclusionAI/Ming-omni-tts-0.5B" \
    --train_data_path "train.jsonl" \
    --output_dir "./finetuned_Ming_0.5B" \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --save_steps 100 \
    --logging_steps 10
```

**显存优化提醒**：如果 `batch_size=1` 仍然发生 OOM，请执行以下两步：
1. 脚本默认已经开启了 `bf16=True` 半精度。
2. 使用 Accelerate / Deepspeed 启动：
`accelerate launch --use_deepspeed --zero_stage=2 finetune.py ...`

训练结束后，模型权重、配置和 Tokenizer 会自动保存在 `./finetuned_Ming_0.5B` 文件夹下。

---

## 4. 微调后的推理测试

模型训练完成后，我们要验证微调后的模型生成效果。
您只需要修改项目原有的 `cookbooks/test.py` 脚本，将模型的加载路径指向您的输出目录即可。

### 步骤：
1. 打开 `cookbooks/test.py`。
2. 将原始的 `inclusionAI/Ming-omni-tts-0.5B` 替换为 `./finetuned_Ming_0.5B`。

```python
# 修改前：
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    "inclusionAI/Ming-omni-tts-0.5B", 
    ...
)
tokenizer = BailingTokenizer.from_pretrained("inclusionAI/Ming-omni-tts-0.5B", ...)

# 修改后 (指向您的微调文件夹)：
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    "./finetuned_Ming_0.5B", 
    ...
)
tokenizer = BailingTokenizer.from_pretrained("./finetuned_Ming_0.5B", ...)
```

保存后，运行官方的测试脚本即可生成测试音频：
```bash
python cookbooks/test.py
```
*(生成的音频默认保存在项目根目录下的 `.wav` 文件中，您可以播放试听，感受微调带来的变化！)*
