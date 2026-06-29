import base64
import gzip
import io
import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import gradio as gr

from loguru import logger
from pypinyin import Style, pinyin

# --- 静态数据 ---
DROPDOWN_CHOICES = {
    "bgm_genres": list(
        set(
            [
                "独立民谣：吉他驱动",
                "当代古典音乐：钢琴驱动",
                "现代流行抒情曲：钢琴驱动的",
                "乡村音乐",
                "流行乐",
                "流行摇滚",
                "电子舞曲",
                "雷鬼顿",
                "迪斯科",
            ]
        )
    ),
    "swb_genres": list(set(["流行摇滚", "迪斯科", "电子舞曲"])),
    "bgm_moods": list(
        set(
            [
                "鼓舞人心/充满希望",
                "壮丽宏大",
                "快乐",
                "平静放松",
                "自信/坚定",
                "轻快无忧无虑",
                "活力四射/精力充沛",
                "悲伤哀愁",
                "温暖/友善",
                "兴奋",
            ]
        )
    ),
    "swb_moods": list(set(["快乐", "兴奋", "活力四射"])),
    "bgm_instruments": list(
        set(["低音鼓", "电吉他", "合成拨弦", "合成铜管乐器", "架子鼓", "定音鼓"])
    ),
    "swb_instruments": list(set(["电吉他", "合成铜管乐器", "架子鼓"])),
    "bgm_themes": list(
        set(
            [
                "励志",
                "生日",
                "分手",
                "旅行",
                "运动",
                "剧院音乐厅",
                "音乐现场",
                "节日",
                "好时光",
                "庆典与喜悦",
            ]
        )
    ),
    "swb_themes": list(set(["生日", "旅行", "运动"])),
    "dialects": list(set(["四川话", "广粤话"])),
    "emotions": list(set(["愤怒", "高兴", "悲伤"])),
    "env_sounds": [],  # 原 Demo 未使用
}

IP_DICT = {
    "爱新觉罗·弘历": "雍正王朝_爱新觉罗·弘历",
    "爱新觉罗·弘时": "雍正王朝_爱新觉罗·弘时",
    "曹操": "三国演义_曹操",
    "刁光斗": "大宋提刑官_刁光斗",
    "丰兰息": "且试天下_丰兰息",
    "公孙胜": "水浒传_公孙胜",
    "关涛": "幸福到万家_关涛",
    "关雪": "哈尔滨一九四四_关雪",
    "郭启东": "风吹半夏_郭启东",
    "何幸福": "幸福到万家_何幸福",
    "灰太狼": "喜羊羊与灰太狼_灰太狼",
    "康熙": "康熙王朝_康熙",
    "李蔷": "法医秦明_李蔷",
    "李涯": "潜伏_李涯",
    "卢怀德": "大宋提刑官_卢怀德",
    "陆建勋": "老九门_陆建勋",
    "陆桥山": "潜伏_陆桥山",
    "穆晚秋": "潜伏_穆晚秋",
    "年羹尧": "雍正王朝_年羹尧",
    "潘金莲": "水浒传_潘金莲",
    "潘越": "哈尔滨一九四四_潘越",
    "佩奇": "小猪佩奇_佩奇",
    "齐铁嘴": "老九门_齐铁嘴",
    "秦明": "法医秦明_秦明",
    "青年康熙": "康熙王朝_青年康熙",
    "裘德考": "老九门_裘德考",
    "荣妃": "康熙王朝_荣妃",
    "四郎": "甄嬛传_四郎",
    "司徒末": "致我们暖暖的小时光_司徒末",
    "宋慈": "大宋提刑官_宋慈",
    "苏麻喇姑": "康熙王朝_苏麻喇姑",
    "苏培盛": "甄嬛传_苏培盛",
    "孙颖莎": "孙颖莎_孙颖莎",
    "唐僧": "西游记_唐僧",
    "铁铉": "山河月明_铁铉",
    "王翠平": "潜伏_王翠平",
    "吴三桂": "康熙王朝_吴三桂",
    "邬思道": "雍正王朝_邬思道",
    "武松": "水浒传_武松",
    "萧崇": "少年歌行_萧崇",
    "孝庄": "康熙王朝_孝庄",
    "许半夏": "风吹半夏_许半夏",
    "徐文昌": "安家_徐文昌",
    "野原美伢 (美伢)": "蜡笔小新_野原美伢 (美伢)",
    "野原新之助 (小新)": "蜡笔小新_野原新之助 (小新)",
    "雍正": "雍正王朝_雍正",
    "余则成": "潜伏_余则成",
    "张启山": "老九门_张启山",
    "朱标": "山河月明_朱标",
    "朱棣": "山河月明_朱棣",
    "朱颜": "玉骨遥_朱颜",
    "朱元璋": "山河月明_朱元璋",
    "左蓝": "潜伏_左蓝",
}


# 辅助函数
def load_and_merge_ips(original_dict: dict, filepath: str) -> dict:
    """
    从txt文件加载新的IP，按拼音排序后，追加到原始字典末尾。
    支持两种格式: 'Key:Value' 或仅 'Value' (此时Key和Value相同)。

    :param original_dict: 原始的IP_DICT。
    :param filepath: 包含新IP的txt文件路径。
    :return: 一个合并后的新字典。
    """
    new_ips = {}
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 忽略空行或注释行
                if not line or line.startswith("#"):
                    continue

                # 判断行中是否包含冒号来决定解析方式
                if ":" in line:
                    # 格式为 'Key:Value'
                    try:
                        key, value = line.split(":", 1)
                        new_ips[key.strip()] = value.strip()
                    except ValueError:
                        logger.warning(f"无法解析行: {line}，格式应为 'Key:Value'")
                else:
                    # 格式仅为 'Value'，此时key和value相同
                    key = value = line
                    new_ips[key] = value

    # 仅对从文件读取的新IP按拼音进行排序
    sorted_new_ips = dict(
        sorted(new_ips.items(), key=lambda item: pinyin(item[0], style=Style.NORMAL))
    )

    # 合并字典：将排序后的新IP追加到原始字典后面
    merged_dict = original_dict.copy()
    merged_dict.update(sorted_new_ips)

    return merged_dict


IP_DICT = load_and_merge_ips(IP_DICT, "uniaudio_ip_list.txt")

REFERENCE_AUDIO_WARNING = "**⚠️ 注意：参考音频建议长度约为 3-7 秒，过长的音频可能导致输出异常。您可以使用下方的音频控件对音频进行剪辑。**"


class MingOmniTTSDemoTab:
    """
    独立实现了基于 Ming-Omni-TTS V4 MOE (WebGW) 的请求逻辑。
    """

    def __init__(self, local_service):
        self.local_service = local_service

    def create_tab(self):
        with gr.TabItem("Ming-omni-tts"):
            gr.Markdown("## Ming-omni-tts 综合能力演示")

            with gr.Tabs():
                # --- Tab 1: 指令TTS ---
                with gr.TabItem("指令TTS (Instruct TTS)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            i_tts_type = gr.Dropdown(
                                [
                                    ("方言 (dialect)", "dialect"),
                                    ("情感 (emotion)", "emotion"),
                                    ("IP (IP)", "IP"),
                                    ("风格 (style)", "style"),
                                    ("基础 (basic)", "basic")
                                ],
                                label="指令类型",
                                value="emotion",
                            )
                            i_tts_text = gr.Textbox(label="合成文本", info="输入要合成的语音文本。")
                            gr.Markdown(REFERENCE_AUDIO_WARNING)
                            i_tts_prompt = gr.Audio(
                                type="filepath",
                                label="参考音频 (3-7秒)上传一段清晰的人声音频用于克隆基础音色。",
                                sources=["upload", "microphone"],
                            )

                            with gr.Accordion("指令详情 (根据指令类型填写)", open=True):
                                i_tts_emotion = gr.Dropdown(
                                    DROPDOWN_CHOICES["emotions"], label="情感", value="高兴"
                                )
                                i_tts_dialect = gr.Dropdown(
                                    DROPDOWN_CHOICES["dialects"],
                                    label="方言",
                                    value="广粤话",
                                    visible=False,
                                )
                                i_tts_ip = gr.Dropdown(
                                    list(IP_DICT.keys()), label="IP角色", visible=False
                                )
                                i_tts_style = gr.Textbox(
                                    label="风格描述",
                                    info="e.g. 以洪亮有力的音量发声,展示出男性特有的坚韧与威严感。语速偏快,语调从头至尾保持流畅,特别是在结尾词句上略微放慢,增强权威与果决的语气",
                                    visible=False,
                                )
                                i_tts_speed = gr.Dropdown(
                                    ["慢速", "中速", "快速"],
                                    label="语速",
                                    value="中速",
                                    visible=False,
                                )
                                i_tts_pitch = gr.Dropdown(
                                    ["低", "中", "高"], label="基频", value="中", visible=False
                                )
                                i_tts_volume = gr.Dropdown(
                                    ["低", "中", "高"], label="音量", value="中", visible=False
                                )

                            i_tts_btn = gr.Button("生成指令语音", variant="primary")

                        with gr.Column(scale=1):
                            i_tts_status = gr.Markdown(value="💡 请选择指令类型并填写参数。")
                            i_tts_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

                    def update_details_visibility(instruct_type):
                        prompt_visible = instruct_type not in ["IP", "style"]
                        return {
                            i_tts_prompt: gr.update(visible=prompt_visible),
                            i_tts_emotion: gr.update(visible=instruct_type == "emotion"),
                            i_tts_dialect: gr.update(visible=instruct_type == "dialect"),
                            i_tts_ip: gr.update(visible=instruct_type == "IP"),
                            i_tts_style: gr.update(visible=instruct_type == "style"),
                            i_tts_speed: gr.update(visible=instruct_type == "basic"),
                            i_tts_pitch: gr.update(visible=instruct_type == "basic"),
                            i_tts_volume: gr.update(visible=instruct_type == "basic"),
                        }

                    i_tts_type.change(
                        fn=update_details_visibility,
                        inputs=i_tts_type,
                        outputs=[
                            i_tts_prompt,
                            i_tts_emotion,
                            i_tts_dialect,
                            i_tts_ip,
                            i_tts_style,
                            i_tts_speed,
                            i_tts_pitch,
                            i_tts_volume,
                        ],
                    )

                # --- Tab 2: 零样本TTS (音色克隆) ---
                with gr.TabItem("音色克隆 (Zero-shot TTS)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            zs_tts_text = gr.Textbox(
                                label="目标文本", info="输入您想合成的语音文本。"
                            )
                            gr.Markdown(REFERENCE_AUDIO_WARNING)
                            zs_tts_prompt = gr.Audio(
                                type="filepath",
                                label="参考音频 (3-7秒)上传一段清晰的人声音频用于克隆音色。",
                                sources=["upload", "microphone"],
                            )
                            zs_tts_btn = gr.Button("克隆并生成语音", variant="primary")
                        with gr.Column(scale=1):
                            zs_tts_status = gr.Markdown(value="💡 请输入文本并上传参考音频。")
                            zs_tts_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

                # --- Tab 3: 多人播客 ---
                with gr.TabItem("播客 (Podcast)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            pod_text = gr.Textbox(
                                lines=5,
                                label="对话脚本",
                                info="使用 'speaker_1:', 'speaker_2:' 区分不同说话人。e.g. speaker_1:就比如说各种就是给别人提供，提供帮助的都可以说是服务的\n speaker_2:是的 不管是什么，就是说感觉都是，大家都，都可以说是服务业的一方面\n",
                            )
                            gr.Markdown(REFERENCE_AUDIO_WARNING)
                            pod_prompt1 = gr.Audio(
                                type="filepath",
                                label="说话人1参考音频",
                                sources=["upload", "microphone"],
                            )
                            gr.Markdown(REFERENCE_AUDIO_WARNING)
                            pod_prompt2 = gr.Audio(
                                type="filepath",
                                label="说话人2参考音频",
                                sources=["upload", "microphone"],
                            )
                            pod_btn = gr.Button("生成播客", variant="primary")
                        with gr.Column(scale=1):
                            pod_status = gr.Markdown(
                                value="💡 请填写脚本并上传两位说话人的参考音频。"
                            )
                            pod_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

                # --- Tab 4: 带背景音乐的语音 ---
                with gr.TabItem("带背景音乐的语音 (Speech with BGM)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            swb_text = gr.Textbox(label="语音文本")
                            gr.Markdown(REFERENCE_AUDIO_WARNING)
                            swb_prompt = gr.Audio(
                                type="filepath",
                                label="说话人参考音频",
                                sources=["upload", "microphone"],
                            )
                            gr.Markdown("##### 背景音乐描述")
                            with gr.Row():
                                swb_genre = gr.Dropdown(
                                    DROPDOWN_CHOICES["swb_genres"],
                                    label="风格 (Genre)",
                                    value="流行摇滚",
                                )
                                swb_mood = gr.Dropdown(
                                    DROPDOWN_CHOICES["swb_moods"],
                                    label="情绪 (Mood)",
                                    value="快乐",
                                )
                            with gr.Row():
                                swb_instrument = gr.Dropdown(
                                    DROPDOWN_CHOICES["swb_instruments"],
                                    label="乐器 (Instrument)",
                                    value="合成铜管乐器",
                                )
                                swb_theme = gr.Dropdown(
                                    DROPDOWN_CHOICES["swb_themes"],
                                    label="主题 (Theme)",
                                    value="旅行",
                                )
                            with gr.Row():
                                swb_snr = gr.Slider(
                                    0,
                                    20,
                                    value=10.0,
                                    step=0.5,
                                    label="信噪比 (SNR)",
                                    info="值越小，背景音乐音量越大。",
                                )
                            swb_btn = gr.Button("生成带BGM的语音", variant="primary")
                        with gr.Column(scale=1):
                            swb_status = gr.Markdown(value="💡 请填写所有字段并上传参考音频。")
                            swb_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

                # --- Tab 5: 纯背景音乐生成 ---
                with gr.TabItem("背景音乐生成 (BGM)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            bgm_genre = gr.Dropdown(
                                DROPDOWN_CHOICES["bgm_genres"],
                                label="风格 (Genre)",
                                value="迪斯科",
                            )
                            bgm_mood = gr.Dropdown(
                                DROPDOWN_CHOICES["bgm_moods"],
                                label="情绪 (Mood)",
                                value="快乐",
                            )
                            bgm_instrument = gr.Dropdown(
                                DROPDOWN_CHOICES["bgm_instruments"],
                                label="乐器 (Instrument)",
                                value="电吉他",
                            )
                            bgm_theme = gr.Dropdown(
                                DROPDOWN_CHOICES["bgm_themes"],
                                label="主题 (Theme)",
                                value="庆典与喜悦",
                            )
                            bgm_duration = gr.Slider(30, 60, value=35, step=1, label="时长 (秒)")
                            bgm_btn = gr.Button("生成背景音乐", variant="primary")
                        with gr.Column(scale=1):
                            bgm_status = gr.Markdown(value="💡 请描述您想要的音乐。")
                            bgm_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

                # --- Tab 6: 音效生成 ---
                with gr.TabItem("音效生成 (TTA)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            tta_text = gr.Textbox(
                                label="音效描述",
                                info="建议使用英文描述，效果更佳。例如: 'Rain is falling continuously'。",
                            )
                            tta_btn = gr.Button("生成音效", variant="primary")
                        with gr.Column(scale=1):
                            tta_status = gr.Markdown(value="💡 请输入音效的文本描述。")
                            tta_output = gr.Audio(
                                label="生成结果", type="filepath", interactive=False
                            )

            # --- 事件绑定 ---
            def i_tts_submit(
                instruct_type,
                text,
                prompt_audio,
                emotion,
                dialect,
                ip_choice,
                style,
                speed,
                pitch,
                volume,
            ):
                details = {}
                if instruct_type == "emotion":
                    details = {"情感": emotion}
                elif instruct_type == "dialect":
                    details = {"方言": dialect}
                elif instruct_type == "IP":
                    backend_ip = IP_DICT.get(ip_choice)
                    if not backend_ip:
                        raise gr.Error(f"未找到IP角色'{ip_choice}'的配置。")
                    details = {"IP": backend_ip}
                elif instruct_type == "style":
                    details = {"风格": style}
                elif instruct_type == "basic":
                    details = {"语速": speed, "基频": pitch, "音量": volume}
                yield from self._submit_and_poll("TTS", instruct_type, text, prompt_audio, details)

            i_tts_btn.click(
                fn=i_tts_submit,
                inputs=[
                    i_tts_type,
                    i_tts_text,
                    i_tts_prompt,
                    i_tts_emotion,
                    i_tts_dialect,
                    i_tts_ip,
                    i_tts_style,
                    i_tts_speed,
                    i_tts_pitch,
                    i_tts_volume,
                ],
                outputs=[i_tts_status, i_tts_btn, i_tts_output],
            )

            zs_tts_btn.click(
                fn=lambda *args: (yield from self._submit_and_poll("zero_shot_TTS", *args)),
                inputs=[zs_tts_text, zs_tts_prompt],
                outputs=[zs_tts_status, zs_tts_btn, zs_tts_output],
            )
            pod_btn.click(
                fn=lambda *args: (yield from self._submit_and_poll("podcast", *args)),
                inputs=[pod_text, pod_prompt1, pod_prompt2],
                outputs=[pod_status, pod_btn, pod_output],
            )
            swb_btn.click(
                fn=lambda *args: (yield from self._submit_and_poll("speech_with_bgm", *args)),
                inputs=[
                    swb_text,
                    swb_prompt,
                    swb_genre,
                    swb_mood,
                    swb_instrument,
                    swb_theme,
                    swb_snr,
                ],
                outputs=[swb_status, swb_btn, swb_output],
            )
            bgm_btn.click(
                fn=lambda *args: (yield from self._submit_and_poll("bgm", *args)),
                inputs=[bgm_genre, bgm_mood, bgm_instrument, bgm_theme, bgm_duration],
                outputs=[bgm_status, bgm_btn, bgm_output],
            )
            tta_btn.click(
                fn=lambda *args: (yield from self._submit_and_poll("TTA", *args)),
                inputs=[tta_text],
                outputs=[tta_status, tta_btn, tta_output],
            )

    # --- 辅助方法 ---
    def _submit_and_poll(self, task_type: str, *args):
        yield (
            gr.update(value="⏳ 正在生成..."),
            gr.update(interactive=False),
            gr.update(value=None),
        )

        try:
            temp_output_path = os.path.join(tempfile.gettempdir(), f"ming_tts_{uuid.uuid4().hex[:8]}.wav")
            if task_type == "TTS":
                instruct_type, text, prompt_audio, caption_details = args
                if not text:
                    raise ValueError("合成文本不能为空。")
                if instruct_type not in ["IP", "style"] and not prompt_audio:
                    raise ValueError(f"指令类型 '{instruct_type}' 需要上传参考音频。")

                instruction_dict = caption_details
                
                # Check zero_spk_emb
                use_zero_spk_emb = (instruct_type in ["IP", "style"])
                use_spk_emb = not use_zero_spk_emb

                self.local_service.speech_generation(
                    prompt="Please generate speech based on the following description.\n",
                    text=text,
                    use_spk_emb=use_spk_emb,
                    use_zero_spk_emb=use_zero_spk_emb,
                    instruction=instruction_dict,
                    prompt_wav_path=prompt_audio,
                    max_decode_steps=200,
                    output_wav_path=temp_output_path
                )
            elif task_type == "zero_shot_TTS":
                text, prompt_audio = args
                if not text or not prompt_audio:
                    raise ValueError("文本和参考音频不能为空。")
                self.local_service.speech_generation(
                    prompt="Please generate speech based on the following description.\n",
                    text=text,
                    use_spk_emb=True,
                    prompt_wav_path=prompt_audio,
                    max_decode_steps=200,
                    output_wav_path=temp_output_path
                )
            elif task_type == "podcast":
                text, prompt_audio_1, prompt_audio_2 = args
                if not text or not prompt_audio_1 or not prompt_audio_2:
                    raise ValueError("对话脚本和两个参考音频均不能为空。")
                    
                self.local_service.speech_generation(
                    prompt="Please generate speech based on the following description.\n",
                    text=text,
                    use_spk_emb=True,
                    prompt_wav_path=[prompt_audio_1, prompt_audio_2],
                    max_decode_steps=200,
                    output_wav_path=temp_output_path
                )
            elif task_type == "bgm":
                genre, mood, instrument, theme, duration = args
                prompt_text = f"Genre: {genre}. Mood: {mood}. Instrument: {instrument}. Theme: {theme}. Duration: {duration}s."
                self.local_service.speech_generation(
                    prompt="Please generate music based on the following description.\n",
                    text=" " + prompt_text,
                    max_decode_steps=400,
                    output_wav_path=temp_output_path
                )
            elif task_type == "TTA":
                (text,) = args
                if not text:
                    raise ValueError("音效描述不能为空。")
                self.local_service.speech_generation(
                    prompt="Please generate audio events based on given text.\n",
                    text=text,
                    max_decode_steps=200,
                    output_wav_path=temp_output_path,
                    cfg=4.5,
                    sigma=0.3,
                    temperature=2.5
                )
            elif task_type == "speech_with_bgm":
                text, prompt_audio, genre, mood, instrument, theme, snr = args
                if not text or not prompt_audio:
                    raise ValueError("文本和参考音频不能为空。")
                bgm_data = {
                    "Genre": f"{genre}.",
                    "Mood": f"{mood}.",
                    "Instrument": f"{instrument}.",
                    "Theme": f"{theme}.",
                    "SNR": float(snr) if snr else 10.0,
                    "ENV": None,
                }
                instruction_dict = {"BGM": bgm_data}
                self.local_service.speech_generation(
                    prompt="Please generate speech based on the following description.\n",
                    text=text,
                    use_spk_emb=True,
                    instruction=instruction_dict,
                    prompt_wav_path=prompt_audio,
                    max_decode_steps=200,
                    output_wav_path=temp_output_path
                )
            else:
                raise ValueError(f"未知的任务类型: {task_type}")

            # 成功后
            yield (
                gr.update(value="✅ 任务完成"),
                gr.update(interactive=True),
                gr.update(value=temp_output_path),
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            yield (
                gr.update(value=f"❌ 错误：{e}"),
                gr.update(interactive=True),
                gr.update(value=None),
            )
            return
