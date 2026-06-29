# -*- coding: utf-8 -*-
import base64
import os

# 在导入 gradio 之前禁用其遥测/分析上报
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
from loguru import logger
from tab_uniaudio_demo import MingOmniTTSDemoTab
from local_speech_service import LocalSpeechService

# Gradio界面构建 =======================================================
class GradioInterface:
    def __init__(self, local_service: LocalSpeechService):
        self.service = local_service

        # 初始化 UniAudio V4 MOE 演示 Tab
        self.uniaudio_demo_tab = MingOmniTTSDemoTab(
            local_service=self.service
        )

        self.custom_css = """
            .equal-height-group {
                height: 100%;
                min-height: 400px;          /* 最小高度 */
                border: 1px solid #e0e0e0;  /* 扁平化边框 */
                border-radius: 4px;         /* 轻微圆角 */
                padding: 16px;
                background-color: transparent; /* 适配深色模式 */
                box-shadow: none;           /* 去掉阴影，保持扁平风格 */
                display: flex;
                flex-direction: column;
                justify-content: space-between; /* 整齐分布 */
                gap: 10px;
            }
            .audio-md {
                background: transparent !important; /* 适配深色模式 */
                border: unset !important;
                padding-bottom: 10px;
            }
            input, textarea {
                font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, Courier, monospace !important;
            }
            """
        self.demo = self._create_interface()

    def _create_interface(self) -> gr.Blocks:
        """构建Gradio界面"""

        theme = gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
            font=["PingFang SC", "SF Pro", "Microsoft YaHei", "Segoe UI", "sans-serif"],
        )
        with gr.Blocks(
            title="tt 演示",
            analytics_enabled=False,
            css=self.custom_css,
            theme=theme,
            fill_width=True,
        ) as demo:

            with gr.Row(variant="panel"):
                gr.Markdown("### 模型加载")
                with gr.Row():
                    model_path_input = gr.Dropdown(
                        choices=["inclusionAI/Ming-omni-tts-0.5B", "./finetuned_Ming_0.5B"],
                        value="inclusionAI/Ming-omni-tts-0.5B",
                        label="选择或输入模型路径",
                        allow_custom_value=True,
                        interactive=True
                    )
                    load_status = gr.Textbox(
                        label="模型加载状态",
                        interactive=False,
                        value=self.service.get_status()
                    )
                    load_btn = gr.Button("重新加载模型", variant="secondary", scale=0)

                    def _do_load(path):
                        _, msg = self.service.load_model(path)
                        return msg

                    load_btn.click(
                        fn=lambda path: "⏳ 正在加载模型中，请稍候...",
                        inputs=[model_path_input],
                        outputs=[load_status]
                    ).then(
                        fn=_do_load,
                        inputs=[model_path_input],
                        outputs=[load_status]
                    )

            with gr.Tabs():
                # 引入 UniAudio V4 MOE 综合演示标签页
                self.uniaudio_demo_tab.create_tab()

        return demo

    def launch(self):
        """启动Gradio应用"""
        server_name = os.getenv("GRADIO_APP_HOST", "127.0.0.1")
        server_port = int(os.getenv("GRADIO_APP_PORT", "7860"))
        self.demo.launch(share=False, server_name=server_name, server_port=server_port)


# 主程序 ==============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="inclusionAI/Ming-omni-tts-0.5B", help="模型路径或HuggingFace ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # 同步加载模型（启动时完成，避免用户在未就绪时操作）
    local_service = LocalSpeechService()
    logger.info(f"正在加载模型: {args.model}")
    ok, msg = local_service.load_model(args.model)
    if not ok:
        logger.error(f"模型加载失败: {msg}")
        logger.error("请检查模型路径或网络连接后重试")
        exit(1)
    logger.info(msg)

    # 创建并启动Gradio界面
    gradio_interface = GradioInterface(local_service)
    gradio_interface.demo.queue(default_concurrency_limit=10).launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )
