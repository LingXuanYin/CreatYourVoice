"""CreatYourVoice 主应用界面

重新设计的简化版本：
- 3个Tab结构：创建角色声音基底、语音合成、音色管理
- 移除复杂的继承和融合机制
- 响应式设计，单栏布局
- 权重计算机械化
"""

import os
import sys
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import logging

from src.core.voice_manager import VoiceManager
from src.core.voice_base_creator import VoiceBaseCreator
from src.core.voice_preset_manager import VoicePresetManager
from src.integrations.ddsp_svc import DDSPSVCIntegration
from src.integrations.index_tts import IndexTTSIntegration
from src.utils.config import get_config, get_config_manager
from src.webui.voice_creation_tab import VoiceCreationTab
from src.webui.synthesis_tab import SynthesisTab
from src.webui.voice_management_tab import VoiceManagementTab
from src.webui.model_management_tab import ModelManagementTab

logger = logging.getLogger(__name__)


class CreatYourVoiceApp:
    """CreatYourVoice 主应用类

    简化设计原则：
    1. 三Tab结构，清晰的工作流
    2. 核心数据结构：{speaker_id: weight}
    3. 移除不必要的复杂性
    """

    def __init__(self):
        """初始化应用"""
        self.config = get_config()

        # 核心组件
        self.voice_manager = VoiceManager(self.config.system.voices_dir)
        self.preset_manager = VoicePresetManager()
        self.ddsp_integration = DDSPSVCIntegration()
        self.index_tts_integration = IndexTTSIntegration(
            model_dir=self.config.index_tts.model_dir
        )

        # 声音基底创建器
        self.voice_creator = VoiceBaseCreator(
            preset_manager=self.preset_manager,
            voice_manager=self.voice_manager,
            ddsp_integration=self.ddsp_integration,
            index_tts_integration=self.index_tts_integration
        )

        # 界面Tab组件
        self.voice_creation_tab = VoiceCreationTab(
            voice_creator=self.voice_creator,
            preset_manager=self.preset_manager
        )
        self.synthesis_tab = SynthesisTab(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )
        self.voice_management_tab = VoiceManagementTab(
            voice_manager=self.voice_manager
        )
        self.model_management_tab = ModelManagementTab(
            ddsp_integration=self.ddsp_integration,
            index_tts_integration=self.index_tts_integration
        )

        # 创建必要目录
        get_config_manager().create_directories()

        logger.info("CreatYourVoice应用初始化完成")

    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        with gr.Blocks(
            title="CreatYourVoice - 音色创建和语音合成系统",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:

            # 应用标题
            gr.HTML("""
            <div class="app-header">
                <h1>🎵 CreatYourVoice</h1>
                <p>基于DDSP-SVC和IndexTTS的音色创建和语音合成系统</p>
                <div class="workflow-guide">
                    <span class="step">1️⃣ 模型管理</span>
                    <span class="arrow">→</span>
                    <span class="step">2️⃣ 创建角色声音基底</span>
                    <span class="arrow">→</span>
                    <span class="step">3️⃣ 语音合成</span>
                    <span class="arrow">→</span>
                    <span class="step">4️⃣ 音色管理</span>
                </div>
            </div>
            """)

            # 四Tab结构
            with gr.Tabs() as tabs:
                # Tab 1: 模型管理
                with gr.Tab("🔧 模型管理", id="model_management"):
                    self.model_management_tab.create_interface()

                # Tab 2: 创建角色声音基底
                with gr.Tab("🎨 创建角色声音基底", id="voice_creation"):
                    self.voice_creation_tab.create_interface()

                # Tab 3: 语音合成
                with gr.Tab("🎤 语音合成", id="synthesis"):
                    self.synthesis_tab.create_interface()

                # Tab 4: 音色管理
                with gr.Tab("📁 音色管理", id="management"):
                    self.voice_management_tab.create_interface()

            # 底部信息
            gr.HTML("""
            <div class="app-footer">
                <p>💡 使用提示：先在模型管理中加载DDSP-SVC和IndexTTS模型，然后创建角色声音基底，最后进行语音合成</p>
            </div>
            """)

        return interface

    def _get_custom_css(self) -> str:
        """获取响应式CSS样式"""
        return """
        /* 响应式设计 - 移动优先 */
        .gradio-container {
            max-width: 800px !important;
            margin: 0 auto;
            padding: 10px;
        }

        /* 应用头部 */
        .app-header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
        }

        .app-header h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }

        .app-header p {
            margin: 0 0 15px 0;
            opacity: 0.9;
        }

        /* 工作流引导 */
        .workflow-guide {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }

        .step {
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            white-space: nowrap;
        }

        .arrow {
            font-size: 1.2em;
            font-weight: bold;
        }

        /* Tab样式 */
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px 8px 0 0;
        }

        /* 组件组样式 */
        .gr-group {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }

        /* 权重输入组件 */
        .weight-input {
            background: #f0f8ff;
            border: 2px solid #4a90e2;
            border-radius: 6px;
        }

        /* 归一化显示 */
        .normalized-display {
            background: #f0fff0;
            border: 1px solid #90ee90;
            border-radius: 6px;
            font-family: monospace;
        }

        /* 按钮样式 */
        .gr-button {
            border-radius: 6px;
            font-weight: 500;
        }

        .gr-button.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
        }

        /* 底部信息 */
        .app-footer {
            text-align: center;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            color: #666;
        }

        /* 响应式断点 */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 5px;
            }

            .app-header h1 {
                font-size: 2em;
            }

            .workflow-guide {
                flex-direction: column;
            }

            .arrow {
                transform: rotate(90deg);
            }
        }

        @media (max-width: 480px) {
            .app-header {
                padding: 15px;
            }

            .app-header h1 {
                font-size: 1.8em;
            }

            .step {
                padding: 6px 12px;
                font-size: 0.9em;
            }
        }

        /* 触摸友好 */
        @media (hover: none) and (pointer: coarse) {
            .gr-button {
                min-height: 44px;
                min-width: 44px;
            }

            .gr-slider input {
                min-height: 44px;
            }
        }
        """


def create_app() -> gr.Blocks:
    """创建应用实例"""
    app = CreatYourVoiceApp()
    return app.create_interface()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="CreatYourVoice Web界面")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建并启动应用
    interface = create_app()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
