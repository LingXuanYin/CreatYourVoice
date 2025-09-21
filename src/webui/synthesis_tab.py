"""语音合成界面

重新设计的简化版本：
- 四种情感控制模式：普通、描述、参考、向量
- 单栏布局，响应式设计
- 简化的参数配置
- 实时状态反馈
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..core.voice_manager import VoiceManager
from ..integrations.index_tts import IndexTTSIntegration

logger = logging.getLogger(__name__)


class SynthesisTab:
    """语音合成Tab

    简化设计原则：
    1. 清晰的工作流：选择音色 → 输入文本 → 情感控制 → 生成
    2. 四种情感模式，满足不同需求
    3. 响应式单栏布局
    """

    def __init__(self, voice_manager: VoiceManager, index_tts_integration: IndexTTSIntegration):
        """初始化合成Tab"""
        self.voice_manager = voice_manager
        self.index_tts_integration = index_tts_integration

        # 当前状态
        self._current_voice_config: Optional[Any] = None
        self._synthesis_history: List[Dict[str, Any]] = []

    def create_interface(self):
        """创建界面"""
        gr.Markdown("""
        ## 🎤 语音合成

        使用IndexTTS进行文本转语音合成，支持多种情感控制模式。
        💡 **架构说明**：IndexTTS是纯文本转语音模型，无speaker概念，只有情感控制。
        """)

        # 单栏响应式布局
        with gr.Column():
            # 步骤1：上传参考音频
            with gr.Group():
                gr.Markdown("### 步骤1：上传说话人参考音频")
                gr.Markdown("💡 **IndexTTS架构**：需要上传参考音频来定义说话人特征，无需预先创建音色")

                speaker_audio = gr.Audio(
                    label="说话人参考音频",
                    type="filepath"
                )
                gr.Markdown("💡 上传包含目标说话人声音特征的音频文件")

                speaker_audio_info = gr.Textbox(
                    label="音频信息",
                    value="请上传参考音频",
                    interactive=False,
                    lines=2
                )

            # 步骤2：输入文本
            with gr.Group():
                gr.Markdown("### 步骤2：输入合成文本")

                text_input = gr.Textbox(
                    label="合成文本",
                    placeholder="请输入要合成的文本内容...",
                    lines=4,
                    max_lines=10,
                    info="支持中文、英文等多种语言"
                )

                with gr.Row():
                    text_length_display = gr.Textbox(
                        label="文本长度",
                        value="0 字符",
                        interactive=False,
                        scale=1
                    )
                    estimated_time_display = gr.Textbox(
                        label="预估时间",
                        value="0 秒",
                        interactive=False,
                        scale=1
                    )

            # 步骤3：情感控制
            with gr.Group():
                gr.Markdown("### 步骤3：情感控制")

                emotion_mode = gr.Radio(
                    label="情感控制模式",
                    choices=[
                        ("普通模式 - 使用音色默认情感", "normal"),
                        ("情感描述 - 文本描述情感", "description"),
                        ("情感参考 - 上传参考音频", "reference"),
                        ("高级模式 - 8维情感向量", "vector")
                    ],
                    value="normal",
                    info="选择适合的情感控制方式"
                )

                # 情感描述模式
                with gr.Group(visible=False) as description_group:
                    emotion_description = gr.Textbox(
                        label="情感描述",
                        placeholder="例如：开心、激动、充满活力、温柔、悲伤...",
                        lines=2,
                        info="用文字描述想要的情感表达"
                    )

                # 情感参考模式
                with gr.Group(visible=False) as reference_group:
                    emotion_reference = gr.Audio(
                        label="情感参考音频",
                        type="filepath"
                    )
                    gr.Markdown("💡 上传包含目标情感的音频文件")

                    emotion_weight = gr.Slider(
                        label="情感权重",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.65,
                        step=0.05,
                        info="参考音频情感的影响程度"
                    )

                # 高级向量模式
                with gr.Group(visible=False) as vector_group:
                    gr.Markdown("#### 8维情感向量控制 (0.0-1.0)")

                    with gr.Row():
                        emotion_happy = gr.Slider(
                            label="高兴", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )
                        emotion_angry = gr.Slider(
                            label="愤怒", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )
                        emotion_sad = gr.Slider(
                            label="悲伤", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )
                        emotion_afraid = gr.Slider(
                            label="恐惧", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )

                    with gr.Row():
                        emotion_disgusted = gr.Slider(
                            label="厌恶", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )
                        emotion_surprised = gr.Slider(
                            label="惊讶", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )
                        emotion_calm = gr.Slider(
                            label="平静", minimum=0.0, maximum=1.0, value=1.0, step=0.1
                        )
                        emotion_neutral = gr.Slider(
                            label="中性", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                        )

                    with gr.Row():
                        normalize_vector_btn = gr.Button("归一化向量", size="sm")
                        reset_vector_btn = gr.Button("重置向量", size="sm")
                        preset_happy_btn = gr.Button("预设：开心", size="sm")
                        preset_sad_btn = gr.Button("预设：悲伤", size="sm")

            # 步骤4：生成参数
            with gr.Group():
                gr.Markdown("### 步骤4：生成参数")

                with gr.Accordion("高级参数", open=False):
                    with gr.Row():
                        speed = gr.Slider(
                            label="语速",
                            minimum=0.1,
                            maximum=3.0,
                            value=1.0,
                            step=0.1,
                            info="语音播放速度"
                        )
                        temperature = gr.Slider(
                            label="温度",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            info="生成随机性"
                        )

                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            info="核采样参数"
                        )
                        top_k = gr.Slider(
                            label="Top-k",
                            minimum=1,
                            maximum=100,
                            value=30,
                            step=1,
                            info="候选词数量"
                        )

            # 步骤5：开始合成
            with gr.Group():
                gr.Markdown("### 步骤5：开始合成")

                with gr.Row():
                    synthesize_btn = gr.Button(
                        "🎤 开始合成",
                        variant="primary",
                        scale=2,
                        elem_classes=["synthesis-btn"]
                    )
                    validate_btn = gr.Button(
                        "✅ 验证参数",
                        scale=1
                    )

                # 进度和状态显示
                progress_display = gr.Textbox(
                    label="合成进度",
                    value="等待开始...",
                    interactive=False,
                    lines=2
                )

            # 结果显示
            with gr.Group():
                gr.Markdown("### 🎵 合成结果")

                with gr.Row():
                    with gr.Column(scale=2):
                        result_audio = gr.Audio(
                            label="合成音频",
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        result_info = gr.JSON(
                            label="合成信息",
                            value={}
                        )

                with gr.Row():
                    save_result_btn = gr.Button("💾 保存结果")
                    download_result_btn = gr.Button("📥 下载音频")

        # 存储组件引用
        self.components = {
            'speaker_audio': speaker_audio,
            'speaker_audio_info': speaker_audio_info,
            'text_input': text_input,
            'text_length_display': text_length_display,
            'estimated_time_display': estimated_time_display,
            'emotion_mode': emotion_mode,
            'description_group': description_group,
            'reference_group': reference_group,
            'vector_group': vector_group,
            'emotion_description': emotion_description,
            'emotion_reference': emotion_reference,
            'emotion_weight': emotion_weight,
            'emotion_happy': emotion_happy,
            'emotion_angry': emotion_angry,
            'emotion_sad': emotion_sad,
            'emotion_afraid': emotion_afraid,
            'emotion_disgusted': emotion_disgusted,
            'emotion_surprised': emotion_surprised,
            'emotion_calm': emotion_calm,
            'emotion_neutral': emotion_neutral,
            'normalize_vector_btn': normalize_vector_btn,
            'reset_vector_btn': reset_vector_btn,
            'preset_happy_btn': preset_happy_btn,
            'preset_sad_btn': preset_sad_btn,
            'speed': speed,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'synthesize_btn': synthesize_btn,
            'validate_btn': validate_btn,
            'progress_display': progress_display,
            'result_audio': result_audio,
            'result_info': result_info,
            'save_result_btn': save_result_btn,
            'download_result_btn': download_result_btn
        }

        # 绑定事件
        self._bind_events()

        # 初始化数据
        self._initialize_data()

    def _bind_events(self):
        """绑定界面事件"""
        # 参考音频上传
        self.components['speaker_audio'].change(
            fn=self._on_speaker_audio_change,
            inputs=[self.components['speaker_audio']],
            outputs=[self.components['speaker_audio_info']]
        )

        # 文本输入变化
        self.components['text_input'].change(
            fn=self._on_text_change,
            inputs=[self.components['text_input']],
            outputs=[
                self.components['text_length_display'],
                self.components['estimated_time_display']
            ]
        )

        # 情感模式切换
        self.components['emotion_mode'].change(
            fn=self._on_emotion_mode_change,
            inputs=[self.components['emotion_mode']],
            outputs=[
                self.components['description_group'],
                self.components['reference_group'],
                self.components['vector_group']
            ]
        )

        # 向量控制按钮
        self.components['normalize_vector_btn'].click(
            fn=self._normalize_emotion_vector,
            inputs=[
                self.components['emotion_happy'],
                self.components['emotion_angry'],
                self.components['emotion_sad'],
                self.components['emotion_afraid'],
                self.components['emotion_disgusted'],
                self.components['emotion_surprised'],
                self.components['emotion_calm'],
                self.components['emotion_neutral']
            ],
            outputs=[
                self.components['emotion_happy'],
                self.components['emotion_angry'],
                self.components['emotion_sad'],
                self.components['emotion_afraid'],
                self.components['emotion_disgusted'],
                self.components['emotion_surprised'],
                self.components['emotion_calm'],
                self.components['emotion_neutral']
            ]
        )

        self.components['reset_vector_btn'].click(
            fn=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            outputs=[
                self.components['emotion_happy'],
                self.components['emotion_angry'],
                self.components['emotion_sad'],
                self.components['emotion_afraid'],
                self.components['emotion_disgusted'],
                self.components['emotion_surprised'],
                self.components['emotion_calm'],
                self.components['emotion_neutral']
            ]
        )

        self.components['preset_happy_btn'].click(
            fn=lambda: [0.8, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            outputs=[
                self.components['emotion_happy'],
                self.components['emotion_angry'],
                self.components['emotion_sad'],
                self.components['emotion_afraid'],
                self.components['emotion_disgusted'],
                self.components['emotion_surprised'],
                self.components['emotion_calm'],
                self.components['emotion_neutral']
            ]
        )

        self.components['preset_sad_btn'].click(
            fn=lambda: [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
            outputs=[
                self.components['emotion_happy'],
                self.components['emotion_angry'],
                self.components['emotion_sad'],
                self.components['emotion_afraid'],
                self.components['emotion_disgusted'],
                self.components['emotion_surprised'],
                self.components['emotion_calm'],
                self.components['emotion_neutral']
            ]
        )

        # 验证参数
        self.components['validate_btn'].click(
            fn=self._validate_parameters,
            inputs=self._get_all_inputs(),
            outputs=[self.components['progress_display']]
        )

        # 开始合成
        self.components['synthesize_btn'].click(
            fn=self._synthesize_speech,
            inputs=self._get_all_inputs(),
            outputs=[
                self.components['result_audio'],
                self.components['result_info'],
                self.components['progress_display']
            ]
        )

    def _get_all_inputs(self) -> List[gr.Component]:
        """获取所有输入组件"""
        return [
            self.components['speaker_audio'],
            self.components['text_input'],
            self.components['emotion_mode'],
            self.components['emotion_description'],
            self.components['emotion_reference'],
            self.components['emotion_weight'],
            self.components['emotion_happy'],
            self.components['emotion_angry'],
            self.components['emotion_sad'],
            self.components['emotion_afraid'],
            self.components['emotion_disgusted'],
            self.components['emotion_surprised'],
            self.components['emotion_calm'],
            self.components['emotion_neutral'],
            self.components['speed'],
            self.components['temperature'],
            self.components['top_p'],
            self.components['top_k']
        ]

    def _initialize_data(self):
        """初始化数据"""
        logger.info("语音合成Tab初始化完成")

    def _on_speaker_audio_change(self, audio_path: str) -> str:
        """参考音频上传时的处理"""
        if not audio_path:
            return "请上传参考音频"

        try:
            import librosa
            from pathlib import Path

            # 获取音频信息
            audio_file = Path(audio_path)
            if not audio_file.exists():
                return "音频文件不存在"

            # 加载音频获取基本信息
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr

            info = f"✅ 音频上传成功\n"
            info += f"文件名: {audio_file.name}\n"
            info += f"时长: {duration:.2f}秒\n"
            info += f"采样率: {sr}Hz"

            return info

        except Exception as e:
            logger.error(f"处理参考音频失败: {e}")
            return f"❌ 音频处理失败: {str(e)}"

    def _on_text_change(self, text: str) -> Tuple[str, str]:
        """文本变化时的处理"""
        length = len(text) if text else 0
        # 简单的时间估算：每100字符约需要10秒
        estimated_seconds = max(5, length // 10)

        return f"{length} 字符", f"约 {estimated_seconds} 秒"

    def _on_emotion_mode_change(self, mode: str) -> Tuple[bool, bool, bool]:
        """情感模式切换"""
        return (
            mode == "description",  # description_group
            mode == "reference",    # reference_group
            mode == "vector"        # vector_group
        )

    def _normalize_emotion_vector(self, *values) -> Tuple[float, ...]:
        """归一化情感向量"""
        try:
            total = sum(values)
            if total > 0:
                normalized = [v / total for v in values]
                return tuple(normalized)
            else:
                # 如果全为0，设置为平静
                return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        except Exception:
            return values

    def _validate_parameters(self, *args) -> str:
        """验证合成参数"""
        try:
            speaker_audio, text, emotion_mode = args[0], args[1], args[2]

            errors = []

            if not speaker_audio:
                errors.append("请上传说话人参考音频")

            if not text.strip():
                errors.append("请输入合成文本")

            if emotion_mode == "description" and not args[3].strip():
                errors.append("请输入情感描述")

            if emotion_mode == "reference" and not args[4]:
                errors.append("请上传情感参考音频")

            if errors:
                return f"❌ 参数验证失败:\n" + "\n".join(f"• {error}" for error in errors)
            else:
                return "✅ 参数验证通过，可以开始合成"

        except Exception as e:
            return f"❌ 参数验证失败: {e}"

    def _synthesize_speech(self, *args) -> Tuple[Optional[str], Dict[str, Any], str]:
        """执行语音合成"""
        try:
            # 解析参数
            speaker_audio = args[0]
            text = args[1]
            emotion_mode = args[2]

            # 验证参数
            if not speaker_audio:
                return None, {"错误": "请上传说话人参考音频"}, "❌ 合成失败"

            if not text.strip():
                return None, {"错误": "请输入合成文本"}, "❌ 合成失败"

            # 构建IndexTTS合成参数
            synthesis_kwargs = {
                "text": text.strip(),
                "speaker_audio": speaker_audio,
                "emotion_control_method": emotion_mode,
                "do_sample": True,
                "top_p": args[16],
                "top_k": args[17],
                "temperature": args[15],
                "length_penalty": 0.0,
                "num_beams": 3,
                "repetition_penalty": 10.0,
                "max_mel_tokens": 1500,
            }

            # 根据情感模式添加参数
            if emotion_mode == "description":
                synthesis_kwargs["emotion_text"] = args[3]
            elif emotion_mode == "reference":
                synthesis_kwargs["emotion_audio"] = args[4]
                synthesis_kwargs["emotion_weight"] = args[5]
            elif emotion_mode == "vector":
                synthesis_kwargs["emotion_vector"] = list(args[6:14])

            # 执行IndexTTS合成
            result = self.index_tts_integration.infer(**synthesis_kwargs)

            # 构建结果信息
            result_info = {
                "状态": "合成成功",
                "文本长度": len(text),
                "情感模式": emotion_mode,
                "处理时间": f"{result.processing_time:.2f}秒",
                "分段数": result.segments_count,
                "情感信息": result.emotion_info
            }

            # 保存到历史
            self._synthesis_history.append({
                "timestamp": "2024-01-01 12:00:00",
                "text": text[:50] + "..." if len(text) > 50 else text,
                "emotion_mode": emotion_mode,
                "success": True
            })

            # 返回音频路径或数据
            if result.audio_path:
                return result.audio_path, result_info, "✅ 合成完成"
            elif result.audio_data:
                # 如果返回的是音频数据，需要保存为文件
                import tempfile
                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, result.audio_data[1], result.audio_data[0])
                    return tmp_file.name, result_info, "✅ 合成完成"
            else:
                return None, result_info, "✅ 合成完成（无音频输出）"

        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            return None, {"错误": str(e)}, "❌ 合成失败"
