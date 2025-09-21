
"""模型管理界面

手动模型管理功能：
- DDSP-SVC模型：加载/卸载、显示speaker列表、权重配置
- IndexTTS模型：加载/卸载、情感控制参数配置
- 模型状态显示和内存使用情况
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from ..integrations.ddsp_svc import DDSPSVCIntegration
from ..integrations.index_tts import IndexTTSIntegration

logger = logging.getLogger(__name__)


class ModelManagementTab:
    """模型管理Tab

    设计原则：
    1. 分离DDSP-SVC和IndexTTS模型管理
    2. 清晰的加载/卸载状态显示
    3. 动态speaker列表获取（仅DDSP-SVC）
    4. 实时内存使用监控
    """

    def __init__(self, ddsp_integration: DDSPSVCIntegration, index_tts_integration: IndexTTSIntegration):
        """初始化模型管理Tab"""
        self.ddsp_integration = ddsp_integration
        self.index_tts_integration = index_tts_integration

        # 当前状态
        self._ddsp_speakers: List[Dict[str, Any]] = []
        self._ddsp_model_loaded = False
        self._index_tts_model_loaded = False

    def create_interface(self):
        """创建界面"""
        gr.Markdown("""
        ## 🔧 模型管理

        手动管理DDSP-SVC和IndexTTS模型的加载、卸载和配置。
        """)

        with gr.Tabs():
            # DDSP-SVC模型管理
            with gr.Tab("🎵 DDSP-SVC模型管理"):
                self._create_ddsp_interface()

            # IndexTTS模型管理
            with gr.Tab("🗣️ IndexTTS模型管理"):
                self._create_index_tts_interface()

            # 系统状态监控
            with gr.Tab("📊 系统状态"):
                self._create_system_status_interface()

    def _create_ddsp_interface(self):
        """创建DDSP-SVC管理界面"""
        gr.Markdown("### DDSP-SVC模型管理")
        gr.Markdown("💡 **架构说明**：一个DDSP-SVC模型包含多个speaker，需要先加载模型才能获取speaker列表")

        with gr.Column():
            # 模型文件选择和加载
            with gr.Group():
                gr.Markdown("#### 步骤1：选择和加载模型")

                ddsp_model_path = gr.Textbox(
                    label="DDSP-SVC模型路径",
                    placeholder="请输入.pth模型文件路径",
                    info="支持DDSP-SVC 6.1和6.3版本"
                )

                ddsp_model_file = gr.File(
                    label="或上传模型文件",
                    file_types=[".pth"],
                    file_count="single"
                )

                with gr.Row():
                    ddsp_load_btn = gr.Button("🔄 加载模型", variant="primary", scale=2)
                    ddsp_unload_btn = gr.Button("❌ 卸载模型", scale=1)
                    ddsp_refresh_btn = gr.Button("🔍 检查状态", scale=1)

            # 模型状态显示
            with gr.Group():
                gr.Markdown("#### 模型状态")

                ddsp_status_display = gr.Textbox(
                    label="加载状态",
                    value="未加载",
                    interactive=False,
                    lines=3
                )

                ddsp_model_info = gr.JSON(
                    label="模型信息",
                    value={}
                )

            # Speaker列表和权重配置
            with gr.Group():
                gr.Markdown("#### Speaker管理（模型加载后可用）")

                ddsp_speakers_display = gr.Dataframe(
                    headers=["Speaker ID", "Speaker Name", "权重"],
                    datatype=["str", "str", "number"],
                    value=[],
                    label="可用Speaker列表",
                    interactive=True,
                    wrap=True
                )

                with gr.Row():
                    ddsp_refresh_speakers_btn = gr.Button("🔄 刷新Speaker列表")
                    ddsp_normalize_weights_btn = gr.Button("⚖️ 归一化权重")

                ddsp_weight_result = gr.JSON(
                    label="归一化权重结果",
                    value={}
                )

        # 存储DDSP组件引用
        self.ddsp_components = {
            'model_path': ddsp_model_path,
            'model_file': ddsp_model_file,
            'load_btn': ddsp_load_btn,
            'unload_btn': ddsp_unload_btn,
            'refresh_btn': ddsp_refresh_btn,
            'status_display': ddsp_status_display,
            'model_info': ddsp_model_info,
            'speakers_display': ddsp_speakers_display,
            'refresh_speakers_btn': ddsp_refresh_speakers_btn,
            'normalize_weights_btn': ddsp_normalize_weights_btn,
            'weight_result': ddsp_weight_result
        }

        # 绑定DDSP事件
        self._bind_ddsp_events()

    def _create_index_tts_interface(self):
        """创建IndexTTS管理界面"""
        gr.Markdown("### IndexTTS模型管理")
        gr.Markdown("💡 **架构说明**：IndexTTS是纯文本转语音模型，无speaker概念，支持情感控制")

        with gr.Column():
            # 模型目录选择和加载
            with gr.Group():
                gr.Markdown("#### 步骤1：选择和加载模型")

                index_tts_model_dir = gr.Textbox(
                    label="IndexTTS模型目录",
                    placeholder="请输入包含checkpoints的目录路径",
                    info="目录应包含config.yaml, gpt.pth等文件"
                )

                with gr.Row():
                    index_tts_load_btn = gr.Button("🔄 加载模型", variant="primary", scale=2)
                    index_tts_unload_btn = gr.Button("❌ 卸载模型", scale=1)
                    index_tts_refresh_btn = gr.Button("🔍 检查状态", scale=1)

            # 模型状态显示
            with gr.Group():
                gr.Markdown("#### 模型状态")

                index_tts_status_display = gr.Textbox(
                    label="加载状态",
                    value="未加载",
                    interactive=False,
                    lines=3
                )

                index_tts_model_info = gr.JSON(
                    label="模型信息",
                    value={}
                )

            # 情感控制参数配置
            with gr.Group():
                gr.Markdown("#### 情感控制参数（模型加载后可用）")

                with gr.Accordion("默认情感参数", open=False):
                    with gr.Row():
                        default_emotion_weight = gr.Slider(
                            label="默认情感权重",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.65,
                            step=0.05
                        )
                        default_temperature = gr.Slider(
                            label="默认温度",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1
                        )

                    with gr.Row():
                        default_top_p = gr.Slider(
                            label="默认Top-p",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.05
                        )
                        default_top_k = gr.Slider(
                            label="默认Top-k",
                            minimum=1,
                            maximum=100,
                            value=30,
                            step=1
                        )

                index_tts_test_btn = gr.Button("🧪 测试情感分析")
                index_tts_test_result = gr.Textbox(
                    label="测试结果",
                    interactive=False,
                    lines=2
                )

        # 存储IndexTTS组件引用
        self.index_tts_components = {
            'model_dir': index_tts_model_dir,
            'load_btn': index_tts_load_btn,
            'unload_btn': index_tts_unload_btn,
            'refresh_btn': index_tts_refresh_btn,
            'status_display': index_tts_status_display,
            'model_info': index_tts_model_info,
            'default_emotion_weight': default_emotion_weight,
            'default_temperature': default_temperature,
            'default_top_p': default_top_p,
            'default_top_k': default_top_k,
            'test_btn': index_tts_test_btn,
            'test_result': index_tts_test_result
        }

        # 绑定IndexTTS事件
        self._bind_index_tts_events()

    def _create_system_status_interface(self):
        """创建系统状态监控界面"""
        gr.Markdown("### 系统状态监控")

        with gr.Column():
            with gr.Group():
                gr.Markdown("#### 内存使用情况")

                memory_info = gr.JSON(
                    label="内存状态",
                    value={}
                )

                with gr.Row():
                    refresh_memory_btn = gr.Button("🔄 刷新内存信息")
                    clear_cache_btn = gr.Button("🧹 清理缓存")

                cache_status = gr.Textbox(
                    label="缓存清理状态",
                    interactive=False,
                    lines=2
                )

            with gr.Group():
                gr.Markdown("#### 模型概览")

                models_overview = gr.Dataframe(
                    headers=["模型类型", "状态", "内存使用", "加载时间"],
                    datatype=["str", "str", "str", "str"],
                    value=[
                        ["DDSP-SVC", "未加载", "0 MB", "-"],
                        ["IndexTTS", "未加载", "0 MB", "-"]
                    ],
                    label="模型状态概览",
                    interactive=False
                )

        # 存储系统状态组件引用
        self.system_components = {
            'memory_info': memory_info,
            'refresh_memory_btn': refresh_memory_btn,
            'clear_cache_btn': clear_cache_btn,
            'cache_status': cache_status,
            'models_overview': models_overview
        }

        # 绑定系统状态事件
        self._bind_system_events()

    def _bind_ddsp_events(self):
        """绑定DDSP-SVC事件"""
        # 加载模型
        self.ddsp_components['load_btn'].click(
            fn=self._load_ddsp_model,
            inputs=[
                self.ddsp_components['model_path'],
                self.ddsp_components['model_file']
            ],
            outputs=[
                self.ddsp_components['status_display'],
                self.ddsp_components['model_info'],
                self.ddsp_components['speakers_display']
            ]
        )

        # 卸载模型
        self.ddsp_components['unload_btn'].click(
            fn=self._unload_ddsp_model,
            outputs=[
                self.ddsp_components['status_display'],
                self.ddsp_components['model_info'],
                self.ddsp_components['speakers_display']
            ]
        )

        # 刷新状态
        self.ddsp_components['refresh_btn'].click(
            fn=self._refresh_ddsp_status,
            outputs=[
                self.ddsp_components['status_display'],
                self.ddsp_components['model_info']
            ]
        )

        # 刷新speaker列表
        self.ddsp_components['refresh_speakers_btn'].click(
            fn=self._refresh_ddsp_speakers,
            outputs=[self.ddsp_components['speakers_display']]
        )

        # 归一化权重
        self.ddsp_components['normalize_weights_btn'].click(
            fn=self._normalize_ddsp_weights,
            inputs=[self.ddsp_components['speakers_display']],
            outputs=[self.ddsp_components['weight_result']]
        )

    def _bind_index_tts_events(self):
        """绑定IndexTTS事件"""
        # 加载模型
        self.index_tts_components['load_btn'].click(
            fn=self._load_index_tts_model,
            inputs=[self.index_tts_components['model_dir']],
            outputs=[
                self.index_tts_components['status_display'],
                self.index_tts_components['model_info']
            ]
        )

        # 卸载模型
        self.index_tts_components['unload_btn'].click(
            fn=self._unload_index_tts_model,
            outputs=[
                self.index_tts_components['status_display'],
                self.index_tts_components['model_info']
            ]
        )

        # 刷新状态
        self.index_tts_components['refresh_btn'].click(
            fn=self._refresh_index_tts_status,
            outputs=[
                self.index_tts_components['status_display'],
                self.index_tts_components['model_info']
            ]
        )

        # 测试情感分析
        self.index_tts_components['test_btn'].click(
            fn=self._test_emotion_analysis,
            outputs=[self.index_tts_components['test_result']]
        )

    def _bind_system_events(self):
        """绑定系统状态事件"""
        # 刷新内存信息
        self.system_components['refresh_memory_btn'].click(
            fn=self._refresh_memory_info,
            outputs=[self.system_components['memory_info']]
        )

        # 清理缓存
        self.system_components['clear_cache_btn'].click(
            fn=self._clear_all_cache,
            outputs=[self.system_components['cache_status']]
        )

    def _load_ddsp_model(self, model_path: str, model_file) -> Tuple[str, Dict[str, Any], List[List[str]]]:
        """加载DDSP-SVC模型"""
        try:
            # 确定模型路径
            if model_file is not None:
                actual_path = model_file.name
            elif model_path.strip():
                actual_path = model_path.strip()
            else:
                return "❌ 错误：请提供模型路径或上传模型文件", {}, []

            # 检查文件存在
            if not Path(actual_path).exists():
                return f"❌ 错误：模型文件不存在: {actual_path}", {}, []

            # 加载模型
            self.ddsp_integration.load_model(actual_path)

            # 获取模型信息
            model_info = self.ddsp_integration.get_model_info() or {}

            # 获取speaker列表
            speakers = self.ddsp_integration.get_available_speakers()
            self._ddsp_speakers = speakers
            self._ddsp_model_loaded = True

            # 构建speaker显示数据
            speaker_data = []
            for speaker in speakers:
                speaker_data.append([
                    str(speaker["id"]),
                    speaker["name"],
                    1000.0  # 默认权重
                ])

            status = f"✅ DDSP-SVC模型加载成功\n"
            status += f"模型路径: {actual_path}\n"
            status += f"检测到 {len(speakers)} 个speaker"

            return status, model_info, speaker_data

        except Exception as e:
            logger.error(f"加载DDSP-SVC模型失败: {e}")
            return f"❌ 加载失败: {str(e)}", {}, []

    def _unload_ddsp_model(self) -> Tuple[str, Dict[str, Any], List[List[str]]]:
        """卸载DDSP-SVC模型"""
        try:
            self.ddsp_integration.unload_model()
            self._ddsp_speakers = []
            self._ddsp_model_loaded = False

            return "✅ DDSP-SVC模型已卸载", {}, []

        except Exception as e:
            logger.error(f"卸载DDSP-SVC模型失败: {e}")
            return f"❌ 卸载失败: {str(e)}", {}, []

    def _refresh_ddsp_status(self) -> Tuple[str, Dict[str, Any]]:
        """刷新DDSP-SVC状态"""
        try:
            is_loaded = self.ddsp_integration.is_model_loaded()

            if is_loaded:
                model_info = self.ddsp_integration.get_model_info() or {}
                speakers = self.ddsp_integration.get_available_speakers()

                status = f"✅ DDSP-SVC模型已加载\n"
                status += f"Speaker数量: {len(speakers)}\n"
                status += f"设备: {model_info.get('device', '未知')}"

                return status, model_info
            else:
                return "❌ DDSP-SVC模型未加载", {}

        except Exception as e:
            logger.error(f"刷新DDSP-SVC状态失败: {e}")
            return f"❌ 状态检查失败: {str(e)}", {}

    def _refresh_ddsp_speakers(self) -> List[List[str]]:
        """刷新DDSP-SVC speaker列表"""
        try:
            if not self._ddsp_model_loaded:
                return []

            speakers = self.ddsp_integration.get_available_speakers()
            self._ddsp_speakers = speakers

            speaker_data = []
            for speaker in speakers:
                speaker_data.append([
                    str(speaker["id"]),
                    speaker["name"],
                    1000.0  # 默认权重
                ])

            return speaker_data

        except Exception as e:
            logger.error(f"刷新speaker列表失败: {e}")
            return []

    def _normalize_ddsp_weights(self, speaker_data: List[List[str]]) -> Dict[str, Any]:
        """归一化DDSP-SVC权重"""
        try:
            if not speaker_data:
                return {"错误": "没有speaker数据"}

            # 解析权重
            weights = {}
            for row in speaker_data:
                if len(row) >= 3:
                    speaker_id = row[0]
                    speaker_name = row[1]
                    weight = float(row[2]) if row[2] else 0.0

                    if weight > 0:
                        weights[f"{speaker_name} ({speaker_id})"] = weight

            if not weights:
                return {"错误": "没有有效的权重值"}

            # 归一化
            total = sum(weights.values())
            if total > 0:
                normalized = {k: v / total for k, v in weights.items()}

                # 格式化显示
                result = {}
                for name, weight in normalized.items():
                    result[name] = f"{weight:.3f} ({weight*100:.1f}%)"

                return result

            return {"错误": "权重总和为0"}

        except Exception as e:
            logger.error(f"归一化权重失败: {e}")
            return {"错误": str(e)}

    def _load_index_tts_model(self, model_dir: str) -> Tuple[str, Dict[str, Any]]:
        """加载IndexTTS模型"""
        try:
            if not model_dir.strip():
                return "❌ 错误：请提供模型目录路径", {}

            # 检查目录存在
            if not Path(model_dir).exists():
                return f"❌ 错误：模型目录不存在: {model_dir}", {}

            # 设置模型目录
            self.index_tts_integration.model_dir = Path(model_dir)
            self.index_tts_integration.config_path = Path(model_dir) / "config.yaml"

            # 加载模型
            self.index_tts_integration.load_model()

            # 获取模型信息
            model_info = self.index_tts_integration.get_model_info() or {}
            self._index_tts_model_loaded = True

            status = f"✅ IndexTTS模型加载成功\n"
            status += f"模型目录: {model_dir}\n"
            status += f"设备: {model_info.get('device', '未知')}"

            return status, model_info

        except Exception as e:
            logger.error(f"加载IndexTTS模型失败: {e}")
            return f"❌ 加载失败: {str(e)}", {}

    def _unload_index_tts_model(self) -> Tuple[str, Dict[str, Any]]:
        """卸载IndexTTS模型"""
        try:
            self.index_tts_integration.clear_cache()
            self._index_tts_model_loaded = False

            return "✅ IndexTTS模型已卸载", {}

        except Exception as e:
            logger.error(f"卸载IndexTTS模型失败: {e}")
            return f"❌ 卸载失败: {str(e)}", {}

    def _refresh_index_tts_status(self) -> Tuple[str, Dict[str, Any]]:
        """刷新IndexTTS状态"""
        try:
            if self._index_tts_model_loaded:
                model_info = self.index_tts_integration.get_model_info() or {}

                status = f"✅ IndexTTS模型已加载\n"
                status += f"模型目录: {model_info.get('model_dir', '未知')}\n"
                status += f"设备: {model_info.get('device', '未知')}"

                return status, model_info
            else:
                return "❌ IndexTTS模型未加载", {}

        except Exception as e:
            logger.error(f"刷新IndexTTS状态失败: {e}")
            return f"❌ 状态检查失败: {str(e)}", {}

    def _test_emotion_analysis(self) -> str:
        """测试情感分析功能"""
        try:
            if not self._index_tts_model_loaded:
                return "❌ 请先加载IndexTTS模型"

            test_text = "今天天气真好，我感到很开心！"

            # 测试情感分析
            emotion_result = self.index_tts_integration.analyze_emotion_from_text(test_text)

            result = f"✅ 情感分析测试成功\n"
            result += f"测试文本: {test_text}\n"
            result += f"分析结果: {emotion_result}"

            return result

        except Exception as e:
            logger.error(f"情感分析测试失败: {e}")
            return f"❌ 测试失败: {str(e)}"

    def _refresh_memory_info(self) -> Dict[str, Any]:
        """刷新内存信息"""
        try:
            import psutil
            import torch

            # 系统内存
            memory = psutil.virtual_memory()

            memory_info = {
                "系统内存": {
                    "总计": f"{memory.total / 1024**3:.1f} GB",
                    "已用": f"{memory.used / 1024**3:.1f} GB",
                    "可用": f"{memory.available / 1024**3:.1f} GB",
                    "使用率": f"{memory.percent:.1f}%"
                }
            }

            # GPU内存（如果可用）
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0)
                gpu_used = torch.cuda.memory_allocated(0)
                gpu_cached = torch.cuda.memory_reserved(0)

                memory_info["GPU内存"] = {
                    "总计": f"{gpu_memory.total_memory / 1024**3:.1f} GB",
                    "已分配": f"{gpu_used / 1024**3:.1f} GB",
                    "已缓存": f"{gpu_cached / 1024**3:.1f} GB",
                    "设备名": gpu_memory.name
                }

            return memory_info

        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {"错误": str(e)}

    def _clear_all_cache(self) -> str:
        """清理所有缓存"""
        try:
            status_messages = []

            # 清理DDSP-SVC缓存
            try:
                self.ddsp_integration.clear_cache()
                status_messages.append("✅ DDSP-SVC缓存已清理")
            except Exception as e:
                status_messages.append(f"❌ DDSP-SVC缓存清理失败: {e}")

            # 清理IndexTTS缓存
            try:
                self.index_tts_integration.clear_cache()
                status_messages.append("✅ IndexTTS缓存已清理")
            except Exception as e:
                status_messages.append(f"❌ IndexTTS缓存清理失败: {e}")

            # 清理GPU缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    status_messages.append("✅ GPU缓存已清理")
            except Exception as e:
                status_messages.append(f"❌ GPU缓存清理失败: {e}")

            return "\n".join(status_messages)

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return f"❌ 缓存清理失败: {str(e)}"
