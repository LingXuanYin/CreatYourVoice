"""角色声音基底创建界面

重新设计的简化版本：
- 三种创建模式：从零开始、从现有产物、融合现有产物
- 权重计算机械化：用户输入任意数字，系统自动归一化
- 单栏引导式设计
- 响应式布局
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

from ..core.voice_base_creator import VoiceBaseCreator, VoiceBaseCreationParams
from ..core.voice_preset_manager import VoicePresetManager
from ..core.models import VoiceConfig

logger = logging.getLogger(__name__)


class VoiceCreationTab:
    """角色声音基底创建Tab

    简化设计原则：
    1. 三种创建模式，清晰的工作流
    2. 权重输入支持任意数字，自动归一化
    3. 实时预览和反馈
    """

    def __init__(self, voice_creator: VoiceBaseCreator, preset_manager: VoicePresetManager):
        """初始化创建Tab"""
        self.voice_creator = voice_creator
        self.preset_manager = preset_manager

        # 当前状态
        self._current_speakers: List[Dict[str, Any]] = []
        self._current_result: Optional[Any] = None

    def create_interface(self):
        """创建界面"""
        gr.Markdown("""
        ## 🎨 创建角色声音基底

        选择创建模式，配置参数，生成个性化的角色声音基底。
        """)

        # 创建模式选择
        with gr.Group():
            gr.Markdown("### 步骤1：选择创建模式")

            creation_mode = gr.Radio(
                label="创建模式",
                choices=[
                    ("从零开始创建", "from_scratch"),
                    ("从现有产物创建", "from_existing"),
                    ("融合现有产物", "merge_existing")
                ],
                value="from_scratch",
                info="选择适合的创建方式"
            )

        # 响应式布局：单栏设计
        with gr.Column():
            # 基本信息
            with gr.Group():
                gr.Markdown("### 步骤2：基本信息")

                voice_name = gr.Textbox(
                    label="角色名称",
                    placeholder="请输入角色名称",
                    elem_classes=["voice-name-input"]
                )

                voice_description = gr.Textbox(
                    label="角色描述",
                    placeholder="描述角色特征（可选）",
                    lines=2
                )

            # 从零开始创建的界面
            with gr.Group(visible=True) as scratch_group:
                gr.Markdown("### 步骤3：加载DDSP-SVC模型")
                gr.Markdown("💡 **架构说明**：需要先加载DDSP-SVC模型才能获取其包含的speaker列表")

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
                    load_ddsp_btn = gr.Button("🔄 加载模型", variant="primary", scale=2)
                    ddsp_status_btn = gr.Button("🔍 检查状态", scale=1)

                ddsp_model_status = gr.Textbox(
                    label="模型状态",
                    value="未加载模型",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("### 步骤4：DDSP-SVC参数")

                with gr.Row():
                    pitch_shift = gr.Slider(
                        label="音调偏移（半音）",
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=0.1,
                        info="正值升高音调，负值降低音调"
                    )

                    formant_shift = gr.Slider(
                        label="共振峰偏移（半音）",
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=0.1,
                        info="调整音色特征"
                    )

                gr.Markdown("### 步骤5：说话人权重配置")
                gr.Markdown("💡 **权重输入说明**：可以输入任意数字（如1000, 4000），系统会自动归一化为权重比例")

                # 动态说话人权重输入
                speaker_components = []
                for i in range(5):  # 最多5个说话人
                    with gr.Row(visible=False) as speaker_row:
                        speaker_name = gr.Textbox(
                            label=f"说话人{i+1}",
                            interactive=False,
                            scale=2
                        )
                        speaker_weight = gr.Number(
                            label="权重值",
                            value=1000,
                            minimum=0,
                            info="任意正数",
                            elem_classes=["weight-input"],
                            scale=1
                        )
                        speaker_enabled = gr.Checkbox(
                            label="启用",
                            value=False,
                            scale=0
                        )
                    speaker_components.append((speaker_row, speaker_name, speaker_weight, speaker_enabled))

                # 权重归一化显示
                normalized_weights_display = gr.JSON(
                    label="归一化权重（实时计算）",
                    value={},
                    elem_classes=["normalized-display"]
                )

                calculate_weights_btn = gr.Button("🧮 计算权重", variant="secondary")

            # 从现有产物创建的界面
            with gr.Group(visible=False) as existing_group:
                gr.Markdown("### 步骤3：选择现有产物")

                existing_voice_dropdown = gr.Dropdown(
                    label="选择现有音色",
                    choices=[],
                    info="基于现有音色进行修改"
                )

                existing_weight = gr.Slider(
                    label="现有产物权重",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    info="现有产物在新音色中的权重比例"
                )

                gr.Markdown("### 步骤4：添加新参数")
                # 这里可以添加新的DDSP参数调整

            # 融合现有产物的界面
            with gr.Group(visible=False) as merge_group:
                gr.Markdown("### 步骤3：选择要融合的音色")

                # 多个音色选择和权重
                merge_components = []
                for i in range(3):  # 最多融合3个音色
                    with gr.Row():
                        merge_voice = gr.Dropdown(
                            label=f"音色{i+1}",
                            choices=[],
                            scale=2
                        )
                        merge_weight = gr.Number(
                            label="权重值",
                            value=1000 if i == 0 else 0,
                            minimum=0,
                            info="任意正数",
                            elem_classes=["weight-input"],
                            scale=1
                        )
                    merge_components.append((merge_voice, merge_weight))

                merge_weights_display = gr.JSON(
                    label="融合权重（实时计算）",
                    value={},
                    elem_classes=["normalized-display"]
                )

            # 预览和保存
            with gr.Group():
                gr.Markdown("### 步骤6：预览和保存")

                preview_text = gr.Textbox(
                    label="预览文本",
                    value="你好，我是一个新的音色角色。",
                    lines=2,
                    info="用于生成预览音频的文本"
                )

                with gr.Row():
                    preview_btn = gr.Button("🎵 生成预览", variant="primary", scale=2)
                    save_btn = gr.Button("💾 保存音色", variant="secondary", scale=1)

                # 状态和结果显示
                status_display = gr.Textbox(
                    label="状态",
                    value="等待操作...",
                    interactive=False,
                    lines=3
                )

                preview_audio = gr.Audio(
                    label="预览音频",
                    interactive=False
                )

                result_info = gr.JSON(
                    label="生成信息",
                    value={}
                )

        # 存储组件引用
        self.components = {
            'creation_mode': creation_mode,
            'voice_name': voice_name,
            'voice_description': voice_description,
            'scratch_group': scratch_group,
            'existing_group': existing_group,
            'merge_group': merge_group,
            'ddsp_model_path': ddsp_model_path,
            'ddsp_model_file': ddsp_model_file,
            'load_ddsp_btn': load_ddsp_btn,
            'ddsp_status_btn': ddsp_status_btn,
            'ddsp_model_status': ddsp_model_status,
            'pitch_shift': pitch_shift,
            'formant_shift': formant_shift,
            'speaker_components': speaker_components,
            'normalized_weights_display': normalized_weights_display,
            'calculate_weights_btn': calculate_weights_btn,
            'existing_voice_dropdown': existing_voice_dropdown,
            'existing_weight': existing_weight,
            'merge_components': merge_components,
            'merge_weights_display': merge_weights_display,
            'preview_text': preview_text,
            'preview_btn': preview_btn,
            'save_btn': save_btn,
            'status_display': status_display,
            'preview_audio': preview_audio,
            'result_info': result_info
        }

        # 绑定事件
        self._bind_events()

        # 初始化数据
        self._initialize_data()

    def _bind_events(self):
        """绑定界面事件"""
        # 创建模式切换
        self.components['creation_mode'].change(
            fn=self._on_mode_change,
            inputs=[self.components['creation_mode']],
            outputs=[
                self.components['scratch_group'],
                self.components['existing_group'],
                self.components['merge_group']
            ]
        )

        # 加载DDSP-SVC模型
        self.components['load_ddsp_btn'].click(
            fn=self._load_ddsp_model,
            inputs=[
                self.components['ddsp_model_path'],
                self.components['ddsp_model_file']
            ],
            outputs=[self.components['ddsp_model_status']] +
                    [comp[0] for comp in self.components['speaker_components']] +
                    [comp[1] for comp in self.components['speaker_components']]
        )

        # 检查DDSP-SVC状态
        self.components['ddsp_status_btn'].click(
            fn=self._check_ddsp_status,
            outputs=[self.components['ddsp_model_status']]
        )

        # 权重计算（实时）
        weight_inputs = []
        for _, _, weight, enabled in self.components['speaker_components']:
            weight_inputs.extend([weight, enabled])

        for weight, enabled in [(comp[2], comp[3]) for comp in self.components['speaker_components']]:
            weight.change(
                fn=self._calculate_weights_realtime,
                inputs=weight_inputs,
                outputs=[self.components['normalized_weights_display']]
            )
            enabled.change(
                fn=self._calculate_weights_realtime,
                inputs=weight_inputs,
                outputs=[self.components['normalized_weights_display']]
            )

        # 手动计算权重
        self.components['calculate_weights_btn'].click(
            fn=self._calculate_weights_realtime,
            inputs=weight_inputs,
            outputs=[self.components['normalized_weights_display']]
        )

        # 预览生成
        self.components['preview_btn'].click(
            fn=self._generate_preview,
            inputs=self._get_all_inputs(),
            outputs=[
                self.components['preview_audio'],
                self.components['status_display'],
                self.components['result_info']
            ]
        )

        # 保存音色
        self.components['save_btn'].click(
            fn=self._save_voice,
            inputs=[self.components['voice_name']],
            outputs=[self.components['status_display']]
        )

    def _get_all_inputs(self) -> List[gr.Component]:
        """获取所有输入组件"""
        inputs = [
            self.components['creation_mode'],
            self.components['voice_name'],
            self.components['voice_description'],
            self.components['pitch_shift'],
            self.components['formant_shift'],
            self.components['preview_text']
        ]

        # 添加说话人权重输入
        for _, _, weight, enabled in self.components['speaker_components']:
            inputs.extend([weight, enabled])

        return inputs

    def _initialize_data(self):
        """初始化数据"""
        # 初始化时不需要加载模型，等待用户手动加载
        logger.info("声音创建Tab初始化完成")

    def _on_mode_change(self, mode: str) -> Tuple[bool, bool, bool]:
        """创建模式切换"""
        return (
            mode == "from_scratch",    # scratch_group
            mode == "from_existing",   # existing_group
            mode == "merge_existing"   # merge_group
        )

    def _load_ddsp_model(self, model_path: str, model_file) -> Tuple[str, ...]:
        """加载DDSP-SVC模型并获取speaker列表"""
        try:
            # 确定模型路径
            if model_file is not None:
                actual_path = model_file.name
            elif model_path.strip():
                actual_path = model_path.strip()
            else:
                return (
                    "❌ 错误：请提供模型路径或上传模型文件",
                    *[gr.Row(visible=False) for _ in range(5)],
                    *[gr.Textbox(value="") for _ in range(5)]
                )

            # 检查文件存在
            from pathlib import Path
            if not Path(actual_path).exists():
                return (
                    f"❌ 错误：模型文件不存在: {actual_path}",
                    *[gr.Row(visible=False) for _ in range(5)],
                    *[gr.Textbox(value="") for _ in range(5)]
                )

            # 加载模型
            ddsp_integration = self.voice_creator.ddsp_integration
            ddsp_integration.load_model(actual_path)

            # 获取speaker列表
            speakers = ddsp_integration.get_available_speakers()
            self._current_speakers = speakers

            # 构建返回值
            updates = []

            # 状态信息
            status = f"✅ DDSP-SVC模型加载成功\n"
            status += f"模型路径: {actual_path}\n"
            status += f"检测到 {len(speakers)} 个speaker"
            updates.append(status)

            # 显示/隐藏speaker行
            for i in range(5):
                if i < len(speakers):
                    updates.append(gr.Row(visible=True))
                else:
                    updates.append(gr.Row(visible=False))

            # 设置speaker名称
            for i in range(5):
                if i < len(speakers):
                    speaker = speakers[i]
                    updates.append(gr.Textbox(value=f"{speaker['name']} (ID: {speaker['id']})"))
                else:
                    updates.append(gr.Textbox(value=""))

            return tuple(updates)

        except Exception as e:
            logger.error(f"加载DDSP-SVC模型失败: {e}")
            return (
                f"❌ 加载失败: {str(e)}",
                *[gr.Row(visible=False) for _ in range(5)],
                *[gr.Textbox(value="") for _ in range(5)]
            )

    def _check_ddsp_status(self) -> str:
        """检查DDSP-SVC模型状态"""
        try:
            ddsp_integration = self.voice_creator.ddsp_integration
            is_loaded = ddsp_integration.is_model_loaded()

            if is_loaded:
                model_info = ddsp_integration.get_model_info() or {}
                speakers = ddsp_integration.get_available_speakers()

                status = f"✅ DDSP-SVC模型已加载\n"
                status += f"Speaker数量: {len(speakers)}\n"
                status += f"设备: {model_info.get('device', '未知')}"

                return status
            else:
                return "❌ DDSP-SVC模型未加载"

        except Exception as e:
            logger.error(f"检查DDSP-SVC状态失败: {e}")
            return f"❌ 状态检查失败: {str(e)}"

    def _calculate_weights_realtime(self, *args) -> Dict[str, Any]:
        """实时计算权重（机械化权重计算）"""
        try:
            if not self._current_speakers:
                return {}

            # 解析权重和启用状态 - args包含weight1, enabled1, weight2, enabled2, ...
            num_speakers = len(self._current_speakers)
            weight_dict = {}

            for i in range(min(num_speakers, 5)):
                weight_idx = i * 2
                enabled_idx = i * 2 + 1

                if enabled_idx < len(args):
                    weight = args[weight_idx] if weight_idx < len(args) else 0
                    enabled = args[enabled_idx]

                    if enabled and weight > 0:
                        speaker = self._current_speakers[i]
                        weight_dict[speaker["id"]] = float(weight)

            if not weight_dict:
                return {}

            # 机械化归一化：总和为1
            total = sum(weight_dict.values())
            if total > 0:
                normalized = {k: v / total for k, v in weight_dict.items()}
                # 添加显示友好的格式
                display_dict = {}
                for speaker_id, weight in normalized.items():
                    speaker_name = next(
                        (s["name"] for s in self._current_speakers if s["id"] == speaker_id),
                        speaker_id
                    )
                    display_dict[f"{speaker_name} ({speaker_id})"] = f"{weight:.3f} ({weight*100:.1f}%)"
                return display_dict

            return {}

        except Exception as e:
            logger.error(f"计算权重失败: {e}")
            return {"错误": str(e)}

    def _generate_preview(self, *args) -> Tuple[Optional[str], str, Dict[str, Any]]:
        """生成预览音频"""
        try:
            # 解析参数
            creation_mode = args[0]
            voice_name = args[1]
            voice_description = args[2]
            pitch_shift = args[3]
            formant_shift = args[4]
            preview_text = args[5]

            # 验证基本参数
            if not voice_name.strip():
                return None, "❌ 错误：请输入角色名称", {}

            if creation_mode == "from_scratch":
                if not self._current_speakers:
                    return None, "❌ 错误：请先加载DDSP-SVC模型", {}

                # 解析说话人权重
                speaker_weights = {}
                num_speakers = len(self._current_speakers)

                for i in range(min(num_speakers, 5)):
                    weight_idx = 6 + i * 2  # 从第6个参数开始
                    enabled_idx = 6 + i * 2 + 1

                    if enabled_idx < len(args):
                        weight = args[weight_idx] if weight_idx < len(args) else 0
                        enabled = args[enabled_idx]

                        if enabled and weight > 0:
                            speaker = self._current_speakers[i]
                            speaker_weights[speaker["id"]] = float(weight)

                if not speaker_weights:
                    return None, "❌ 错误：请至少启用一个说话人", {}

                # 创建参数对象
                params = VoiceBaseCreationParams(
                    voice_name=voice_name,
                    description=voice_description,
                    selected_tag="default",  # 使用默认标签
                    pitch_shift=pitch_shift,
                    formant_shift=formant_shift,
                    speaker_weights=speaker_weights,
                    preview_text=preview_text
                )

                # 执行创建
                def progress_callback(progress: float, message: str):
                    logger.info(f"进度: {progress:.1%} - {message}")

                result = self.voice_creator.create_voice_base(params, progress_callback)
                self._current_result = result

                if result.success:
                    # 构建结果信息
                    info = {
                        "处理时间": f"{result.processing_time:.2f}秒",
                        "音色ID": result.voice_config.voice_id if result.voice_config else "未生成",
                        "说话人权重": speaker_weights,
                        "归一化权重": self._normalize_weights(speaker_weights)
                    }

                    return (
                        result.preview_audio_path,
                        f"✅ 预览生成成功！\n处理时间: {result.processing_time:.2f}秒",
                        info
                    )
                else:
                    return None, f"❌ 预览生成失败：{result.error_message}", {}

            else:
                return None, "❌ 该创建模式暂未实现", {}

        except Exception as e:
            logger.error(f"生成预览失败: {e}")
            return None, f"❌ 生成预览失败：{str(e)}", {}

    def _save_voice(self, voice_name: str) -> str:
        """保存音色"""
        try:
            if not self._current_result or not self._current_result.success:
                return "❌ 错误：请先生成预览"

            if not self._current_result.voice_config:
                return "❌ 错误：音色配置不存在"

            # 更新音色名称（如果用户修改了）
            if voice_name.strip():
                self._current_result.voice_config.name = voice_name.strip()

            # 保存音色配置
            self.voice_creator.save_voice_base(self._current_result.voice_config)

            return f"✅ 音色 '{self._current_result.voice_config.name}' 保存成功！"

        except Exception as e:
            logger.error(f"保存音色失败: {e}")
            return f"❌ 保存失败：{str(e)}"

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化权重"""
        if not weights:
            return {}

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights
