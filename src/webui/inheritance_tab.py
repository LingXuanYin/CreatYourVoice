
"""音色继承界面

这个模块实现音色继承功能的Gradio界面。
设计原则：
1. 直观操作 - 用户友好的界面设计
2. 实时预览 - 提供继承结果的实时预览
3. 参数调整 - 支持继承比例和参数的动态调整
4. 错误处理 - 友好的错误提示和处理
"""

import json
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    # 如果gradio未安装，创建一个模拟的gr对象
    class MockGradio:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    gr = MockGradio()

from ..core import (
    VoiceManager,
    VoiceInheritor,
    InheritanceConfig,
    InheritancePresetManager,
    DDSPSVCConfig,
    IndexTTSConfig,
    VoiceInheritanceError
)

logger = logging.getLogger(__name__)


class InheritanceTab:
    """音色继承界面类

    提供完整的音色继承功能界面，包括：
    1. 父音色选择
    2. 新参数配置
    3. 继承比例调整
    4. 实时预览
    5. 结果保存
    """

    def __init__(self, voice_manager: VoiceManager):
        """初始化继承界面

        Args:
            voice_manager: 音色管理器实例
        """
        self.voice_manager = voice_manager
        self.inheritor = VoiceInheritor(voice_manager)

        # 界面状态
        self.current_preview = None
        self.last_inheritance_result = None

        logger.info("音色继承界面初始化完成")

    def create_interface(self):
        """创建Gradio界面"""

        with gr.Blocks(title="音色继承") as interface:
            gr.Markdown("# 🧬 音色继承")
            gr.Markdown("从现有音色创建新音色，支持参数继承和权重融合")

            with gr.Row():
                # 左侧：配置区域
                with gr.Column(scale=2):
                    self._create_config_section()

                # 右侧：预览和结果区域
                with gr.Column(scale=1):
                    self._create_preview_section()

            # 底部：操作按钮
            with gr.Row():
                self._create_action_buttons()

        return interface

    def _create_config_section(self):
        """创建配置区域"""

        with gr.Tab("基础配置"):
            # 父音色选择
            gr.Markdown("### 选择父音色")

            with gr.Row():
                self.parent_source = gr.Radio(
                    choices=["现有音色", "语音产物文件"],
                    value="现有音色",
                    label="父音色来源"
                )

            # 现有音色选择
            with gr.Group(visible=True) as self.existing_voice_group:
                self.parent_voice_dropdown = gr.Dropdown(
                    choices=self._get_voice_choices(),
                    label="选择父音色",
                    info="选择要继承的音色"
                )

                self.refresh_voices_btn = gr.Button("🔄 刷新音色列表", size="sm")

            # 语音产物文件选择
            with gr.Group(visible=False) as self.voice_product_group:
                self.voice_product_file = gr.File(
                    label="上传语音产物文件",
                    file_types=[".json"]
                )
                gr.Markdown("*选择包含音色配置的JSON文件*", elem_classes=["file-info"])

            # 新音色配置
            gr.Markdown("### 新音色配置")

            self.new_voice_name = gr.Textbox(
                label="新音色名称",
                placeholder="输入新音色的名称"
            )
            gr.Markdown("*继承后的音色名称*", elem_classes=["component-info"])

            # DDSP配置
            with gr.Accordion("DDSP-SVC配置", open=True):
                with gr.Row():
                    self.ddsp_model_path = gr.Textbox(
                        label="模型路径",
                        placeholder="path/to/ddsp/model.pth"
                    )
                    self.ddsp_config_path = gr.Textbox(
                        label="配置路径",
                        placeholder="path/to/ddsp/config.yaml"
                    )

                with gr.Row():
                    self.ddsp_speaker_id = gr.Number(
                        label="说话人ID",
                        value=0,
                        precision=0
                    )
                    self.ddsp_f0_predictor = gr.Dropdown(
                        choices=["rmvpe", "fcpe", "crepe", "harvest"],
                        value="rmvpe",
                        label="F0预测器"
                    )

                with gr.Row():
                    self.ddsp_f0_min = gr.Number(
                        label="F0最小值",
                        value=50.0,
                        minimum=20.0,
                        maximum=200.0
                    )
                    self.ddsp_f0_max = gr.Number(
                        label="F0最大值",
                        value=1100.0,
                        minimum=200.0,
                        maximum=2000.0
                    )

                self.ddsp_threshold = gr.Number(
                    label="响应阈值(dB)",
                    value=-60.0,
                    minimum=-100.0,
                    maximum=0.0
                )

            # IndexTTS配置
            with gr.Accordion("IndexTTS配置", open=True):
                with gr.Row():
                    self.index_model_path = gr.Textbox(
                        label="模型路径",
                        placeholder="path/to/index/model"
                    )
                    self.index_config_path = gr.Textbox(
                        label="配置路径",
                        placeholder="path/to/index/config.yaml"
                    )

                with gr.Row():
                    self.index_speaker_name = gr.Textbox(
                        label="说话人名称",
                        placeholder="speaker_name"
                    )
                    self.index_emotion_strength = gr.Slider(
                        label="情感强度",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                with gr.Row():
                    self.index_speed = gr.Slider(
                        label="语速",
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1
                    )
                    self.index_temperature = gr.Slider(
                        label="温度",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1
                    )

        with gr.Tab("继承设置"):
            # 继承比例
            gr.Markdown("### 继承比例")

            self.inheritance_ratio = gr.Slider(
                label="继承比例",
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.05
            )
            gr.Markdown("*0.0=完全使用新参数，1.0=完全继承父音色*", elem_classes=["component-info"])

            # 预设选择
            gr.Markdown("### 继承预设")

            with gr.Row():
                self.preset_conservative = gr.Button("🛡️ 保守继承 (80%)", size="sm")
                self.preset_balanced = gr.Button("⚖️ 平衡继承 (50%)", size="sm")
                self.preset_innovative = gr.Button("🚀 创新继承 (20%)", size="sm")

            # 高级选项
            with gr.Accordion("高级选项", open=False):
                self.preserve_metadata = gr.Checkbox(
                    label="保留元数据",
                    value=True
                )
                gr.Markdown("*保留父音色的标签和描述信息*", elem_classes=["component-info"])

                self.copy_tags = gr.Checkbox(
                    label="复制标签",
                    value=True
                )
                gr.Markdown("*将父音色的标签复制到新音色*", elem_classes=["component-info"])

                self.auto_generate_name = gr.Checkbox(
                    label="自动生成名称",
                    value=True
                )
                gr.Markdown("*如果名称为空，自动生成继承音色名称*", elem_classes=["component-info"])

    def _create_preview_section(self):
        """创建预览区域"""

        gr.Markdown("### 🔍 继承预览")

        # 预览按钮
        self.preview_btn = gr.Button("生成预览", variant="secondary")

        # 预览结果显示
        with gr.Group():
            self.preview_info = gr.JSON(
                label="预览信息",
                visible=False
            )

            # 权重分布可视化
            self.weight_distribution = gr.Plot(
                label="权重分布",
                visible=False
            )

            # 参数对比
            self.parameter_comparison = gr.DataFrame(
                label="参数对比",
                headers=["参数", "父音色", "新配置", "继承结果"],
                visible=False
            )

        # 警告和建议
        self.warnings_display = gr.Markdown(
            visible=False,
            elem_classes=["warning-box"]
        )

    def _create_action_buttons(self):
        """创建操作按钮"""

        with gr.Row():
            self.inherit_btn = gr.Button(
                "🧬 执行继承",
                variant="primary",
                size="lg"
            )

            self.save_btn = gr.Button(
                "💾 保存音色",
                variant="secondary",
                size="lg",
                visible=False
            )

            self.export_btn = gr.Button(
                "📤 导出配置",
                variant="secondary",
                size="lg",
                visible=False
            )

        # 结果显示
        self.result_display = gr.Markdown(visible=False)

        # 设置事件处理
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """设置事件处理器"""

        # 父音色来源切换
        self.parent_source.change(
            fn=self._toggle_parent_source,
            inputs=[self.parent_source],
            outputs=[self.existing_voice_group, self.voice_product_group]
        )

        # 刷新音色列表
        self.refresh_voices_btn.click(
            fn=self._refresh_voice_list,
            outputs=[self.parent_voice_dropdown]
        )

        # 预设按钮
        self.preset_conservative.click(
            fn=lambda: 0.8,
            outputs=[self.inheritance_ratio]
        )

        self.preset_balanced.click(
            fn=lambda: 0.5,
            outputs=[self.inheritance_ratio]
        )

        self.preset_innovative.click(
            fn=lambda: 0.2,
            outputs=[self.inheritance_ratio]
        )

        # 预览按钮
        self.preview_btn.click(
            fn=self._generate_preview,
            inputs=[
                self.parent_source,
                self.parent_voice_dropdown,
                self.voice_product_file,
                self.ddsp_model_path,
                self.ddsp_config_path,
                self.ddsp_speaker_id,
                self.ddsp_f0_predictor,
                self.ddsp_f0_min,
                self.ddsp_f0_max,
                self.ddsp_threshold,
                self.index_model_path,
                self.index_config_path,
                self.index_speaker_name,
                self.index_emotion_strength,
                self.index_speed,
                self.index_temperature,
                self.inheritance_ratio
            ],
            outputs=[
                self.preview_info,
                self.weight_distribution,
                self.parameter_comparison,
                self.warnings_display
            ]
        )

        # 执行继承
        self.inherit_btn.click(
            fn=self._execute_inheritance,
            inputs=[
                self.parent_source,
                self.parent_voice_dropdown,
                self.voice_product_file,
                self.new_voice_name,
                self.ddsp_model_path,
                self.ddsp_config_path,
                self.ddsp_speaker_id,
                self.ddsp_f0_predictor,
                self.ddsp_f0_min,
                self.ddsp_f0_max,
                self.ddsp_threshold,
                self.index_model_path,
                self.index_config_path,
                self.index_speaker_name,
                self.index_emotion_strength,
                self.index_speed,
                self.index_temperature,
                self.inheritance_ratio,
                self.preserve_metadata,
                self.copy_tags,
                self.auto_generate_name
            ],
            outputs=[
                self.result_display,
                self.save_btn,
                self.export_btn
            ]
        )

        # 保存音色
        self.save_btn.click(
            fn=self._save_voice,
            outputs=[self.result_display]
        )

        # 导出配置
        self.export_btn.click(
            fn=self._export_config,
            outputs=[gr.File()]
        )

    def _toggle_parent_source(self, source: str):
        """切换父音色来源"""
        if source == "现有音色":
            return gr.Group(visible=True), gr.Group(visible=False)
        else:
            return gr.Group(visible=False), gr.Group(visible=True)

    def _get_voice_choices(self) -> List[str]:
        """获取音色选择列表"""
        try:
            voices = self.voice_manager.list_voices()
            return [f"{voice.name} ({voice.voice_id[:8]})" for voice in voices]
        except Exception as e:
            logger.error(f"获取音色列表失败: {e}")
            return []

    def _refresh_voice_list(self):
        """刷新音色列表"""
        choices = self._get_voice_choices()
        return gr.Dropdown(choices=choices)

    def _generate_preview(self, *args):
        """生成继承预览"""
        try:
            # 解析输入参数
            (parent_source, parent_voice_dropdown, voice_product_file,
             ddsp_model_path, ddsp_config_path, ddsp_speaker_id, ddsp_f0_predictor,
             ddsp_f0_min, ddsp_f0_max, ddsp_threshold,
             index_model_path, index_config_path, index_speaker_name,
             index_emotion_strength, index_speed, index_temperature,
             inheritance_ratio) = args

            # 验证输入
            if not self._validate_preview_inputs(parent_source, parent_voice_dropdown, voice_product_file):
                return self._empty_preview_result("请选择有效的父音色")

            # 创建配置对象
            ddsp_config = DDSPSVCConfig(
                model_path=ddsp_model_path,
                config_path=ddsp_config_path,
                speaker_id=int(ddsp_speaker_id),
                f0_predictor=ddsp_f0_predictor,
                f0_min=float(ddsp_f0_min),
                f0_max=float(ddsp_f0_max),
                threhold=float(ddsp_threshold)
            )

            index_config = IndexTTSConfig(
                model_path=index_model_path,
                config_path=index_config_path,
                speaker_name=index_speaker_name,
                emotion_strength=float(index_emotion_strength),
                speed=float(index_speed),
                temperature=float(index_temperature)
            )

            # 获取父音色ID
            if parent_source == "现有音色":
                parent_voice_id = self._extract_voice_id(parent_voice_dropdown)
                preview_data = self.inheritor.preview_inheritance(
                    parent_voice_id,
                    ddsp_config,
                    index_config,
                    float(inheritance_ratio)
                )
            else:
                # 处理语音产物文件
                if not voice_product_file:
                    return self._empty_preview_result("请上传语音产物文件")

                # 这里需要处理文件上传的逻辑
                preview_data = {"error": "语音产物文件预览功能待实现"}

            # 处理预览数据
            if "error" in preview_data:
                return self._empty_preview_result(f"预览失败: {preview_data['error']}")

            # 生成可视化数据
            preview_info = gr.JSON(value=preview_data, visible=True)
            weight_plot = self._create_weight_plot(preview_data.get("resulting_weights", {}))
            param_table = self._create_parameter_table(preview_data)
            warnings = self._format_warnings(preview_data.get("weight_changes", {}))

            return preview_info, weight_plot, param_table, warnings

        except Exception as e:
            logger.error(f"生成预览失败: {e}")
            return self._empty_preview_result(f"预览失败: {str(e)}")

    def _execute_inheritance(self, *args):
        """执行音色继承"""
        try:
            # 解析输入参数
            (parent_source, parent_voice_dropdown, voice_product_file, new_voice_name,
             ddsp_model_path, ddsp_config_path, ddsp_speaker_id, ddsp_f0_predictor,
             ddsp_f0_min, ddsp_f0_max, ddsp_threshold,
             index_model_path, index_config_path, index_speaker_name,
             index_emotion_strength, index_speed, index_temperature,
             inheritance_ratio, preserve_metadata, copy_tags, auto_generate_name) = args

            # 验证输入
            if not new_voice_name and not auto_generate_name:
                return (
                    gr.Markdown("❌ 请输入新音色名称或启用自动生成名称", visible=True),
                    gr.Button(visible=False),
                    gr.Button(visible=False)
                )

            # 创建配置对象
            ddsp_config = DDSPSVCConfig(
                model_path=ddsp_model_path,
                config_path=ddsp_config_path,
                speaker_id=int(ddsp_speaker_id),
                f0_predictor=ddsp_f0_predictor,
                f0_min=float(ddsp_f0_min),
                f0_max=float(ddsp_f0_max),
                threhold=float(ddsp_threshold)
            )

            index_config = IndexTTSConfig(
                model_path=index_model_path,
                config_path=index_config_path,
                speaker_name=index_speaker_name,
                emotion_strength=float(index_emotion_strength),
                speed=float(index_speed),
                temperature=float(index_temperature)
            )

            # 创建继承配置
            inheritance_config = InheritanceConfig(
                inheritance_ratio=float(inheritance_ratio),
                preserve_metadata=preserve_metadata,
                auto_generate_name=auto_generate_name,
                copy_tags=copy_tags
            )

            # 执行继承
            if parent_source == "现有音色":
                parent_voice_id = self._extract_voice_id(parent_voice_dropdown)
                result = self.inheritor.inherit_from_voice(
                    parent_voice_id,
                    new_voice_name or "继承音色",
                    ddsp_config,
                    index_config,
                    inheritance_config
                )
            else:
                # 处理语音产物文件
                if not voice_product_file:
                    return (
                        gr.Markdown("❌ 请上传语音产物文件", visible=True),
                        gr.Button(visible=False),
                        gr.Button(visible=False)
                    )

                result = self.inheritor.inherit_from_voice_product(
                    voice_product_file.name,
                    new_voice_name or "继承音色",
                    ddsp_config,
                    index_config,
                    inheritance_config
                )

            # 保存结果
            self.last_inheritance_result = result

            # 格式化结果信息
            result_info = self._format_inheritance_result(result)

            return (
                gr.Markdown(result_info, visible=True),
                gr.Button("💾 保存音色", visible=True),
                gr.Button("📤 导出配置", visible=True)
            )

        except VoiceInheritanceError as e:
            logger.error(f"音色继承失败: {e}")
            return (
                gr.Markdown(f"❌ 继承失败: {str(e)}", visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False)
            )
        except Exception as e:
            logger.error(f"执行继承时发生错误: {e}")
            return (
                gr.Markdown(f"❌ 发生错误: {str(e)}", visible=True),
                gr.Button(visible=False),
                gr.Button(visible=False)
            )

    def _save_voice(self):
        """保存音色"""
        try:
            if not self.last_inheritance_result:
                return gr.Markdown("❌ 没有可保存的继承结果", visible=True)

            self.voice_manager.save_voice(self.last_inheritance_result.new_voice_config)

            return gr.Markdown(
                f"✅ 音色已保存: {self.last_inheritance_result.new_voice_config.name}",
                visible=True
            )

        except Exception as e:
            logger.error(f"保存音色失败: {e}")
            return gr.Markdown(f"❌ 保存失败: {str(e)}", visible=True)

    def _export_config(self):
        """导出配置"""
        try:
            if not self.last_inheritance_result:
                return gr.File(visible=False)

            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = self.last_inheritance_result.new_voice_config.to_dict()
                json.dump(config_data, f, ensure_ascii=False, indent=2)
                temp_path = f.name

            return gr.File(value=temp_path, visible=True)

        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return gr.File(visible=False)

    def _validate_preview_inputs(self, parent_source: str, parent_voice_dropdown: str, voice_product_file) -> bool:
        """验证预览输入"""
        if parent_source == "现有音色":
            return bool(parent_voice_dropdown)
        else:
            return bool(voice_product_file)

    def _extract_voice_id(self, voice_dropdown_value: str) -> str:
        """从下拉框值中提取音色ID"""
        if not voice_dropdown_value:
            raise ValueError("未选择音色")

        # 格式: "音色名称 (voice_id前8位)"
        if "(" in voice_dropdown_value and ")" in voice_dropdown_value:
            voice_id_part = voice_dropdown_value.split("(")[-1].split(")")[0]
            # 需要根据前8位找到完整的voice_id
            voices = self.voice_manager.list_voices()
            for voice in voices:
                if voice.voice_id.startswith(voice_id_part):
                    return voice.voice_id

        raise ValueError(f"无法解析音色ID: {voice_dropdown_value}")

    def _empty_preview_result(self, message: str):
        """返回空的预览结果"""
        return (
            gr.JSON(visible=False),
            gr.Plot(visible=False),
            gr.DataFrame(visible=False),
            gr.Markdown(f"⚠️ {message}", visible=True)
        )

    def _create_weight_plot(self, weights_data: Dict[str, Any]):
        """创建权重分布图"""
        try:
            import matplotlib.pyplot as plt

            ddsp_weights = weights_data.get("ddsp", {})
            if not ddsp_weights:
                return gr.Plot(visible=False)

            speakers = list(ddsp_weights.keys())
            weights = list(ddsp_weights.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(speakers, weights)
            ax.set_title("DDSP说话人权重分布")
            ax.set_xlabel("说话人ID")
            ax.set_ylabel("权重")
            ax.set_ylim(0, 1)

            # 添加数值标签
            for bar, weight in zip(bars, weights):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom')

            plt.xticks(rotation=45)
            plt.tight_layout()

            return gr.Plot(value=fig, visible=True)

        except Exception as e:
            logger.error(f"创建权重图失败: {e}")
            return gr.Plot(visible=False)

    def _create_parameter_table(self, preview_data: Dict[str, Any]):
        """创建参数对比表"""
        try:
            parent_info = preview_data.get("parent_voice", {})

            # 构建对比数据
            comparison_data = [
                ["继承比例", "-", "-", f"{preview_data.get('inheritance_ratio', 0):.1%}"],
                ["DDSP说话人数", str(len(parent_info.get('ddsp_speakers', []))), "-",
                 str(len(preview_data.get('resulting_weights', {}).get('ddsp', {})))],
                ["IndexTTS说话人", parent_info.get('index_speaker', ''), "-", "-"]
            ]

            return gr.DataFrame(
                value=comparison_data,
                headers=["参数", "父音色", "新配置", "继承结果"],
                visible=True
            )

        except Exception as e:
            logger.error(f"创建参数表失败: {e}")
            return gr.DataFrame(visible=False)

    def _format_warnings(self, weight_changes: Dict[str, Any]):
        """格式化警告信息"""
        try:
            warnings = []

            new_speakers = weight_changes.get("new_speakers", 0)
            removed_speakers = weight_changes.get("removed_speakers", 0)

            if new_speakers > 0:
                warnings.append(f"🆕 新增 {new_speakers} 个说话人")

            if removed_speakers > 0:
                warnings.append(f"🗑️ 移除 {removed_speakers} 个说话人")

            if not warnings:
                warnings.append("✅ 无警告")

            warning_text = "### ⚠️ 继承分析\n" + "\n".join(f"- {w}" for w in warnings)

            return gr.Markdown(warning_text, visible=True)

        except Exception as e:
            logger.error(f"格式化警告失败: {e}")
            return gr.Markdown("⚠️ 无法生成警告信息", visible=True)

    def _format_inheritance_result(self, result) -> str:
        """格式化继承结果"""
        try:
            summary = result.get_summary()

            result_text = f"""
### ✅ 音色继承成功

**新音色信息：**
- 名称: {summary['new_voice_name']}
- ID: {summary['new_voice_id'][:8]}...
- 父音色: {summary['parent_voice_id'][:8]}...
- 继承比例: {summary['inheritance_ratio']:.1%}

**权重分布：**
- DDSP说话人: {len(summary['ddsp_speakers'])} 个
- IndexTTS说话人: {len(summary['index_speakers'])} 个

**处理信息：**
- 处理时间: {summary['processing_time']:.2f}s
- 警告数量: {summary['warnings_count']}

点击"保存音色"将音色添加到音色库中。
"""

            return result_text

        except Exception as e:
            logger.error(f"格式化结果失败: {e}")
            return f"✅ 继承完成，但无法显示详细信息: {str(e)}"


def create_inheritance_interface(voice_manager: VoiceManager):
    """创建音色继承界面的便捷函数

    Args:
        voice_manager: 音色管理器实例

    Returns:
        Gradio界面对象
    """
    tab = InheritanceTab(voice_manager)
    return tab.create_interface()
