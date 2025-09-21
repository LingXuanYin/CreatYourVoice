
"""音色融合界面

这个模块实现音色融合功能的Gradio界面。
设计原则：
1. 多音色管理 - 支持添加、删除、调整多个音色
2. 权重可视化 - 实时显示权重分布和融合效果
3. 智能建议 - 提供融合优化建议和冲突解决
4. 批量操作 - 支持批量导入和融合链创建
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
    VoiceFuser,
    FusionConfig,
    FusionSource,
    FusionPresetManager,
    FusionOptimizer,
    VoiceFusionError,
    VoiceConfig
)

logger = logging.getLogger(__name__)


class FusionTab:
    """音色融合界面类

    提供完整的音色融合功能界面，包括：
    1. 多音色选择和权重配置
    2. 融合参数设置
    3. 实时预览和可视化
    4. 融合执行和结果管理
    5. 批量操作和融合链
    """

    def __init__(self, voice_manager: VoiceManager):
        """初始化融合界面

        Args:
            voice_manager: 音色管理器实例
        """
        self.voice_manager = voice_manager
        self.fuser = VoiceFuser(voice_manager)
        self.optimizer = FusionOptimizer(self.fuser.weight_calculator)

        # 界面状态
        self.fusion_sources = []  # 当前融合源列表
        self.current_preview = None
        self.last_fusion_result = None

        logger.info("音色融合界面初始化完成")

    def create_interface(self):
        """创建Gradio界面"""

        with gr.Blocks(title="音色融合") as interface:
            gr.Markdown("# 🔀 音色融合")
            gr.Markdown("融合多个音色创建新的音色，支持复杂的权重配置和参数融合")

            with gr.Row():
                # 左侧：音色选择和配置
                with gr.Column(scale=2):
                    self._create_source_management_section()
                    self._create_fusion_config_section()

                # 右侧：预览和结果
                with gr.Column(scale=1):
                    self._create_preview_section()
                    self._create_results_section()

            # 底部：操作按钮
            with gr.Row():
                self._create_action_buttons()

        return interface

    def _create_source_management_section(self):
        """创建音色源管理区域"""

        gr.Markdown("### 🎯 音色源管理")

        with gr.Tab("添加音色"):
            # 音色选择方式
            with gr.Row():
                self.source_type = gr.Radio(
                    choices=["现有音色", "语音产物文件", "批量导入"],
                    value="现有音色",
                    label="音色来源"
                )

            # 现有音色选择
            with gr.Group(visible=True) as self.existing_voice_group:
                with gr.Row():
                    self.voice_dropdown = gr.Dropdown(
                        choices=self._get_voice_choices(),
                        label="选择音色",
                        info="选择要融合的音色"
                    )
                    self.refresh_voices_btn = gr.Button("🔄 刷新", size="sm")

                with gr.Row():
                    self.voice_weight = gr.Number(
                        label="权重",
                        value=1.0,
                        minimum=0.0,
                        step=0.1
                    )
                    gr.Markdown("*音色在融合中的权重*", elem_classes=["component-info"])
                    self.voice_priority = gr.Number(
                        label="优先级",
                        value=0,
                        precision=0
                    )
                    gr.Markdown("*数字越大优先级越高*", elem_classes=["component-info"])

                self.add_voice_btn = gr.Button("➕ 添加音色", variant="primary")

            # 语音产物文件
            with gr.Group(visible=False) as self.voice_product_group:
                self.voice_product_files = gr.File(
                    label="上传语音产物文件",
                    file_count="multiple",
                    file_types=[".json"]
                )
                gr.Markdown("*选择包含音色配置的JSON文件*", elem_classes=["component-info"])

                self.add_products_btn = gr.Button("➕ 添加产物", variant="primary")

            # 批量导入
            with gr.Group(visible=False) as self.batch_import_group:
                self.batch_config = gr.Textbox(
                    label="批量配置",
                    placeholder='{"voice_id_1": 0.5, "voice_id_2": 0.3, "voice_id_3": 0.2}',
                    lines=5
                )
                gr.Markdown("*JSON格式的音色ID到权重映射*", elem_classes=["component-info"])

                self.import_batch_btn = gr.Button("📥 批量导入", variant="primary")

        with gr.Tab("管理音色"):
            # 当前音色源列表
            self.sources_display = gr.DataFrame(
                headers=["音色名称", "音色ID", "权重", "优先级", "操作"],
                label="当前融合音色",
                interactive=False
            )

            with gr.Row():
                self.remove_source_btn = gr.Button("🗑️ 移除选中", variant="secondary")
                self.clear_sources_btn = gr.Button("🧹 清空全部", variant="secondary")
                self.normalize_weights_btn = gr.Button("⚖️ 归一化权重", variant="secondary")

            # 权重调整
            gr.Markdown("#### 快速权重调整")
            with gr.Row():
                self.equal_weights_btn = gr.Button("📊 等权重", size="sm")
                self.dominant_weight_btn = gr.Button("👑 主导权重", size="sm")
                self.random_weights_btn = gr.Button("🎲 随机权重", size="sm")

    def _create_fusion_config_section(self):
        """创建融合配置区域"""

        gr.Markdown("### ⚙️ 融合配置")

        with gr.Tab("基础设置"):
            # 融合音色名称
            self.fused_voice_name = gr.Textbox(
                label="融合音色名称",
                placeholder="输入融合后的音色名称"
            )
            gr.Markdown("*新音色的名称*", elem_classes=["component-info"])

            # 融合预设
            gr.Markdown("#### 融合预设")
            with gr.Row():
                self.preset_balanced = gr.Button("⚖️ 平衡融合", size="sm")
                self.preset_conservative = gr.Button("🛡️ 保守融合", size="sm")
                self.preset_aggressive = gr.Button("⚡ 激进融合", size="sm")

            # 高级选项
            with gr.Accordion("高级选项", open=False):
                self.auto_normalize = gr.Checkbox(
                    label="自动归一化权重",
                    value=True
                )
                gr.Markdown("*自动将权重归一化为总和1.0*", elem_classes=["component-info"])

                self.resolve_conflicts = gr.Checkbox(
                    label="自动解决冲突",
                    value=True
                )
                gr.Markdown("*自动解决参数冲突*", elem_classes=["component-info"])

                with gr.Row():
                    self.max_speakers = gr.Number(
                        label="最大说话人数",
                        value=8,
                        minimum=1,
                        maximum=20,
                        precision=0
                    )
                    gr.Markdown("*融合后的最大说话人数量*", elem_classes=["component-info"])

                    self.min_weight_threshold = gr.Number(
                        label="最小权重阈值",
                        value=0.05,
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01
                    )
                    gr.Markdown("*低于此值的权重将被移除*", elem_classes=["component-info"])

                self.preserve_dominant = gr.Checkbox(
                    label="保留主导配置",
                    value=True
                )
                gr.Markdown("*保留权重最大音色的配置参数*", elem_classes=["component-info"])

        with gr.Tab("融合链"):
            gr.Markdown("#### 创建融合链")
            gr.Markdown("支持多步骤融合，逐步构建复杂音色")

            self.chain_steps = gr.Number(
                label="融合步数",
                value=2,
                minimum=2,
                maximum=10,
                precision=0
            )
            gr.Markdown("*融合链的步骤数量*", elem_classes=["component-info"])

            self.chain_config = gr.Textbox(
                label="融合链配置",
                placeholder="每步的配置信息",
                lines=8
            )
            gr.Markdown("*JSON格式的融合链配置*", elem_classes=["component-info"])

            self.create_chain_btn = gr.Button("🔗 创建融合链", variant="primary")

    def _create_preview_section(self):
        """创建预览区域"""

        gr.Markdown("### 🔍 融合预览")

        # 预览控制
        with gr.Row():
            self.preview_btn = gr.Button("生成预览", variant="secondary")
            self.optimize_btn = gr.Button("🎯 优化权重", variant="secondary")

        # 预览结果
        with gr.Group():
            # 权重分布图
            self.weight_chart = gr.Plot(
                label="权重分布",
                visible=False
            )

            # 兼容性分析
            self.compatibility_info = gr.JSON(
                label="兼容性分析",
                visible=False
            )

            # 融合建议
            self.fusion_suggestions = gr.Markdown(
                label="融合建议",
                visible=False
            )

        # 警告和冲突
        self.warnings_display = gr.Markdown(
            visible=False,
            elem_classes=["warning-box"]
        )

    def _create_results_section(self):
        """创建结果区域"""

        gr.Markdown("### 📊 融合结果")

        # 结果显示
        self.result_info = gr.JSON(
            label="融合信息",
            visible=False
        )

        # 结果统计
        self.result_stats = gr.DataFrame(
            headers=["指标", "值", "说明"],
            label="融合统计",
            visible=False
        )

    def _create_action_buttons(self):
        """创建操作按钮"""

        with gr.Row():
            self.fuse_btn = gr.Button(
                "🔀 执行融合",
                variant="primary",
                size="lg"
            )

            self.save_result_btn = gr.Button(
                "💾 保存音色",
                variant="secondary",
                size="lg",
                visible=False
            )

            self.export_result_btn = gr.Button(
                "📤 导出配置",
                variant="secondary",
                size="lg",
                visible=False
            )

        # 结果显示
        self.fusion_result_display = gr.Markdown(visible=False)

        # 设置事件处理
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """设置事件处理器"""

        # 音色来源切换
        self.source_type.change(
            fn=self._toggle_source_type,
            inputs=[self.source_type],
            outputs=[self.existing_voice_group, self.voice_product_group, self.batch_import_group]
        )

        # 刷新音色列表
        self.refresh_voices_btn.click(
            fn=self._refresh_voice_list,
            outputs=[self.voice_dropdown]
        )

        # 添加音色
        self.add_voice_btn.click(
            fn=self._add_voice_source,
            inputs=[self.voice_dropdown, self.voice_weight, self.voice_priority],
            outputs=[self.sources_display, self.voice_dropdown, self.voice_weight]
        )

        # 添加语音产物
        self.add_products_btn.click(
            fn=self._add_voice_products,
            inputs=[self.voice_product_files],
            outputs=[self.sources_display]
        )

        # 批量导入
        self.import_batch_btn.click(
            fn=self._import_batch_config,
            inputs=[self.batch_config],
            outputs=[self.sources_display, self.batch_config]
        )

        # 管理操作
        self.clear_sources_btn.click(
            fn=self._clear_sources,
            outputs=[self.sources_display]
        )

        self.normalize_weights_btn.click(
            fn=self._normalize_weights,
            outputs=[self.sources_display]
        )

        # 快速权重调整
        self.equal_weights_btn.click(
            fn=self._set_equal_weights,
            outputs=[self.sources_display]
        )

        self.dominant_weight_btn.click(
            fn=self._set_dominant_weights,
            outputs=[self.sources_display]
        )

        self.random_weights_btn.click(
            fn=self._set_random_weights,
            outputs=[self.sources_display]
        )

        # 预设按钮
        self.preset_balanced.click(
            fn=self._apply_balanced_preset,
            outputs=[self.auto_normalize, self.resolve_conflicts, self.max_speakers, self.min_weight_threshold]
        )

        self.preset_conservative.click(
            fn=self._apply_conservative_preset,
            outputs=[self.auto_normalize, self.resolve_conflicts, self.max_speakers, self.min_weight_threshold]
        )

        self.preset_aggressive.click(
            fn=self._apply_aggressive_preset,
            outputs=[self.auto_normalize, self.resolve_conflicts, self.max_speakers, self.min_weight_threshold]
        )

        # 预览和优化
        self.preview_btn.click(
            fn=self._generate_preview,
            inputs=[
                self.auto_normalize,
                self.resolve_conflicts,
                self.max_speakers,
                self.min_weight_threshold,
                self.preserve_dominant
            ],
            outputs=[
                self.weight_chart,
                self.compatibility_info,
                self.fusion_suggestions,
                self.warnings_display
            ]
        )

        self.optimize_btn.click(
            fn=self._optimize_fusion,
            inputs=[self.max_speakers],
            outputs=[self.sources_display, self.fusion_suggestions]
        )

        # 执行融合
        self.fuse_btn.click(
            fn=self._execute_fusion,
            inputs=[
                self.fused_voice_name,
                self.auto_normalize,
                self.resolve_conflicts,
                self.max_speakers,
                self.min_weight_threshold,
                self.preserve_dominant
            ],
            outputs=[
                self.fusion_result_display,
                self.result_info,
                self.result_stats,
                self.save_result_btn,
                self.export_result_btn
            ]
        )

        # 保存和导出
        self.save_result_btn.click(
            fn=self._save_fusion_result,
            outputs=[self.fusion_result_display]
        )

        self.export_result_btn.click(
            fn=self._export_fusion_result,
            outputs=[gr.File()]
        )

        # 融合链
        self.create_chain_btn.click(
            fn=self._create_fusion_chain,
            inputs=[self.chain_steps, self.chain_config, self.fused_voice_name],
            outputs=[self.fusion_result_display, self.result_info]
        )

    def _toggle_source_type(self, source_type: str):
        """切换音色来源类型"""
        if source_type == "现有音色":
            return gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False)
        elif source_type == "语音产物文件":
            return gr.Group(visible=False), gr.Group(visible=True), gr.Group(visible=False)
        else:  # 批量导入
            return gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=True)

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

    def _add_voice_source(self, voice_dropdown: str, weight: float, priority: int):
        """添加音色源"""
        try:
            if not voice_dropdown:
                return self._get_sources_display(), voice_dropdown, weight

            # 提取音色ID
            voice_id = self._extract_voice_id(voice_dropdown)
            voice_config = self.voice_manager.load_voice(voice_id)

            # 检查是否已存在
            for source in self.fusion_sources:
                if source.voice_config.voice_id == voice_id:
                    return self._get_sources_display(), "", 1.0  # 清空输入

            # 添加新源
            fusion_source = FusionSource(
                voice_config=voice_config,
                weight=max(0.0, weight),
                priority=int(priority)
            )

            self.fusion_sources.append(fusion_source)

            return self._get_sources_display(), "", 1.0  # 清空输入

        except Exception as e:
            logger.error(f"添加音色源失败: {e}")
            return self._get_sources_display(), voice_dropdown, weight

    def _add_voice_products(self, files):
        """添加语音产物文件"""
        try:
            if not files:
                return self._get_sources_display()

            for file in files:
                try:
                    voice_config = VoiceConfig.load_from_file(file.name)
                    fusion_source = FusionSource(
                        voice_config=voice_config,
                        weight=1.0,
                        priority=0
                    )
                    self.fusion_sources.append(fusion_source)
                except Exception as e:
                    logger.warning(f"加载语音产物失败 {file.name}: {e}")

            return self._get_sources_display()

        except Exception as e:
            logger.error(f"添加语音产物失败: {e}")
            return self._get_sources_display()

    def _import_batch_config(self, batch_config: str):
        """批量导入配置"""
        try:
            if not batch_config.strip():
                return self._get_sources_display(), batch_config

            config_data = json.loads(batch_config)

            for voice_id, weight in config_data.items():
                try:
                    voice_config = self.voice_manager.load_voice(voice_id)
                    fusion_source = FusionSource(
                        voice_config=voice_config,
                        weight=float(weight),
                        priority=0
                    )
                    self.fusion_sources.append(fusion_source)
                except Exception as e:
                    logger.warning(f"加载音色失败 {voice_id}: {e}")

            return self._get_sources_display(), ""  # 清空配置

        except json.JSONDecodeError:
            logger.error("批量配置JSON格式错误")
            return self._get_sources_display(), batch_config
        except Exception as e:
            logger.error(f"批量导入失败: {e}")
            return self._get_sources_display(), batch_config

    def _clear_sources(self):
        """清空所有音色源"""
        self.fusion_sources.clear()
        return self._get_sources_display()

    def _normalize_weights(self):
        """归一化权重"""
        if not self.fusion_sources:
            return self._get_sources_display()

        total_weight = sum(source.weight for source in self.fusion_sources)
        if total_weight > 0:
            for source in self.fusion_sources:
                source.weight = source.weight / total_weight

        return self._get_sources_display()

    def _set_equal_weights(self):
        """设置等权重"""
        if not self.fusion_sources:
            return self._get_sources_display()

        equal_weight = 1.0 / len(self.fusion_sources)
        for source in self.fusion_sources:
            source.weight = equal_weight

        return self._get_sources_display()

    def _set_dominant_weights(self):
        """设置主导权重"""
        if not self.fusion_sources:
            return self._get_sources_display()

        # 第一个音色权重0.7，其他平分0.3
        if len(self.fusion_sources) == 1:
            self.fusion_sources[0].weight = 1.0
        else:
            self.fusion_sources[0].weight = 0.7
            other_weight = 0.3 / (len(self.fusion_sources) - 1)
            for source in self.fusion_sources[1:]:
                source.weight = other_weight

        return self._get_sources_display()

    def _set_random_weights(self):
        """设置随机权重"""
        if not self.fusion_sources:
            return self._get_sources_display()

        import random

        # 生成随机权重并归一化
        random_weights = [random.random() for _ in self.fusion_sources]
        total = sum(random_weights)

        for source, weight in zip(self.fusion_sources, random_weights):
            source.weight = weight / total

        return self._get_sources_display()

    def _get_sources_display(self):
        """获取音色源显示数据"""
        if not self.fusion_sources:
            return gr.DataFrame(value=[], headers=["音色名称", "音色ID", "权重", "优先级", "操作"])

        data = []
        for i, source in enumerate(self.fusion_sources):
            data.append([
                source.voice_config.name,
                source.voice_config.voice_id[:8] + "...",
                f"{source.weight:.3f}",
                str(source.priority),
                f"索引: {i}"
            ])

        return gr.DataFrame(value=data, headers=["音色名称", "音色ID", "权重", "优先级", "操作"])

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

    def _apply_balanced_preset(self):
        """应用平衡预设"""
        preset = FusionPresetManager.get_balanced_preset()
        return (
            preset.auto_normalize_weights,
            preset.resolve_conflicts,
            preset.max_speakers,
            preset.min_weight_threshold
        )

    def _apply_conservative_preset(self):
        """应用保守预设"""
        preset = FusionPresetManager.get_conservative_preset()
        return (
            preset.auto_normalize_weights,
            preset.resolve_conflicts,
            preset.max_speakers,
            preset.min_weight_threshold
        )

    def _apply_aggressive_preset(self):
        """应用激进预设"""
        preset = FusionPresetManager.get_aggressive_preset()
        return (
            preset.auto_normalize_weights,
            preset.resolve_conflicts,
            preset.max_speakers,
            preset.min_weight_threshold
        )

    def _generate_preview(self, auto_normalize: bool, resolve_conflicts: bool,
                         max_speakers: int, min_weight_threshold: float, preserve_dominant: bool):
        """生成融合预览"""
        try:
            if len(self.fusion_sources) < 2:
                return self._empty_preview_result("至少需要两个音色进行融合")

            # 创建融合配置
            fusion_config = FusionConfig(
                auto_normalize_weights=auto_normalize,
                resolve_conflicts=resolve_conflicts,
                max_speakers=int(max_speakers),
                min_weight_threshold=float(min_weight_threshold),
                preserve_dominant_config=preserve_dominant
            )

            # 生成预览
            preview_data = self.fuser.preview_fusion(self.fusion_sources, fusion_config)

            if "error" in preview_data:
                return self._empty_preview_result(f"预览失败: {preview_data['error']}")

            # 生成可视化
            weight_chart = self._create_fusion_chart(preview_data.get("fusion_weights", {}))
            compatibility_info = gr.JSON(value=preview_data.get("compatibility_analysis", {}), visible=True)
            suggestions = self._format_fusion_suggestions(preview_data)
            warnings = self._format_fusion_warnings(preview_data.get("estimated_conflicts", []))

            return weight_chart, compatibility_info, suggestions, warnings

        except Exception as e:
            logger.error(f"生成预览失败: {e}")
            return self._empty_preview_result(f"预览失败: {str(e)}")

    def _optimize_fusion(self, target_speakers: int):
        """优化融合配置"""
        try:
            if not self.fusion_sources:
                return self._get_sources_display(), gr.Markdown("没有音色源需要优化", visible=True)

            # 优化权重
            optimized_sources = self.optimizer.optimize_fusion_weights(
                self.fusion_sources,
                int(target_speakers)
            )

            # 更新融合源
            self.fusion_sources = optimized_sources

            # 生成建议
            suggestions = ["### 🎯 优化完成", "- 权重已优化", f"- 目标说话人数: {target_speakers}"]
            suggestion_text = "\n".join(suggestions)

            return self._get_sources_display(), gr.Markdown(suggestion_text, visible=True)

        except Exception as e:
            logger.error(f"优化失败: {e}")
            return self._get_sources_display(), gr.Markdown(f"优化失败: {str(e)}", visible=True)

    def _execute_fusion(self, fused_name: str, auto_normalize: bool, resolve_conflicts: bool,
                       max_speakers: int, min_weight_threshold: float, preserve_dominant: bool):
        """执行音色融合"""
        try:
            if len(self.fusion_sources) < 2:
                return self._empty_fusion_result("至少需要两个音色进行融合")

            if not fused_name.strip():
                return self._empty_fusion_result("请输入融合音色名称")

            # 创建融合配置
            fusion_config = FusionConfig(
                auto_normalize_weights=auto_normalize,
                resolve_conflicts=resolve_conflicts,
                max_speakers=int(max_speakers),
                min_weight_threshold=float(min_weight_threshold),
                preserve_dominant_config=preserve_dominant
            )

            # 执行融合
            result = self.fuser.fuse_voices(self.fusion_sources, fused_name.strip(), fusion_config)

            # 保存结果
            self.last_fusion_result = result

            # 格式化结果
            result_display = self._format_fusion_result(result)
            result_info = gr.JSON(value=result.get_summary(), visible=True)
            result_stats = self._create_result_stats(result)

            return (
                gr.Markdown(result_display, visible=True),
                result_info,
                result_stats,
                gr.Button("💾 保存音色", visible=True),
                gr.Button("📤 导出配置", visible=True)
            )

        except VoiceFusionError as e:
            logger.error(f"音色融合失败: {e}")
            return self._empty_fusion_result(f"融合失败: {str(e)}")
        except Exception as e:
            logger.error(f"执行融合时发生错误: {e}")
            return self._empty_fusion_result(f"发生错误: {str(e)}")

    def _save_fusion_result(self):
        """保存融合结果"""
        try:
            if not self.last_fusion_result:
                return gr.Markdown("❌ 没有可保存的融合结果", visible=True)

            self.voice_manager.save_voice(self.last_fusion_result.fused_voice_config)

            return gr.Markdown(
                f"✅ 音色已保存: {self.last_fusion_result.fused_voice_config.name}",
                visible=True
            )

        except Exception as e:
            logger.error(f"保存融合结果失败: {e}")
            return gr.Markdown(f"❌ 保存失败: {str(e)}", visible=True)

    def _export_fusion_result(self):
        """导出融合结果"""
        try:
            if not self.last_fusion_result:
                return gr.File(visible=False)

            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = self.last_fusion_result.fused_voice_config.to_dict()
                json.dump(config_data, f, ensure_ascii=False, indent=2)
                temp_path = f.name

            return gr.File(value=temp_path, visible=True)

        except Exception as e:
            logger.error(f"导出融合结果失败: {e}")
            return gr.File(visible=False)

    def _create_fusion_chain(self, chain_steps: int, chain_config: str, final_name: str):
        """创建融合链"""
        try:
            if not final_name.strip():
                return (
                    gr.Markdown("❌ 请输入最终音色名称", visible=True),
                    gr.JSON(visible=False)
                )

            if len(self.fusion_sources) < 2:
                return (
                    gr.Markdown("❌ 至少需要两个音色创建融合链", visible=True),
                    gr.JSON(visible=False)
                )

            # 解析融合链配置
            try:
                if chain_config.strip():
                    chain_data = json.loads(chain_config)
                else:
                    # 默认配置：平均分配到各步骤
                    chain_data = self._generate_default_chain_config(int(chain_steps))
            except json.JSONDecodeError:
                return (
                    gr.Markdown("❌ 融合链配置JSON格式错误", visible=True),
                    gr.JSON(visible=False)
                )

            # 执行融合链
            from ..core import create_fusion_chain

            # 构建融合步骤
            fusion_steps = []
            for i in range(int(chain_steps)):
                step_config = chain_data.get(f"step_{i+1}", {})
                voice_weights = step_config.get("voice_ids_and_weights", {})

                if not voice_weights and i == 0:
                    # 第一步使用当前融合源
                    voice_weights = {
                        source.voice_config.voice_id: source.weight
                        for source in self.fusion_sources
                    }

                fusion_steps.append({
                    "voice_ids_and_weights": voice_weights,
                    "previous_weight": step_config.get("previous_weight", 0.5),
                    "fusion_config": FusionPresetManager.get_balanced_preset()
                })

            # 执行融合链
            results = create_fusion_chain(self.voice_manager, fusion_steps, final_name.strip())

            # 保存最终结果
            if results:
                self.last_fusion_result = results[-1]

            # 格式化结果
            result_info = {
                "chain_length": len(results),
                "final_voice_id": results[-1].fused_voice_config.voice_id if results else None,
                "total_processing_time": sum(r.processing_time for r in results),
                "steps": [r.get_summary() for r in results]
            }

            result_text = f"""
### ✅ 融合链创建成功

**链信息：**
- 步骤数量: {len(results)}
- 最终音色: {final_name}
- 总处理时间: {sum(r.processing_time for r in results):.2f}s

**各步骤结果：**
{self._format_chain_steps(results)}
"""

            return (
                gr.Markdown(result_text, visible=True),
                gr.JSON(value=result_info, visible=True)
            )

        except Exception as e:
            logger.error(f"创建融合链失败: {e}")
            return (
                gr.Markdown(f"❌ 创建融合链失败: {str(e)}", visible=True),
                gr.JSON(visible=False)
            )

    def _empty_preview_result(self, message: str):
        """返回空的预览结果"""
        return (
            gr.Plot(visible=False),
            gr.JSON(visible=False),
            gr.Markdown(visible=False),
            gr.Markdown(f"⚠️ {message}", visible=True)
        )

    def _empty_fusion_result(self, message: str):
        """返回空的融合结果"""
        return (
            gr.Markdown(f"❌ {message}", visible=True),
            gr.JSON(visible=False),
            gr.DataFrame(visible=False),
            gr.Button(visible=False),
            gr.Button(visible=False)
        )

    def _create_fusion_chart(self, fusion_weights: Dict[str, Any]):
        """创建融合权重图表"""
        try:
            import matplotlib.pyplot as plt

            combined_weights = fusion_weights.get("combined", {})
            if not combined_weights:
                return gr.Plot(visible=False)

            speakers = list(combined_weights.keys())
            weights = list(combined_weights.values())

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 柱状图
            bars = ax1.bar(speakers, weights)
            ax1.set_title("融合权重分布")
            ax1.set_xlabel("说话人ID")
            ax1.set_ylabel("权重")
            ax1.set_ylim(0, 1)

            # 添加数值标签
            for bar, weight in zip(bars, weights):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{weight:.3f}', ha='center', va='bottom')

            # 饼图
            ax2.pie(weights, labels=speakers, autopct='%1.1f%%', startangle=90)
            ax2.set_title("权重占比")

            plt.xticks(rotation=45)
            plt.tight_layout()

            return gr.Plot(value=fig, visible=True)

        except Exception as e:
            logger.error(f"创建融合图表失败: {e}")
            return gr.Plot(visible=False)

    def _format_fusion_suggestions(self, preview_data: Dict[str, Any]):
        """格式化融合建议"""
        try:
            suggestions = ["### 🎯 融合建议"]

            # 兼容性分析
            compatibility = preview_data.get("compatibility_analysis", {})
            if not compatibility.get("model_compatibility", True):
                suggestions.append("⚠️ 检测到模型兼容性问题，建议检查模型路径")

            # 说话人分析
            speaker_dist = preview_data.get("speaker_distribution", {})
            total_speakers = speaker_dist.get("total_speakers", 0)

            if total_speakers > 10:
                suggestions.append(f"📊 说话人数量较多({total_speakers})，建议优化以提高性能")
            elif total_speakers < 3:
                suggestions.append("📊 说话人数量较少，可能影响融合效果的丰富性")

            # 权重分布分析
            max_weight = speaker_dist.get("max_weight", 0)
            if max_weight > 0.8:
                suggestions.append("⚖️ 存在主导说话人，考虑调整权重以获得更好的融合效果")

            # 冲突分析
            conflicts = preview_data.get("estimated_conflicts", [])
            if conflicts:
                suggestions.append(f"⚠️ 检测到 {len(conflicts)} 个潜在冲突，建议启用自动解决冲突")

            if len(suggestions) == 1:
                suggestions.append("✅ 融合配置良好，可以执行融合")

            return gr.Markdown("\n".join(f"- {s}" if not s.startswith("#") else s for s in suggestions), visible=True)

        except Exception as e:
            logger.error(f"格式化融合建议失败: {e}")
            return gr.Markdown("⚠️ 无法生成融合建议", visible=True)

    def _format_fusion_warnings(self, conflicts: List[str]):
        """格式化融合警告"""
        try:
            if not conflicts:
                return gr.Markdown("✅ 无冲突检测", visible=True)

            warning_text = "### ⚠️ 融合警告\n" + "\n".join(f"- {conflict}" for conflict in conflicts)
            return gr.Markdown(warning_text, visible=True)

        except Exception as e:
            logger.error(f"格式化融合警告失败: {e}")
            return gr.Markdown("⚠️ 无法生成警告信息", visible=True)

    def _format_fusion_result(self, result) -> str:
        """格式化融合结果"""
        try:
            summary = result.get_summary()

            result_text = f"""
### ✅ 音色融合成功

**融合音色信息：**
- 名称: {summary['fused_voice_name']}
- ID: {summary['fused_voice_id'][:8]}...
- 源音色数量: {summary['source_count']}

**权重分布：**
- 最终说话人: {len(summary['final_speakers'])} 个
- 源音色: {', '.join(id[:8] + '...' for id in summary['source_voice_ids'])}

**处理信息：**
- 处理时间: {summary['processing_time']:.2f}s
- 警告数量: {summary['warnings_count']}
- 冲突解决: {summary['conflicts_count']} 个

点击"保存音色"将音色添加到音色库中。
"""

            return result_text

        except Exception as e:
            logger.error(f"格式化融合结果失败: {e}")
            return f"✅ 融合完成，但无法显示详细信息: {str(e)}"

    def _create_result_stats(self, result):
        """创建结果统计表"""
        try:
            summary = result.get_summary()

            stats_data = [
                ["源音色数量", str(summary['source_count']), "参与融合的音色数量"],
                ["最终说话人数", str(len(summary['final_speakers'])), "融合后的说话人数量"],
                ["处理时间", f"{summary['processing_time']:.2f}s", "融合处理耗时"],
                ["警告数量", str(summary['warnings_count']), "融合过程中的警告"],
                ["冲突解决", str(summary['conflicts_count']), "自动解决的参数冲突"]
            ]

            return gr.DataFrame(
                value=stats_data,
                headers=["指标", "值", "说明"],
                visible=True
            )

        except Exception as e:
            logger.error(f"创建结果统计失败: {e}")
            return gr.DataFrame(visible=False)

    def _generate_default_chain_config(self, steps: int) -> Dict[str, Any]:
        """生成默认融合链配置"""
        config = {}

        # 将音色源分配到各步骤
        sources_per_step = max(2, len(self.fusion_sources) // steps)

        for i in range(steps):
            start_idx = i * sources_per_step
            end_idx = min(start_idx + sources_per_step, len(self.fusion_sources))

            if i == steps - 1:  # 最后一步包含剩余所有音色
                end_idx = len(self.fusion_sources)

            step_sources = self.fusion_sources[start_idx:end_idx]
            voice_weights = {
                source.voice_config.voice_id: source.weight
                for source in step_sources
            }

            config[f"step_{i+1}"] = {
                "voice_ids_and_weights": voice_weights,
                "previous_weight": 0.5 if i > 0 else 0.0
            }

        return config

    def _format_chain_steps(self, results) -> str:
        """格式化融合链步骤"""
        try:
            steps_text = []
            for i, result in enumerate(results, 1):
                summary = result.get_summary()
                steps_text.append(
                    f"步骤 {i}: {summary['fused_voice_name']} "
                    f"({summary['source_count']} 源音色, "
                    f"{summary['processing_time']:.2f}s)"
                )

            return "\n".join(f"- {step}" for step in steps_text)

        except Exception as e:
            logger.error(f"格式化融合链步骤失败: {e}")
            return "- 无法显示步骤详情"


def create_fusion_interface(voice_manager: VoiceManager):
    """创建音色融合界面的便捷函数

    Args:
        voice_manager: 音色管理器实例

    Returns:
        Gradio界面对象
    """
    tab = FusionTab(voice_manager)
    return tab.create_interface()
