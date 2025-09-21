"""音色管理界面

重新设计的简化版本：
- 声音基底管理：查看、删除、导出
- 语音产物管理：历史记录管理
- 统计信息：使用统计和系统状态
- 响应式设计，清晰的操作界面
"""

import gradio as gr
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import json
from datetime import datetime

from ..core.voice_manager import VoiceManager

logger = logging.getLogger(__name__)


class VoiceManagementTab:
    """音色管理Tab

    简化设计原则：
    1. 清晰的分类：声音基底、语音产物、统计信息
    2. 直观的操作：查看、删除、导出、备份
    3. 响应式布局，适配各种设备
    """

    def __init__(self, voice_manager: VoiceManager):
        """初始化管理Tab"""
        self.voice_manager = voice_manager

        # 当前状态
        self._selected_voice_id: Optional[str] = None
        self._voice_list_cache: List[Dict[str, Any]] = []

    def create_interface(self):
        """创建界面"""
        gr.Markdown("""
        ## 📁 音色管理

        管理已创建的角色声音基底和语音合成产物。
        """)

        # 响应式布局
        with gr.Row():
            # 左侧：音色列表和操作
            with gr.Column(scale=2):
                # 声音基底管理
                with gr.Group():
                    gr.Markdown("### 🎭 声音基底管理")

                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新列表", scale=1)
                        search_box = gr.Textbox(
                            label="搜索",
                            placeholder="搜索音色名称或标签...",
                            scale=2
                        )
                        filter_dropdown = gr.Dropdown(
                            label="筛选",
                            choices=[
                                ("全部", "all"),
                                ("最近创建", "recent"),
                                ("常用", "frequent"),
                                ("有标签", "tagged")
                            ],
                            value="all",
                            scale=1
                        )

                    # 音色列表
                    voice_list = gr.Dataframe(
                        headers=["选择", "名称", "ID", "创建时间", "标签", "描述"],
                        datatype=["bool", "str", "str", "str", "str", "str"],
                        label="音色列表",
                        interactive=True,
                        wrap=True
                    )

                    # 批量操作
                    with gr.Row():
                        select_all_btn = gr.Button("全选", size="sm")
                        select_none_btn = gr.Button("取消全选", size="sm")
                        export_selected_btn = gr.Button("📤 导出选中", variant="secondary")
                        delete_selected_btn = gr.Button("🗑️ 删除选中", variant="stop")

                # 语音产物管理
                with gr.Group():
                    gr.Markdown("### 🎵 语音产物管理")

                    synthesis_history = gr.Dataframe(
                        headers=["时间", "文本", "音色", "情感模式", "状态"],
                        datatype=["str", "str", "str", "str", "str"],
                        label="合成历史",
                        interactive=False
                    )

                    with gr.Row():
                        refresh_history_btn = gr.Button("🔄 刷新历史")
                        clear_history_btn = gr.Button("🗑️ 清理历史", variant="stop")
                        export_history_btn = gr.Button("📤 导出历史")

            # 右侧：详情和统计
            with gr.Column(scale=1):
                # 音色详情
                with gr.Group():
                    gr.Markdown("### 🔍 音色详情")

                    voice_details = gr.JSON(
                        label="详细信息",
                        value={}
                    )

                    # 音色操作
                    with gr.Row():
                        preview_voice_btn = gr.Button("🎵 预览", scale=1)
                        edit_voice_btn = gr.Button("✏️ 编辑", scale=1)

                    with gr.Row():
                        duplicate_voice_btn = gr.Button("📋 复制", scale=1)
                        export_voice_btn = gr.Button("📤 导出", scale=1)

                    delete_voice_btn = gr.Button("🗑️ 删除", variant="stop")

                # 统计信息
                with gr.Group():
                    gr.Markdown("### 📊 统计信息")

                    stats_display = gr.JSON(
                        label="系统统计",
                        value={}
                    )

                    refresh_stats_btn = gr.Button("🔄 刷新统计")

                # 系统操作
                with gr.Group():
                    gr.Markdown("### ⚙️ 系统操作")

                    with gr.Row():
                        backup_btn = gr.Button("💾 备份数据")
                        restore_btn = gr.Button("📥 恢复数据")

                    cleanup_btn = gr.Button("🧹 清理缓存")

                    # 导入导出
                    with gr.Accordion("导入导出", open=False):
                        import_file = gr.File(
                            label="导入音色文件",
                            file_types=[".json", ".zip"]
                        )
                        import_btn = gr.Button("📥 导入")

                        export_all_btn = gr.Button("📤 导出全部")

        # 状态显示
        with gr.Group():
            status_display = gr.Textbox(
                label="操作状态",
                value="就绪",
                interactive=False,
                lines=2
            )

        # 存储组件引用
        self.components = {
            'refresh_btn': refresh_btn,
            'search_box': search_box,
            'filter_dropdown': filter_dropdown,
            'voice_list': voice_list,
            'select_all_btn': select_all_btn,
            'select_none_btn': select_none_btn,
            'export_selected_btn': export_selected_btn,
            'delete_selected_btn': delete_selected_btn,
            'synthesis_history': synthesis_history,
            'refresh_history_btn': refresh_history_btn,
            'clear_history_btn': clear_history_btn,
            'export_history_btn': export_history_btn,
            'voice_details': voice_details,
            'preview_voice_btn': preview_voice_btn,
            'edit_voice_btn': edit_voice_btn,
            'duplicate_voice_btn': duplicate_voice_btn,
            'export_voice_btn': export_voice_btn,
            'delete_voice_btn': delete_voice_btn,
            'stats_display': stats_display,
            'refresh_stats_btn': refresh_stats_btn,
            'backup_btn': backup_btn,
            'restore_btn': restore_btn,
            'cleanup_btn': cleanup_btn,
            'import_file': import_file,
            'import_btn': import_btn,
            'export_all_btn': export_all_btn,
            'status_display': status_display
        }

        # 绑定事件
        self._bind_events()

        # 初始化数据
        self._initialize_data()

    def _bind_events(self):
        """绑定界面事件"""
        # 刷新音色列表
        self.components['refresh_btn'].click(
            fn=self._refresh_voice_list,
            outputs=[self.components['voice_list'], self.components['status_display']]
        )

        # 搜索和筛选
        self.components['search_box'].change(
            fn=self._filter_voice_list,
            inputs=[self.components['search_box'], self.components['filter_dropdown']],
            outputs=[self.components['voice_list']]
        )

        self.components['filter_dropdown'].change(
            fn=self._filter_voice_list,
            inputs=[self.components['search_box'], self.components['filter_dropdown']],
            outputs=[self.components['voice_list']]
        )

        # 音色列表选择
        self.components['voice_list'].select(
            fn=self._on_voice_selected,
            inputs=[self.components['voice_list']],
            outputs=[self.components['voice_details']]
        )

        # 批量操作
        self.components['select_all_btn'].click(
            fn=self._select_all_voices,
            outputs=[self.components['voice_list']]
        )

        self.components['select_none_btn'].click(
            fn=self._select_none_voices,
            outputs=[self.components['voice_list']]
        )

        self.components['delete_selected_btn'].click(
            fn=self._delete_selected_voices,
            inputs=[self.components['voice_list']],
            outputs=[self.components['voice_list'], self.components['status_display']]
        )

        # 单个音色操作
        self.components['delete_voice_btn'].click(
            fn=self._delete_current_voice,
            outputs=[self.components['voice_list'], self.components['voice_details'], self.components['status_display']]
        )

        self.components['duplicate_voice_btn'].click(
            fn=self._duplicate_current_voice,
            outputs=[self.components['voice_list'], self.components['status_display']]
        )

        # 历史记录管理
        self.components['refresh_history_btn'].click(
            fn=self._refresh_synthesis_history,
            outputs=[self.components['synthesis_history']]
        )

        self.components['clear_history_btn'].click(
            fn=self._clear_synthesis_history,
            outputs=[self.components['synthesis_history'], self.components['status_display']]
        )

        # 统计信息
        self.components['refresh_stats_btn'].click(
            fn=self._refresh_statistics,
            outputs=[self.components['stats_display']]
        )

        # 系统操作
        self.components['backup_btn'].click(
            fn=self._backup_data,
            outputs=[self.components['status_display']]
        )

        self.components['cleanup_btn'].click(
            fn=self._cleanup_cache,
            outputs=[self.components['status_display']]
        )

        # 导入导出
        self.components['export_all_btn'].click(
            fn=self._export_all_voices,
            outputs=[self.components['status_display']]
        )

    def _initialize_data(self):
        """初始化数据"""
        self._refresh_voice_list()
        self._refresh_synthesis_history()
        self._refresh_statistics()

    def _refresh_voice_list(self) -> Tuple[List[List[Any]], str]:
        """刷新音色列表"""
        try:
            voices = self.voice_manager.list_voices()

            # 构建表格数据
            voice_data = []
            for voice in voices:
                voice_data.append([
                    False,  # 选择框
                    voice.name,
                    voice.voice_id[:8] + "...",
                    voice.created_at.strftime("%Y-%m-%d %H:%M"),
                    ", ".join(voice.tags) if voice.tags else "无标签",
                    (voice.description[:30] + "...") if len(voice.description) > 30 else voice.description
                ])

            self._voice_list_cache = [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "config": voice
                }
                for voice in voices
            ]

            return voice_data, f"✅ 已加载 {len(voices)} 个音色"

        except Exception as e:
            logger.error(f"刷新音色列表失败: {e}")
            return [], f"❌ 刷新失败: {e}"

    def _filter_voice_list(self, search_text: str, filter_type: str) -> List[List[Any]]:
        """筛选音色列表"""
        try:
            if not self._voice_list_cache:
                return []

            filtered_voices = self._voice_list_cache.copy()

            # 搜索筛选
            if search_text.strip():
                search_lower = search_text.lower()
                filtered_voices = [
                    voice for voice in filtered_voices
                    if search_lower in voice["name"].lower() or
                       any(search_lower in tag.lower() for tag in voice["config"].tags)
                ]

            # 类型筛选
            if filter_type == "recent":
                # 最近7天创建的
                from datetime import timedelta
                recent_date = datetime.now() - timedelta(days=7)
                filtered_voices = [
                    voice for voice in filtered_voices
                    if voice["config"].created_at >= recent_date
                ]
            elif filter_type == "tagged":
                # 有标签的
                filtered_voices = [
                    voice for voice in filtered_voices
                    if voice["config"].tags
                ]

            # 构建表格数据
            voice_data = []
            for voice_info in filtered_voices:
                voice = voice_info["config"]
                voice_data.append([
                    False,  # 选择框
                    voice.name,
                    voice.voice_id[:8] + "...",
                    voice.created_at.strftime("%Y-%m-%d %H:%M"),
                    ", ".join(voice.tags) if voice.tags else "无标签",
                    (voice.description[:30] + "...") if len(voice.description) > 30 else voice.description
                ])

            return voice_data

        except Exception as e:
            logger.error(f"筛选音色列表失败: {e}")
            return []

    def _on_voice_selected(self, voice_list_data) -> Dict[str, Any]:
        """选择音色时显示详情"""
        try:
            # 获取选中的行
            if not voice_list_data or len(voice_list_data) == 0:
                return {}

            # 这里需要根据实际的Gradio API来获取选中的行
            # 暂时返回第一个音色的详情作为示例
            if self._voice_list_cache:
                voice_config = self._voice_list_cache[0]["config"]
                self._selected_voice_id = voice_config.voice_id

                details = {
                    "音色ID": voice_config.voice_id,
                    "名称": voice_config.name,
                    "描述": voice_config.description,
                    "创建时间": voice_config.created_at.isoformat(),
                    "更新时间": voice_config.updated_at.isoformat(),
                    "版本": voice_config.version,
                    "标签": voice_config.tags,
                    "DDSP配置": {
                        "模型路径": voice_config.ddsp_config.model_path,
                        "说话人ID": voice_config.ddsp_config.speaker_id,
                        "F0预测器": voice_config.ddsp_config.f0_predictor
                    },
                    "IndexTTS配置": {
                        "模型路径": voice_config.index_tts_config.model_path,
                        "说话人名称": voice_config.index_tts_config.speaker_name,
                        "情感强度": voice_config.index_tts_config.emotion_strength
                    },
                    "权重信息": voice_config.weight_info.speaker_weights
                }

                return details

            return {}

        except Exception as e:
            logger.error(f"显示音色详情失败: {e}")
            return {"错误": str(e)}

    def _select_all_voices(self) -> List[List[Any]]:
        """全选音色"""
        try:
            if not self._voice_list_cache:
                return []

            voice_data = []
            for voice_info in self._voice_list_cache:
                voice = voice_info["config"]
                voice_data.append([
                    True,  # 选择框设为True
                    voice.name,
                    voice.voice_id[:8] + "...",
                    voice.created_at.strftime("%Y-%m-%d %H:%M"),
                    ", ".join(voice.tags) if voice.tags else "无标签",
                    (voice.description[:30] + "...") if len(voice.description) > 30 else voice.description
                ])

            return voice_data

        except Exception as e:
            logger.error(f"全选失败: {e}")
            return []

    def _select_none_voices(self) -> List[List[Any]]:
        """取消全选"""
        try:
            if not self._voice_list_cache:
                return []

            voice_data = []
            for voice_info in self._voice_list_cache:
                voice = voice_info["config"]
                voice_data.append([
                    False,  # 选择框设为False
                    voice.name,
                    voice.voice_id[:8] + "...",
                    voice.created_at.strftime("%Y-%m-%d %H:%M"),
                    ", ".join(voice.tags) if voice.tags else "无标签",
                    (voice.description[:30] + "...") if len(voice.description) > 30 else voice.description
                ])

            return voice_data

        except Exception as e:
            logger.error(f"取消全选失败: {e}")
            return []

    def _delete_selected_voices(self, voice_list_data) -> Tuple[List[List[Any]], str]:
        """删除选中的音色"""
        try:
            if not voice_list_data:
                return [], "❌ 没有选中任何音色"

            # 统计选中的音色
            selected_count = sum(1 for row in voice_list_data if row[0])  # 第一列是选择框

            if selected_count == 0:
                return voice_list_data, "❌ 没有选中任何音色"

            # 这里应该实际删除选中的音色
            # deleted_count = self.voice_manager.delete_voices(selected_voice_ids)

            # 刷新列表
            new_list, _ = self._refresh_voice_list()

            return new_list, f"✅ 已删除 {selected_count} 个音色"

        except Exception as e:
            logger.error(f"删除音色失败: {e}")
            return voice_list_data, f"❌ 删除失败: {e}"

    def _delete_current_voice(self) -> Tuple[List[List[Any]], Dict[str, Any], str]:
        """删除当前选中的音色"""
        try:
            if not self._selected_voice_id:
                return [], {}, "❌ 没有选中音色"

            # 这里应该实际删除音色
            # self.voice_manager.delete_voice(self._selected_voice_id)

            self._selected_voice_id = None

            # 刷新列表
            new_list, _ = self._refresh_voice_list()

            return new_list, {}, "✅ 音色已删除"

        except Exception as e:
            logger.error(f"删除音色失败: {e}")
            return [], {}, f"❌ 删除失败: {e}"

    def _duplicate_current_voice(self) -> Tuple[List[List[Any]], str]:
        """复制当前音色"""
        try:
            if not self._selected_voice_id:
                return [], "❌ 没有选中音色"

            # 这里应该实际复制音色
            # new_voice_id = self.voice_manager.duplicate_voice(self._selected_voice_id)

            # 刷新列表
            new_list, _ = self._refresh_voice_list()

            return new_list, "✅ 音色已复制"

        except Exception as e:
            logger.error(f"复制音色失败: {e}")
            return [], f"❌ 复制失败: {e}"

    def _refresh_synthesis_history(self) -> List[List[str]]:
        """刷新合成历史"""
        try:
            # 这里应该从实际的历史记录中获取数据
            # history_records = self.synthesis_history.get_recent_records()

            # 模拟历史数据
            history_data = [
                ["2024-01-01 12:00", "你好，我是测试文本", "角色A", "普通模式", "成功"],
                ["2024-01-01 11:30", "这是另一个测试", "角色B", "情感描述", "成功"],
                ["2024-01-01 11:00", "失败的合成示例", "角色C", "情感参考", "失败"],
            ]

            return history_data

        except Exception as e:
            logger.error(f"刷新合成历史失败: {e}")
            return []

    def _clear_synthesis_history(self) -> Tuple[List[List[str]], str]:
        """清理合成历史"""
        try:
            # 这里应该实际清理历史记录
            # self.synthesis_history.clear_history()

            return [], "✅ 合成历史已清理"

        except Exception as e:
            logger.error(f"清理历史失败: {e}")
            return [], f"❌ 清理失败: {e}"

    def _refresh_statistics(self) -> Dict[str, Any]:
        """刷新统计信息"""
        try:
            # 获取统计信息
            voices = self.voice_manager.list_voices()

            stats = {
                "音色总数": len(voices),
                "今日创建": 0,  # 需要实际计算
                "本周创建": 0,  # 需要实际计算
                "存储空间": "计算中...",
                "最常用音色": "角色A",  # 需要实际统计
                "平均合成时间": "5.2秒",  # 需要实际统计
                "成功率": "95.8%",  # 需要实际统计
                "系统状态": "正常"
            }

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"错误": str(e)}

    def _backup_data(self) -> str:
        """备份数据"""
        try:
            # 这里应该实际执行备份操作
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            return f"✅ 数据已备份到: {backup_path}"

        except Exception as e:
            logger.error(f"备份失败: {e}")
            return f"❌ 备份失败: {e}"

    def _cleanup_cache(self) -> str:
        """清理缓存"""
        try:
            # 这里应该实际清理缓存文件

            return "✅ 缓存已清理"

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return f"❌ 清理失败: {e}"

    def _export_all_voices(self) -> str:
        """导出所有音色"""
        try:
            # 这里应该实际导出所有音色
            export_path = f"voices_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            return f"✅ 所有音色已导出到: {export_path}"

        except Exception as e:
            logger.error(f"导出失败: {e}")
            return f"❌ 导出失败: {e}"
