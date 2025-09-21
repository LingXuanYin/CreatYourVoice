"""DDSP-SVC版本选择界面

这个模块提供DDSP-SVC版本选择和管理的用户界面。
功能包括：
1. 版本检测和显示
2. 版本切换
3. 版本特性对比
4. 配置管理
"""

import gradio as gr
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..utils.version_detector import DDSPSVCVersion, get_ddsp_svc_version, is_version_supported
from ..integrations.version_manager import get_version_manager
from ..integrations.ddsp_svc_unified import DDSPSVCUnified

logger = logging.getLogger(__name__)


class DDSPSVCVersionUI:
    """DDSP-SVC版本选择界面"""

    def __init__(self):
        """初始化版本选择界面"""
        self.version_manager = get_version_manager()
        self.current_unified: Optional[DDSPSVCUnified] = None

    def create_version_tab(self) -> gr.Tab:
        """创建版本管理标签页

        Returns:
            gr.Tab: 版本管理标签页
        """
        with gr.Tab("版本管理") as tab:
            gr.Markdown("## DDSP-SVC版本管理")
            gr.Markdown("管理和切换DDSP-SVC的不同版本（6.1和6.3）")

            with gr.Row():
                with gr.Column(scale=2):
                    # 版本信息显示
                    gr.Markdown("### 当前版本信息")
                    version_info_display = gr.JSON(
                        label="版本详情",
                        value=self._get_version_info_display()
                    )

                    # 版本切换
                    gr.Markdown("### 版本切换")
                    with gr.Row():
                        version_dropdown = gr.Dropdown(
                            choices=["auto", "6.1", "6.3"],
                            value="auto",
                            label="选择版本"
                        )
                gr.Markdown("*auto表示自动检测*", elem_classes=["component-info"])
                        switch_btn = gr.Button("切换版本", variant="primary")

                    switch_status = gr.Textbox(
                        label="切换状态",
                        interactive=False,
                        placeholder="版本切换状态将在这里显示"
                    )

                with gr.Column(scale=1):
                    # 版本特性对比
                    gr.Markdown("### 版本特性对比")
                    features_display = gr.HTML(self._get_features_comparison())

                    # 刷新按钮
                    refresh_btn = gr.Button("刷新信息", variant="secondary")

            # 高级设置
            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    ddsp_path_input = gr.Textbox(
                        label="DDSP-SVC路径",
                        placeholder="留空使用默认路径"
                    )
                gr.Markdown("*自定义DDSP-SVC项目路径*", elem_classes=["component-info"])
                    set_path_btn = gr.Button("设置路径")

                with gr.Row():
                    clear_cache_btn = gr.Button("清理缓存", variant="stop")
                    force_detect_btn = gr.Button("强制重新检测")

                cache_status = gr.Textbox(
                    label="操作状态",
                    interactive=False
                )

            # 事件绑定
            switch_btn.click(
                fn=self._switch_version,
                inputs=[version_dropdown],
                outputs=[switch_status, version_info_display]
            )

            refresh_btn.click(
                fn=self._refresh_info,
                outputs=[version_info_display, features_display]
            )

            set_path_btn.click(
                fn=self._set_ddsp_path,
                inputs=[ddsp_path_input],
                outputs=[cache_status, version_info_display]
            )

            clear_cache_btn.click(
                fn=self._clear_cache,
                outputs=[cache_status]
            )

            force_detect_btn.click(
                fn=self._force_detect,
                outputs=[cache_status, version_info_display]
            )

        return tab

    def create_version_selector(self) -> Tuple[gr.Dropdown, gr.Button, gr.Textbox]:
        """创建简单的版本选择器（用于其他标签页）

        Returns:
            Tuple[gr.Dropdown, gr.Button, gr.Textbox]: 版本选择器组件
        """
        with gr.Row():
            version_dropdown = gr.Dropdown(
                choices=["auto", "6.1", "6.3"],
                value="auto",
                label="DDSP-SVC版本",
                scale=2
            )
            switch_btn = gr.Button("切换", scale=1)

        status_text = gr.Textbox(
            label="版本状态",
            interactive=False,
            visible=False
        )

        # 绑定事件
        switch_btn.click(
            fn=self._switch_version_simple,
            inputs=[version_dropdown],
            outputs=[status_text]
        )

        return version_dropdown, switch_btn, status_text

    def _get_version_info_display(self) -> Dict[str, Any]:
        """获取版本信息显示"""
        try:
            version_info = self.version_manager.detect_and_set_version()
            config = self.version_manager.get_version_config()

            return {
                "检测到的版本": version_info.version.value,
                "Git分支": version_info.branch or "未知",
                "提交哈希": version_info.commit_hash or "未知",
                "项目路径": str(version_info.path),
                "版本特性": version_info.features,
                "配置信息": config,
                "支持的版本": [v.value for v in self.version_manager.get_supported_versions()]
            }
        except Exception as e:
            logger.error(f"获取版本信息失败: {e}")
            return {"错误": str(e)}

    def _get_features_comparison(self) -> str:
        """获取版本特性对比HTML"""
        html = """
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #f0f0f0;">
                <th style="border: 1px solid #ddd; padding: 8px;">特性</th>
                <th style="border: 1px solid #ddd; padding: 8px;">6.1版本</th>
                <th style="border: 1px solid #ddd; padding: 8px;">6.3版本</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">声域偏移</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: red;">❌ 不支持</td>
                <td style="border: 1px solid #ddd; padding: 8px; color: green;">✅ 支持</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">音量处理</td>
                <td style="border: 1px solid #ddd; padding: 8px;">基础处理</td>
                <td style="border: 1px solid #ddd; padding: 8px;">改进处理</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">推理方式</td>
                <td style="border: 1px solid #ddd; padding: 8px;">直接返回音频</td>
                <td style="border: 1px solid #ddd; padding: 8px;">分离mel和vocoder</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">默认t_start</td>
                <td style="border: 1px solid #ddd; padding: 8px;">0.7</td>
                <td style="border: 1px solid #ddd; padding: 8px;">0.0</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;">稳定性</td>
                <td style="border: 1px solid #ddd; padding: 8px;">稳定</td>
                <td style="border: 1px solid #ddd; padding: 8px;">最新功能</td>
            </tr>
        </table>
        """
        return html

    def _switch_version(self, version: str) -> Tuple[str, Dict[str, Any]]:
        """切换版本

        Args:
            version: 目标版本

        Returns:
            Tuple[str, Dict[str, Any]]: 状态信息和版本信息
        """
        try:
            if version == "auto":
                # 自动检测
                version_info = self.version_manager.detect_and_set_version(force_refresh=True)
                status = f"自动检测到版本: {version_info.version.value}"
            else:
                # 手动切换
                target_version = DDSPSVCVersion.V6_1 if version == "6.1" else DDSPSVCVersion.V6_3
                success = self.version_manager.switch_version(target_version)

                if success:
                    status = f"成功切换到版本: {version}"
                else:
                    status = f"版本切换失败: {version}"

            # 更新统一接口
            self.current_unified = DDSPSVCUnified(version=version)

            return status, self._get_version_info_display()

        except Exception as e:
            error_msg = f"版本切换失败: {e}"
            logger.error(error_msg)
            return error_msg, self._get_version_info_display()

    def _switch_version_simple(self, version: str) -> str:
        """简单版本切换（用于其他标签页）

        Args:
            version: 目标版本

        Returns:
            str: 状态信息
        """
        status, _ = self._switch_version(version)
        return status

    def _refresh_info(self) -> Tuple[Dict[str, Any], str]:
        """刷新版本信息

        Returns:
            Tuple[Dict[str, Any], str]: 版本信息和特性对比
        """
        return self._get_version_info_display(), self._get_features_comparison()

    def _set_ddsp_path(self, path: str) -> Tuple[str, Dict[str, Any]]:
        """设置DDSP-SVC路径

        Args:
            path: DDSP-SVC项目路径

        Returns:
            Tuple[str, Dict[str, Any]]: 状态信息和版本信息
        """
        try:
            if not path.strip():
                return "路径为空，使用默认路径", self._get_version_info_display()

            ddsp_path = Path(path)
            if not ddsp_path.exists():
                return f"路径不存在: {path}", self._get_version_info_display()

            # 重新初始化版本管理器
            from ..integrations.version_manager import VersionManagerConfig
            config = VersionManagerConfig(ddsp_svc_path=ddsp_path)
            self.version_manager = get_version_manager(config)

            return f"成功设置路径: {path}", self._get_version_info_display()

        except Exception as e:
            error_msg = f"设置路径失败: {e}"
            logger.error(error_msg)
            return error_msg, self._get_version_info_display()

    def _clear_cache(self) -> str:
        """清理缓存

        Returns:
            str: 状态信息
        """
        try:
            self.version_manager.clear_cache()
            if self.current_unified:
                self.current_unified.clear_cache()
            return "缓存清理完成"
        except Exception as e:
            error_msg = f"清理缓存失败: {e}"
            logger.error(error_msg)
            return error_msg

    def _force_detect(self) -> Tuple[str, Dict[str, Any]]:
        """强制重新检测版本

        Returns:
            Tuple[str, Dict[str, Any]]: 状态信息和版本信息
        """
        try:
            version_info = self.version_manager.detect_and_set_version(force_refresh=True)
            status = f"重新检测完成，版本: {version_info.version.value}"
            return status, self._get_version_info_display()
        except Exception as e:
            error_msg = f"强制检测失败: {e}"
            logger.error(error_msg)
            return error_msg, self._get_version_info_display()

    def get_current_unified(self) -> Optional[DDSPSVCUnified]:
        """获取当前的统一接口实例

        Returns:
            Optional[DDSPSVCUnified]: 统一接口实例
        """
        if self.current_unified is None:
            try:
                self.current_unified = DDSPSVCUnified()
            except Exception as e:
                logger.error(f"创建统一接口失败: {e}")
                return None

        return self.current_unified


# 全局版本UI实例
_global_version_ui: Optional[DDSPSVCVersionUI] = None


def get_version_ui() -> DDSPSVCVersionUI:
    """获取全局版本UI实例

    Returns:
        DDSPSVCVersionUI: 版本UI实例
    """
    global _global_version_ui

    if _global_version_ui is None:
        _global_version_ui = DDSPSVCVersionUI()

    return _global_version_ui


def create_version_management_tab() -> gr.Tab:
    """创建版本管理标签页

    Returns:
        gr.Tab: 版本管理标签页
    """
    version_ui = get_version_ui()
    return version_ui.create_version_tab()


def create_version_selector() -> Tuple[gr.Dropdown, gr.Button, gr.Textbox]:
    """创建版本选择器

    Returns:
        Tuple[gr.Dropdown, gr.Button, gr.Textbox]: 版本选择器组件
    """
    version_ui = get_version_ui()
    return version_ui.create_version_selector()
