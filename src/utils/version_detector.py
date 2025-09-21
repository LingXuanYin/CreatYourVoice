"""DDSP-SVC版本检测工具

这个模块负责检测DDSP-SVC项目的版本，并提供版本相关的配置信息。
设计原则：
1. 简单直接 - 通过文件特征和Git信息检测版本
2. 容错性强 - 检测失败时提供合理的默认值
3. 缓存机制 - 避免重复检测
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DDSPSVCVersion(Enum):
    """DDSP-SVC版本枚举"""
    V6_1 = "6.1"
    V6_3 = "6.3"
    UNKNOWN = "unknown"


@dataclass
class VersionInfo:
    """版本信息"""
    version: DDSPSVCVersion
    branch: Optional[str]
    commit_hash: Optional[str]
    path: Path
    features: Dict[str, bool]


class DDSPSVCVersionDetector:
    """DDSP-SVC版本检测器

    通过多种方式检测DDSP-SVC的版本：
    1. Git分支信息
    2. 文件特征分析
    3. 代码签名检测
    """

    def __init__(self, ddsp_svc_path: Optional[Path] = None):
        """初始化版本检测器

        Args:
            ddsp_svc_path: DDSP-SVC项目路径，None表示自动检测
        """
        if ddsp_svc_path is None:
            # 自动检测DDSP-SVC路径
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            ddsp_svc_path = project_root / "DDSP-SVC"

        self.ddsp_svc_path = Path(ddsp_svc_path)
        self._version_cache: Optional[VersionInfo] = None

        logger.debug(f"DDSP-SVC路径: {self.ddsp_svc_path}")

    def detect_version(self, force_refresh: bool = False) -> VersionInfo:
        """检测DDSP-SVC版本

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            VersionInfo: 版本信息
        """
        if self._version_cache is not None and not force_refresh:
            return self._version_cache

        logger.info("开始检测DDSP-SVC版本")

        # 检查路径是否存在
        if not self.ddsp_svc_path.exists():
            logger.warning(f"DDSP-SVC路径不存在: {self.ddsp_svc_path}")
            return self._create_unknown_version()

        # 1. 尝试通过Git检测
        version_info = self._detect_by_git()
        if version_info.version != DDSPSVCVersion.UNKNOWN:
            self._version_cache = version_info
            return version_info

        # 2. 通过文件特征检测
        version_info = self._detect_by_file_features()
        if version_info.version != DDSPSVCVersion.UNKNOWN:
            self._version_cache = version_info
            return version_info

        # 3. 通过代码签名检测
        version_info = self._detect_by_code_signature()
        self._version_cache = version_info
        return version_info

    def _detect_by_git(self) -> VersionInfo:
        """通过Git信息检测版本"""
        try:
            # 获取当前分支
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.ddsp_svc_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                branch = result.stdout.strip()
                logger.debug(f"Git分支: {branch}")

                # 获取commit hash
                commit_result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.ddsp_svc_path,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                commit_hash = None
                if commit_result.returncode == 0:
                    commit_hash = commit_result.stdout.strip()[:8]

                # 根据分支名判断版本
                if branch == "6.1":
                    version = DDSPSVCVersion.V6_1
                elif branch == "6.3":
                    version = DDSPSVCVersion.V6_3
                else:
                    # 尝试检测是否在6.1或6.3分支的基础上
                    version = self._detect_version_by_branch_history()

                if version != DDSPSVCVersion.UNKNOWN:
                    features = self._detect_version_features(version)
                    return VersionInfo(
                        version=version,
                        branch=branch,
                        commit_hash=commit_hash,
                        path=self.ddsp_svc_path,
                        features=features
                    )

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Git检测失败: {e}")

        return self._create_unknown_version()

    def _detect_version_by_branch_history(self) -> DDSPSVCVersion:
        """通过分支历史检测版本"""
        try:
            # 检查是否包含6.3分支的特征提交
            result = subprocess.run(
                ["git", "log", "--oneline", "--grep=vocal_register", "-n", "1"],
                cwd=self.ddsp_svc_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                return DDSPSVCVersion.V6_3

            # 检查是否包含6.1分支的特征
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=2024-01-01", "-n", "10"],
                cwd=self.ddsp_svc_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                log_content = result.stdout.lower()
                if "6.1" in log_content or "v6.1" in log_content:
                    return DDSPSVCVersion.V6_1
                elif "6.3" in log_content or "v6.3" in log_content:
                    return DDSPSVCVersion.V6_3

        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.debug(f"分支历史检测失败: {e}")

        return DDSPSVCVersion.UNKNOWN

    def _detect_by_file_features(self) -> VersionInfo:
        """通过文件特征检测版本"""
        try:
            main_reflow_path = self.ddsp_svc_path / "main_reflow.py"
            if not main_reflow_path.exists():
                return self._create_unknown_version()

            with open(main_reflow_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查6.3版本特征
            if "vocal_register_shift_key" in content and "win_size" in content:
                version = DDSPSVCVersion.V6_3
            # 检查6.1版本特征
            elif "return_wav=True" in content and "vocal_register_shift_key" not in content:
                version = DDSPSVCVersion.V6_1
            else:
                return self._create_unknown_version()

            features = self._detect_version_features(version)
            return VersionInfo(
                version=version,
                branch=None,
                commit_hash=None,
                path=self.ddsp_svc_path,
                features=features
            )

        except Exception as e:
            logger.debug(f"文件特征检测失败: {e}")
            return self._create_unknown_version()

    def _detect_by_code_signature(self) -> VersionInfo:
        """通过代码签名检测版本"""
        try:
            # 检查Volume_Extractor的调用方式
            main_reflow_path = self.ddsp_svc_path / "main_reflow.py"
            if main_reflow_path.exists():
                with open(main_reflow_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 6.3版本: Volume_Extractor(hop_size, win_size)
                if "Volume_Extractor(hop_size, win_size)" in content:
                    version = DDSPSVCVersion.V6_3
                # 6.1版本: Volume_Extractor(hop_size)
                elif "Volume_Extractor(hop_size)" in content and "win_size" not in content:
                    version = DDSPSVCVersion.V6_1
                else:
                    version = DDSPSVCVersion.UNKNOWN

                features = self._detect_version_features(version)
                return VersionInfo(
                    version=version,
                    branch=None,
                    commit_hash=None,
                    path=self.ddsp_svc_path,
                    features=features
                )

        except Exception as e:
            logger.debug(f"代码签名检测失败: {e}")

        return self._create_unknown_version()

    def _detect_version_features(self, version: DDSPSVCVersion) -> Dict[str, bool]:
        """检测版本特性"""
        if version == DDSPSVCVersion.V6_1:
            return {
                "vocal_register_shift": False,
                "win_size_volume": False,
                "return_wav_direct": True,
                "mask_padding": True
            }
        elif version == DDSPSVCVersion.V6_3:
            return {
                "vocal_register_shift": True,
                "win_size_volume": True,
                "return_wav_direct": False,
                "mask_padding": False
            }
        else:
            return {}

    def _create_unknown_version(self) -> VersionInfo:
        """创建未知版本信息"""
        return VersionInfo(
            version=DDSPSVCVersion.UNKNOWN,
            branch=None,
            commit_hash=None,
            path=self.ddsp_svc_path,
            features={}
        )

    def get_version_config(self, version: DDSPSVCVersion) -> Dict[str, Any]:
        """获取版本配置"""
        if version == DDSPSVCVersion.V6_1:
            return {
                "volume_extractor_args": ["hop_size"],
                "mask_processing": "padding",
                "model_return_type": "wav",
                "supports_vocal_register": False,
                "default_t_start": 0.7
            }
        elif version == DDSPSVCVersion.V6_3:
            return {
                "volume_extractor_args": ["hop_size", "win_size"],
                "mask_processing": "upsample",
                "model_return_type": "mel",
                "supports_vocal_register": True,
                "default_t_start": 0.0
            }
        else:
            # 默认使用6.3配置
            return self.get_version_config(DDSPSVCVersion.V6_3)


# 全局版本检测器实例
_global_detector: Optional[DDSPSVCVersionDetector] = None


def get_ddsp_svc_version(ddsp_svc_path: Optional[Path] = None, force_refresh: bool = False) -> VersionInfo:
    """获取DDSP-SVC版本信息

    Args:
        ddsp_svc_path: DDSP-SVC项目路径
        force_refresh: 是否强制刷新缓存

    Returns:
        VersionInfo: 版本信息
    """
    global _global_detector

    if _global_detector is None or ddsp_svc_path is not None:
        _global_detector = DDSPSVCVersionDetector(ddsp_svc_path)

    return _global_detector.detect_version(force_refresh)


def get_version_config(version: DDSPSVCVersion) -> Dict[str, Any]:
    """获取版本配置

    Args:
        version: DDSP-SVC版本

    Returns:
        Dict[str, Any]: 版本配置
    """
    global _global_detector

    if _global_detector is None:
        _global_detector = DDSPSVCVersionDetector()

    return _global_detector.get_version_config(version)


def is_version_supported(version: DDSPSVCVersion) -> bool:
    """检查版本是否支持

    Args:
        version: DDSP-SVC版本

    Returns:
        bool: 是否支持
    """
    return version in [DDSPSVCVersion.V6_1, DDSPSVCVersion.V6_3]
