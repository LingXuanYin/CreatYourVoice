"""DDSP-SVC版本管理器

这个模块负责管理不同版本的DDSP-SVC，提供版本切换和配置管理功能。
设计原则：
1. 统一接口 - 对外提供一致的API，内部处理版本差异
2. 自动适配 - 根据检测到的版本自动选择合适的适配器
3. 配置隔离 - 不同版本的配置互不干扰
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Type, Union
from dataclasses import dataclass

from ..utils.version_detector import (
    DDSPSVCVersionDetector,
    DDSPSVCVersion,
    VersionInfo,
    get_ddsp_svc_version,
    get_version_config
)

logger = logging.getLogger(__name__)


@dataclass
class VersionManagerConfig:
    """版本管理器配置"""
    ddsp_svc_path: Optional[Path] = None
    preferred_version: Optional[DDSPSVCVersion] = None
    auto_switch: bool = True
    cache_adapters: bool = True


class DDSPSVCVersionManager:
    """DDSP-SVC版本管理器

    负责管理不同版本的DDSP-SVC适配器，提供统一的版本管理接口。
    """

    def __init__(self, config: Optional[VersionManagerConfig] = None):
        """初始化版本管理器

        Args:
            config: 版本管理器配置
        """
        self.config = config or VersionManagerConfig()

        # 版本检测器
        self.detector = DDSPSVCVersionDetector(self.config.ddsp_svc_path)

        # 适配器缓存
        self._adapter_cache: Dict[DDSPSVCVersion, Any] = {}

        # 当前版本信息
        self._current_version_info: Optional[VersionInfo] = None
        self._current_adapter: Optional[Any] = None

        logger.info("DDSP-SVC版本管理器初始化完成")

    def detect_and_set_version(self, force_refresh: bool = False) -> VersionInfo:
        """检测并设置当前版本

        Args:
            force_refresh: 是否强制刷新版本检测

        Returns:
            VersionInfo: 检测到的版本信息
        """
        version_info = self.detector.detect_version(force_refresh)

        if version_info.version == DDSPSVCVersion.UNKNOWN:
            logger.warning("无法检测DDSP-SVC版本，将使用默认版本6.3")
            # 创建默认版本信息
            version_info = VersionInfo(
                version=DDSPSVCVersion.V6_3,
                branch=None,
                commit_hash=None,
                path=self.detector.ddsp_svc_path,
                features=self.detector._detect_version_features(DDSPSVCVersion.V6_3)
            )

        self._current_version_info = version_info

        logger.info(f"检测到DDSP-SVC版本: {version_info.version.value}")
        if version_info.branch:
            logger.info(f"Git分支: {version_info.branch}")
        if version_info.commit_hash:
            logger.info(f"提交哈希: {version_info.commit_hash}")

        return version_info

    def get_current_version(self) -> Optional[VersionInfo]:
        """获取当前版本信息

        Returns:
            Optional[VersionInfo]: 当前版本信息，如果未检测则返回None
        """
        return self._current_version_info

    def get_adapter(self, version: Optional[DDSPSVCVersion] = None) -> Any:
        """获取指定版本的适配器

        Args:
            version: 指定版本，None表示使用当前版本

        Returns:
            适配器实例

        Raises:
            ValueError: 版本不支持或未检测到版本
        """
        if version is None:
            if self._current_version_info is None:
                self.detect_and_set_version()
            version = self._current_version_info.version

        if version == DDSPSVCVersion.UNKNOWN:
            raise ValueError("无法获取未知版本的适配器")

        # 检查缓存
        if self.config.cache_adapters and version in self._adapter_cache:
            logger.debug(f"从缓存获取{version.value}版本适配器")
            return self._adapter_cache[version]

        # 创建适配器
        adapter = self._create_adapter(version)

        # 缓存适配器
        if self.config.cache_adapters:
            self._adapter_cache[version] = adapter

        return adapter

    def _create_adapter(self, version: DDSPSVCVersion) -> Any:
        """创建指定版本的适配器

        Args:
            version: DDSP-SVC版本

        Returns:
            适配器实例

        Raises:
            ValueError: 版本不支持
        """
        logger.info(f"创建{version.value}版本适配器")

        if version == DDSPSVCVersion.V6_1:
            from .ddsp_svc_v61 import DDSPSVCv61Adapter
            return DDSPSVCv61Adapter(
                ddsp_svc_path=self.detector.ddsp_svc_path,
                version_info=self._current_version_info
            )
        elif version == DDSPSVCVersion.V6_3:
            from .ddsp_svc_v63 import DDSPSVCv63Adapter
            return DDSPSVCv63Adapter(
                ddsp_svc_path=self.detector.ddsp_svc_path,
                version_info=self._current_version_info
            )
        else:
            raise ValueError(f"不支持的DDSP-SVC版本: {version.value}")

    def switch_version(self, version: DDSPSVCVersion) -> bool:
        """切换到指定版本

        Args:
            version: 目标版本

        Returns:
            bool: 切换是否成功
        """
        try:
            logger.info(f"尝试切换到DDSP-SVC版本: {version.value}")

            # 检查版本是否支持
            if version == DDSPSVCVersion.UNKNOWN:
                logger.error("无法切换到未知版本")
                return False

            # 尝试通过Git切换分支
            if self._switch_git_branch(version):
                # 重新检测版本
                self.detect_and_set_version(force_refresh=True)

                # 验证切换是否成功
                if self._current_version_info and self._current_version_info.version == version:
                    logger.info(f"成功切换到版本: {version.value}")
                    return True
                else:
                    logger.warning(f"Git分支切换成功，但版本检测结果不匹配")

            # Git切换失败，尝试手动设置版本
            logger.warning("Git切换失败，尝试手动设置版本")
            self._current_version_info = VersionInfo(
                version=version,
                branch=None,
                commit_hash=None,
                path=self.detector.ddsp_svc_path,
                features=self.detector._detect_version_features(version)
            )

            return True

        except Exception as e:
            logger.error(f"版本切换失败: {e}")
            return False

    def _switch_git_branch(self, version: DDSPSVCVersion) -> bool:
        """通过Git切换分支

        Args:
            version: 目标版本

        Returns:
            bool: 切换是否成功
        """
        try:
            import subprocess

            branch_name = version.value

            # 检查分支是否存在
            result = subprocess.run(
                ["git", "branch", "-a"],
                cwd=self.detector.ddsp_svc_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.debug("无法获取Git分支列表")
                return False

            branches = result.stdout
            if f"origin/{branch_name}" not in branches and branch_name not in branches:
                logger.debug(f"分支{branch_name}不存在")
                return False

            # 切换分支
            result = subprocess.run(
                ["git", "checkout", branch_name],
                cwd=self.detector.ddsp_svc_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"成功切换Git分支到: {branch_name}")
                return True
            else:
                logger.debug(f"Git分支切换失败: {result.stderr}")
                return False

        except Exception as e:
            logger.debug(f"Git分支切换异常: {e}")
            return False

    def get_supported_versions(self) -> list[DDSPSVCVersion]:
        """获取支持的版本列表

        Returns:
            list[DDSPSVCVersion]: 支持的版本列表
        """
        return [DDSPSVCVersion.V6_1, DDSPSVCVersion.V6_3]

    def get_version_config(self, version: Optional[DDSPSVCVersion] = None) -> Dict[str, Any]:
        """获取版本配置

        Args:
            version: 指定版本，None表示使用当前版本

        Returns:
            Dict[str, Any]: 版本配置
        """
        if version is None:
            if self._current_version_info is None:
                self.detect_and_set_version()
            version = self._current_version_info.version

        return get_version_config(version)

    def clear_cache(self) -> None:
        """清理适配器缓存"""
        logger.info("清理版本适配器缓存")

        # 清理适配器
        for adapter in self._adapter_cache.values():
            if hasattr(adapter, 'clear_cache'):
                adapter.clear_cache()

        self._adapter_cache.clear()
        self._current_adapter = None

    def get_version_summary(self) -> Dict[str, Any]:
        """获取版本摘要信息

        Returns:
            Dict[str, Any]: 版本摘要
        """
        if self._current_version_info is None:
            self.detect_and_set_version()

        version_info = self._current_version_info
        config = self.get_version_config()

        return {
            "version": version_info.version.value,
            "branch": version_info.branch,
            "commit_hash": version_info.commit_hash,
            "path": str(version_info.path),
            "features": version_info.features,
            "config": config,
            "supported_versions": [v.value for v in self.get_supported_versions()],
            "cache_enabled": self.config.cache_adapters,
            "cached_adapters": list(self._adapter_cache.keys())
        }


# 全局版本管理器实例
_global_manager: Optional[DDSPSVCVersionManager] = None


def get_version_manager(config: Optional[VersionManagerConfig] = None) -> DDSPSVCVersionManager:
    """获取全局版本管理器实例

    Args:
        config: 版本管理器配置

    Returns:
        DDSPSVCVersionManager: 版本管理器实例
    """
    global _global_manager

    if _global_manager is None or config is not None:
        _global_manager = DDSPSVCVersionManager(config)

    return _global_manager


def get_current_adapter() -> Any:
    """获取当前版本的适配器

    Returns:
        当前版本的适配器实例
    """
    manager = get_version_manager()
    return manager.get_adapter()


def switch_ddsp_svc_version(version: DDSPSVCVersion) -> bool:
    """切换DDSP-SVC版本

    Args:
        version: 目标版本

    Returns:
        bool: 切换是否成功
    """
    manager = get_version_manager()
    return manager.switch_version(version)
