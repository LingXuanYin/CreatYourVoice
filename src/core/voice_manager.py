"""音色管理器

这个模块负责音色的保存、加载、删除和搜索。
设计原则：
1. 单一职责 - 只管理音色配置，不处理推理
2. 简单接口 - 避免复杂的查询语法
3. 原子操作 - 每个操作要么成功要么失败，不留中间状态
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime
import logging

from .models import VoiceConfig, VoicePreset
from .weight_calculator import WeightCalculator

logger = logging.getLogger(__name__)


class VoiceManagerError(Exception):
    """音色管理器异常基类"""
    pass


class VoiceNotFoundError(VoiceManagerError):
    """音色未找到异常"""
    pass


class VoiceAlreadyExistsError(VoiceManagerError):
    """音色已存在异常"""
    pass


class VoiceManager:
    """音色管理器

    负责音色配置的持久化存储和管理。
    存储结构：
    voices_dir/
    ├── voice_id_1/
    │   ├── config.json
    │   └── metadata.json
    ├── voice_id_2/
    │   └── config.json
    └── presets/
        └── preset_name.json
    """

    def __init__(self, voices_dir: Union[str, Path] = "voices"):
        """初始化音色管理器

        Args:
            voices_dir: 音色存储目录
        """
        self.voices_dir = Path(voices_dir)
        self.presets_dir = self.voices_dir / "presets"

        # 创建目录
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._voice_cache: Dict[str, VoiceConfig] = {}
        self._cache_dirty = True

        logger.info(f"音色管理器初始化完成，存储目录: {self.voices_dir}")

    def _get_voice_dir(self, voice_id: str) -> Path:
        """获取音色存储目录"""
        return self.voices_dir / voice_id

    def _get_config_path(self, voice_id: str) -> Path:
        """获取音色配置文件路径"""
        return self._get_voice_dir(voice_id) / "config.json"

    def _refresh_cache(self) -> None:
        """刷新内存缓存"""
        if not self._cache_dirty:
            return

        self._voice_cache.clear()

        for voice_dir in self.voices_dir.iterdir():
            if not voice_dir.is_dir() or voice_dir.name == "presets":
                continue

            config_path = voice_dir / "config.json"
            if config_path.exists():
                try:
                    voice_config = VoiceConfig.load_from_file(config_path)
                    self._voice_cache[voice_config.voice_id] = voice_config
                except Exception as e:
                    logger.warning(f"加载音色配置失败: {config_path}, 错误: {e}")

        self._cache_dirty = False
        logger.debug(f"缓存刷新完成，加载了 {len(self._voice_cache)} 个音色")

    def save_voice(self, voice_config: VoiceConfig, overwrite: bool = False) -> None:
        """保存音色配置

        Args:
            voice_config: 音色配置
            overwrite: 是否覆盖已存在的音色

        Raises:
            VoiceAlreadyExistsError: 音色已存在且不允许覆盖
        """
        voice_dir = self._get_voice_dir(voice_config.voice_id)
        config_path = self._get_config_path(voice_config.voice_id)

        # 检查是否已存在
        if config_path.exists() and not overwrite:
            raise VoiceAlreadyExistsError(
                f"音色已存在: {voice_config.voice_id} ({voice_config.name})"
            )

        # 创建目录
        voice_dir.mkdir(parents=True, exist_ok=True)

        # 更新时间戳
        voice_config.update_timestamp()

        # 保存配置
        voice_config.save_to_file(config_path)

        # 更新缓存
        self._voice_cache[voice_config.voice_id] = voice_config

        logger.info(f"音色保存成功: {voice_config.name} ({voice_config.voice_id})")

    def load_voice(self, voice_id: str) -> VoiceConfig:
        """加载音色配置

        Args:
            voice_id: 音色ID

        Returns:
            音色配置

        Raises:
            VoiceNotFoundError: 音色不存在
        """
        # 先检查缓存
        self._refresh_cache()
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        # 从文件加载
        config_path = self._get_config_path(voice_id)
        if not config_path.exists():
            raise VoiceNotFoundError(f"音色不存在: {voice_id}")

        try:
            voice_config = VoiceConfig.load_from_file(config_path)
            self._voice_cache[voice_id] = voice_config
            return voice_config
        except Exception as e:
            raise VoiceManagerError(f"加载音色配置失败: {voice_id}, 错误: {e}")

    def delete_voice(self, voice_id: str) -> None:
        """删除音色

        Args:
            voice_id: 音色ID

        Raises:
            VoiceNotFoundError: 音色不存在
        """
        voice_dir = self._get_voice_dir(voice_id)
        if not voice_dir.exists():
            raise VoiceNotFoundError(f"音色不存在: {voice_id}")

        # 删除目录
        shutil.rmtree(voice_dir)

        # 从缓存中移除
        self._voice_cache.pop(voice_id, None)

        logger.info(f"音色删除成功: {voice_id}")

    def list_voices(self) -> List[VoiceConfig]:
        """列出所有音色

        Returns:
            音色配置列表
        """
        self._refresh_cache()
        return list(self._voice_cache.values())

    def search_voices(
        self,
        name_pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
        has_parent: Optional[bool] = None
    ) -> List[VoiceConfig]:
        """搜索音色

        Args:
            name_pattern: 名称模式（支持部分匹配）
            tags: 标签列表（包含任一标签即匹配）
            created_after: 创建时间下限
            created_before: 创建时间上限
            has_parent: 是否有父音色

        Returns:
            匹配的音色列表
        """
        voices = self.list_voices()
        results = []

        for voice in voices:
            # 名称匹配
            if name_pattern and name_pattern.lower() not in voice.name.lower():
                continue

            # 标签匹配
            if tags and not any(tag in voice.tags for tag in tags):
                continue

            # 时间范围匹配
            if created_after and voice.created_at < created_after:
                continue
            if created_before and voice.created_at > created_before:
                continue

            # 父音色匹配
            if has_parent is not None:
                has_parents = bool(voice.parent_voice_ids)
                if has_parent != has_parents:
                    continue

            results.append(voice)

        return results

    def get_voice_by_name(self, name: str) -> Optional[VoiceConfig]:
        """根据名称获取音色（精确匹配）

        Args:
            name: 音色名称

        Returns:
            音色配置，如果不存在则返回None
        """
        voices = self.list_voices()
        for voice in voices:
            if voice.name == name:
                return voice
        return None

    def duplicate_voice(
        self,
        source_voice_id: str,
        new_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> VoiceConfig:
        """复制音色

        Args:
            source_voice_id: 源音色ID
            new_name: 新音色名称
            modifications: 要修改的配置项

        Returns:
            新的音色配置
        """
        source_voice = self.load_voice(source_voice_id)

        # 创建副本
        new_voice_data = source_voice.to_dict()
        new_voice_data["voice_id"] = str(uuid.uuid4())
        new_voice_data["name"] = new_name
        new_voice_data["created_at"] = datetime.now().isoformat()
        new_voice_data["updated_at"] = datetime.now().isoformat()
        new_voice_data["parent_voice_ids"] = [source_voice_id]

        # 应用修改
        if modifications:
            self._apply_modifications(new_voice_data, modifications)

        new_voice = VoiceConfig.from_dict(new_voice_data)
        self.save_voice(new_voice)

        return new_voice

    def create_fusion_voice(
        self,
        name: str,
        source_voices: Dict[str, float],
        description: str = ""
    ) -> VoiceConfig:
        """创建融合音色

        Args:
            name: 新音色名称
            source_voices: 源音色ID到权重的映射
            description: 描述

        Returns:
            融合后的音色配置
        """
        if not source_voices:
            raise ValueError("至少需要一个源音色")

        # 归一化权重
        weight_result = WeightCalculator.normalize_weights(source_voices)

        # 加载源音色
        source_configs = {}
        for voice_id in source_voices.keys():
            source_configs[voice_id] = self.load_voice(voice_id)

        # 选择第一个音色作为基础模板
        base_voice_id = next(iter(source_configs.keys()))
        base_config = source_configs[base_voice_id]

        # 创建融合配置
        fusion_data = base_config.to_dict()
        fusion_data["voice_id"] = str(uuid.uuid4())
        fusion_data["name"] = name
        fusion_data["description"] = description or f"融合音色，基于 {len(source_voices)} 个源音色"
        fusion_data["created_at"] = datetime.now().isoformat()
        fusion_data["updated_at"] = datetime.now().isoformat()
        fusion_data["parent_voice_ids"] = list(source_voices.keys())
        fusion_data["fusion_weights"] = weight_result.normalized_weights

        # 更新DDSP配置中的混合权重
        fusion_data["ddsp_config"]["spk_mix_dict"] = weight_result.normalized_weights
        fusion_data["ddsp_config"]["use_spk_mix"] = True

        fusion_voice = VoiceConfig.from_dict(fusion_data)
        self.save_voice(fusion_voice)

        logger.info(f"融合音色创建成功: {name}, 源音色: {list(source_voices.keys())}")
        return fusion_voice

    def export_voice(self, voice_id: str, export_path: Union[str, Path]) -> None:
        """导出音色配置

        Args:
            voice_id: 音色ID
            export_path: 导出路径
        """
        voice_config = self.load_voice(voice_id)
        export_path = Path(export_path)

        if export_path.is_dir():
            export_path = export_path / f"{voice_config.name}_{voice_id}.json"

        voice_config.save_to_file(export_path)
        logger.info(f"音色导出成功: {export_path}")

    def import_voice(self, import_path: Union[str, Path]) -> VoiceConfig:
        """导入音色配置

        Args:
            import_path: 导入路径

        Returns:
            导入的音色配置
        """
        voice_config = VoiceConfig.load_from_file(import_path)

        # 检查是否已存在同ID的音色
        try:
            existing = self.load_voice(voice_config.voice_id)
            logger.warning(f"音色ID已存在，将覆盖: {voice_config.voice_id}")
            self.save_voice(voice_config, overwrite=True)
        except VoiceNotFoundError:
            self.save_voice(voice_config)

        logger.info(f"音色导入成功: {voice_config.name}")
        return voice_config

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        voices = self.list_voices()

        if not voices:
            return {
                "total_voices": 0,
                "fusion_voices": 0,
                "original_voices": 0,
                "most_used_tags": [],
                "creation_timeline": {}
            }

        # 基本统计
        total = len(voices)
        fusion_count = sum(1 for v in voices if v.parent_voice_ids)
        original_count = total - fusion_count

        # 标签统计
        tag_counts = {}
        for voice in voices:
            for tag in voice.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        most_used_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # 创建时间线
        timeline = {}
        for voice in voices:
            date_key = voice.created_at.strftime("%Y-%m")
            timeline[date_key] = timeline.get(date_key, 0) + 1

        return {
            "total_voices": total,
            "fusion_voices": fusion_count,
            "original_voices": original_count,
            "most_used_tags": most_used_tags,
            "creation_timeline": timeline
        }

    def _apply_modifications(self, voice_data: Dict[str, Any], modifications: Dict[str, Any]) -> None:
        """应用配置修改"""
        for key, value in modifications.items():
            if "." in key:
                # 支持嵌套键，如 "ddsp_config.speaker_id"
                keys = key.split(".")
                current = voice_data
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                voice_data[key] = value


# 导入uuid模块
import uuid
