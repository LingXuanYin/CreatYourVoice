"""音色预设管理器

这个模块负责管理音色标签预设和说话人配置。
设计原则：
1. 数据驱动 - 所有预设通过配置文件定义
2. 缓存优化 - 避免重复加载配置文件
3. 类型安全 - 使用数据类确保类型正确性
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeakerInfo:
    """说话人信息"""
    id: str
    name: str
    model_path: str
    config_path: str
    speaker_id: int = 0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class VoiceTagInfo:
    """音色标签信息"""
    name: str
    description: str
    audio_file: str
    f0_range: List[float]
    speakers: List[SpeakerInfo]
    default_ddsp_params: Dict[str, Any]


@dataclass
class SpeakerCombination:
    """说话人组合预设"""
    name: str
    description: str
    speakers: List[Dict[str, Union[str, float]]]  # [{"id": str, "weight": float}]


class VoicePresetManagerError(Exception):
    """音色预设管理器异常基类"""
    pass


class ConfigLoadError(VoicePresetManagerError):
    """配置加载异常"""
    pass


class VoicePresetManager:
    """音色预设管理器

    负责加载和管理音色标签预设、说话人配置等。
    """

    def __init__(self, presets_dir: Union[str, Path] = "src/data/presets"):
        """初始化预设管理器

        Args:
            presets_dir: 预设配置目录
        """
        self.presets_dir = Path(presets_dir)
        self.voice_tags_file = self.presets_dir / "voice_tags.yaml"
        self.speakers_file = self.presets_dir / "speakers.yaml"

        # 缓存
        self._voice_tags: Optional[Dict[str, VoiceTagInfo]] = None
        self._speakers: Optional[Dict[str, SpeakerInfo]] = None
        self._speaker_combinations: Optional[Dict[str, SpeakerCombination]] = None
        self._test_texts: Optional[List[str]] = None

        logger.info(f"音色预设管理器初始化完成，预设目录: {self.presets_dir}")

    def _load_voice_tags(self) -> None:
        """加载音色标签配置"""
        if self._voice_tags is not None:
            return

        try:
            if not self.voice_tags_file.exists():
                raise ConfigLoadError(f"音色标签配置文件不存在: {self.voice_tags_file}")

            with open(self.voice_tags_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self._voice_tags = {}
            self._test_texts = data.get('test_texts', [])

            # 加载普通音色标签
            voice_tags_data = data.get('voice_tags', {})
            for tag_name, tag_info in voice_tags_data.items():
                speakers = []
                for speaker_data in tag_info.get('speakers', []):
                    speaker = SpeakerInfo(
                        id=speaker_data['id'],
                        name=speaker_data['name'],
                        model_path="",  # 将从speakers.yaml中获取
                        config_path="",  # 将从speakers.yaml中获取
                        weight=speaker_data.get('weight', 1.0)
                    )
                    speakers.append(speaker)

                voice_tag = VoiceTagInfo(
                    name=tag_name,
                    description=tag_info['description'],
                    audio_file=tag_info['audio_file'],
                    f0_range=tag_info['f0_range'],
                    speakers=speakers,
                    default_ddsp_params=tag_info.get('default_ddsp_params', {})
                )
                self._voice_tags[tag_name] = voice_tag

            # 加载特殊音色标签
            special_tags_data = data.get('special_tags', {})
            for tag_name, tag_info in special_tags_data.items():
                speakers = []
                for speaker_data in tag_info.get('speakers', []):
                    speaker = SpeakerInfo(
                        id=speaker_data['id'],
                        name=speaker_data['name'],
                        model_path="",
                        config_path="",
                        weight=speaker_data.get('weight', 1.0)
                    )
                    speakers.append(speaker)

                voice_tag = VoiceTagInfo(
                    name=tag_name,
                    description=tag_info['description'],
                    audio_file=tag_info['audio_file'],
                    f0_range=tag_info['f0_range'],
                    speakers=speakers,
                    default_ddsp_params=tag_info.get('default_ddsp_params', {})
                )
                self._voice_tags[tag_name] = voice_tag

            logger.info(f"加载了 {len(self._voice_tags)} 个音色标签")

        except Exception as e:
            raise ConfigLoadError(f"加载音色标签配置失败: {e}")

    def _load_speakers(self) -> None:
        """加载说话人配置"""
        if self._speakers is not None:
            return

        try:
            if not self.speakers_file.exists():
                raise ConfigLoadError(f"说话人配置文件不存在: {self.speakers_file}")

            with open(self.speakers_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            self._speakers = {}
            self._speaker_combinations = {}

            # 加载说话人信息
            speakers_data = data.get('speakers', {})
            for speaker_id, speaker_info in speakers_data.items():
                speaker = SpeakerInfo(
                    id=speaker_id,
                    name=speaker_info['name'],
                    model_path=speaker_info['model_path'],
                    config_path=speaker_info['config_path'],
                    speaker_id=speaker_info.get('speaker_id', 0),
                    description=speaker_info.get('description', ''),
                    tags=speaker_info.get('tags', [])
                )
                self._speakers[speaker_id] = speaker

            # 加载说话人组合预设
            combinations_data = data.get('speaker_combinations', {})
            for combo_id, combo_info in combinations_data.items():
                combination = SpeakerCombination(
                    name=combo_info['name'],
                    description=combo_info['description'],
                    speakers=combo_info['speakers']
                )
                self._speaker_combinations[combo_id] = combination

            logger.info(f"加载了 {len(self._speakers)} 个说话人和 {len(self._speaker_combinations)} 个组合预设")

        except Exception as e:
            raise ConfigLoadError(f"加载说话人配置失败: {e}")

    def get_voice_tags(self) -> Dict[str, VoiceTagInfo]:
        """获取所有音色标签

        Returns:
            音色标签字典
        """
        self._load_voice_tags()
        self._load_speakers()

        # 确保缓存已加载
        if self._voice_tags is None or self._speakers is None:
            return {}

        # 更新音色标签中的说话人信息
        for tag_info in self._voice_tags.values():
            for speaker in tag_info.speakers:
                if speaker.id in self._speakers:
                    speaker_info = self._speakers[speaker.id]
                    speaker.model_path = speaker_info.model_path
                    speaker.config_path = speaker_info.config_path
                    speaker.speaker_id = speaker_info.speaker_id
                    speaker.description = speaker_info.description
                    speaker.tags = speaker_info.tags

        return self._voice_tags

    def get_voice_tag(self, tag_name: str) -> Optional[VoiceTagInfo]:
        """获取指定音色标签

        Args:
            tag_name: 标签名称

        Returns:
            音色标签信息，如果不存在则返回None
        """
        voice_tags = self.get_voice_tags()
        return voice_tags.get(tag_name)

    def get_speakers(self) -> Dict[str, SpeakerInfo]:
        """获取所有说话人信息

        Returns:
            说话人信息字典
        """
        self._load_speakers()
        return self._speakers or {}

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerInfo]:
        """获取指定说话人信息

        Args:
            speaker_id: 说话人ID

        Returns:
            说话人信息，如果不存在则返回None
        """
        speakers = self.get_speakers()
        return speakers.get(speaker_id)

    def get_speaker_combinations(self) -> Dict[str, SpeakerCombination]:
        """获取说话人组合预设

        Returns:
            说话人组合字典
        """
        self._load_speakers()
        return self._speaker_combinations or {}

    def get_speakers_by_tag(self, tag_name: str) -> List[SpeakerInfo]:
        """根据音色标签获取对应的说话人列表

        Args:
            tag_name: 音色标签名称

        Returns:
            说话人信息列表
        """
        voice_tag = self.get_voice_tag(tag_name)
        if voice_tag:
            return voice_tag.speakers
        return []

    def get_test_texts(self) -> List[str]:
        """获取测试文本列表

        Returns:
            测试文本列表
        """
        self._load_voice_tags()
        return self._test_texts or []

    def get_audio_file_path(self, tag_name: str) -> Optional[Path]:
        """获取音色标签对应的音频文件路径

        Args:
            tag_name: 音色标签名称

        Returns:
            音频文件路径，如果不存在则返回None
        """
        voice_tag = self.get_voice_tag(tag_name)
        if voice_tag:
            # 相对于预设目录的路径
            audio_path = self.presets_dir.parent / voice_tag.audio_file
            return audio_path
        return None

    def validate_speaker_paths(self, speaker_id: str) -> bool:
        """验证说话人模型文件是否存在

        Args:
            speaker_id: 说话人ID

        Returns:
            文件是否存在
        """
        speaker = self.get_speaker(speaker_id)
        if not speaker:
            return False

        model_path = Path(speaker.model_path)
        config_path = Path(speaker.config_path)

        return model_path.exists() and config_path.exists()

    def get_default_ddsp_params(self, tag_name: str) -> Dict[str, Any]:
        """获取音色标签的默认DDSP参数

        Args:
            tag_name: 音色标签名称

        Returns:
            默认DDSP参数字典
        """
        voice_tag = self.get_voice_tag(tag_name)
        if voice_tag:
            return voice_tag.default_ddsp_params
        return {}

    def create_speaker_mix_dict(self, tag_name: str, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """创建说话人混合字典

        Args:
            tag_name: 音色标签名称
            custom_weights: 自定义权重字典，格式为 {speaker_id: weight}

        Returns:
            归一化后的说话人权重字典
        """
        speakers = self.get_speakers_by_tag(tag_name)
        if not speakers:
            return {}

        # 使用自定义权重或默认权重
        weights = {}
        for speaker in speakers:
            if custom_weights and speaker.id in custom_weights:
                weights[speaker.id] = custom_weights[speaker.id]
            else:
                weights[speaker.id] = speaker.weight

        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def clear_cache(self) -> None:
        """清理缓存"""
        self._voice_tags = None
        self._speakers = None
        self._speaker_combinations = None
        self._test_texts = None
        logger.info("预设管理器缓存已清理")

    def reload_config(self) -> None:
        """重新加载配置"""
        self.clear_cache()
        self.get_voice_tags()  # 触发重新加载
        logger.info("预设配置已重新加载")
