"""核心数据模型定义

这个模块定义了音色系统的核心数据结构。
按照Linus的哲学：好的数据结构比复杂的算法更重要。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import uuid
from datetime import datetime


@dataclass
class DDSPSVCConfig:
    """DDSP-SVC配置参数

    包含DDSP-SVC模型推理所需的所有参数。
    设计原则：简单直接，避免嵌套复杂性。
    """
    model_path: str
    config_path: str
    speaker_id: int = 0
    f0_predictor: str = "rmvpe"
    f0_min: float = 50.0
    f0_max: float = 1100.0
    threhold: float = -60.0
    spk_mix_dict: Optional[Dict[str, float]] = None
    use_spk_mix: bool = False

    def __post_init__(self) -> None:
        """初始化后处理，确保数据一致性"""
        if self.spk_mix_dict is None:
            self.spk_mix_dict = {}

        # 如果有混合说话人配置，自动启用混合模式
        if self.spk_mix_dict:
            self.use_spk_mix = True


@dataclass
class IndexTTSConfig:
    """IndexTTS配置参数

    包含IndexTTS v2情感语音合成所需的参数。
    """
    model_path: str
    config_path: str
    speaker_name: str = "default"
    emotion_reference: Optional[str] = None
    emotion_strength: float = 1.0
    speed: float = 1.0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self) -> None:
        """参数范围检查"""
        self.emotion_strength = max(0.0, min(2.0, self.emotion_strength))
        self.speed = max(0.1, min(3.0, self.speed))
        self.temperature = max(0.1, min(2.0, self.temperature))


@dataclass
class WeightInfo:
    """权重信息

    用于音色融合的权重计算结果。
    设计原则：权重总和始终为1.0，避免特殊情况处理。
    """
    speaker_weights: Dict[str, float] = field(default_factory=dict)
    total_weight: float = 1.0

    def __post_init__(self) -> None:
        """自动归一化权重"""
        if self.speaker_weights:
            total = sum(self.speaker_weights.values())
            if total > 0:
                self.speaker_weights = {
                    k: v / total for k, v in self.speaker_weights.items()
                }
                self.total_weight = 1.0

    def add_speaker(self, speaker_id: str, weight: float) -> None:
        """添加说话人权重并重新归一化"""
        self.speaker_weights[speaker_id] = max(0.0, weight)
        self.__post_init__()  # 重新归一化

    def remove_speaker(self, speaker_id: str) -> None:
        """移除说话人权重并重新归一化"""
        if speaker_id in self.speaker_weights:
            del self.speaker_weights[speaker_id]
            self.__post_init__()  # 重新归一化


@dataclass
class VoiceConfig:
    """音色配置主类

    这是系统的核心数据结构，包含一个音色的完整配置。
    设计原则：
    1. 单一数据源 - 所有音色信息都在这里
    2. 不可变ID - 创建后ID不可更改
    3. 版本控制 - 支持配置演进
    """
    name: str
    ddsp_config: DDSPSVCConfig
    index_tts_config: IndexTTSConfig
    weight_info: WeightInfo = field(default_factory=WeightInfo)

    # 元数据
    voice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 继承信息
    parent_voice_ids: List[str] = field(default_factory=list)
    fusion_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """确保数据一致性"""
        # 确保时间戳正确
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.now()
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.now()

    def update_timestamp(self) -> None:
        """更新修改时间戳"""
        self.updated_at = datetime.now()

    def add_parent_voice(self, parent_id: str, weight: float) -> None:
        """添加父音色信息"""
        if parent_id not in self.parent_voice_ids:
            self.parent_voice_ids.append(parent_id)
        self.fusion_weights[parent_id] = weight
        self.update_timestamp()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "parent_voice_ids": self.parent_voice_ids,
            "fusion_weights": self.fusion_weights,
            "ddsp_config": {
                "model_path": self.ddsp_config.model_path,
                "config_path": self.ddsp_config.config_path,
                "speaker_id": self.ddsp_config.speaker_id,
                "f0_predictor": self.ddsp_config.f0_predictor,
                "f0_min": self.ddsp_config.f0_min,
                "f0_max": self.ddsp_config.f0_max,
                "threhold": self.ddsp_config.threhold,
                "spk_mix_dict": self.ddsp_config.spk_mix_dict,
                "use_spk_mix": self.ddsp_config.use_spk_mix,
            },
            "index_tts_config": {
                "model_path": self.index_tts_config.model_path,
                "config_path": self.index_tts_config.config_path,
                "speaker_name": self.index_tts_config.speaker_name,
                "emotion_reference": self.index_tts_config.emotion_reference,
                "emotion_strength": self.index_tts_config.emotion_strength,
                "speed": self.index_tts_config.speed,
                "temperature": self.index_tts_config.temperature,
                "top_k": self.index_tts_config.top_k,
                "top_p": self.index_tts_config.top_p,
            },
            "weight_info": {
                "speaker_weights": self.weight_info.speaker_weights,
                "total_weight": self.weight_info.total_weight,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceConfig":
        """从字典创建VoiceConfig实例"""
        # 解析时间戳
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])

        # 创建子配置对象
        ddsp_config = DDSPSVCConfig(**data["ddsp_config"])
        index_tts_config = IndexTTSConfig(**data["index_tts_config"])
        weight_info = WeightInfo(**data["weight_info"])

        return cls(
            voice_id=data["voice_id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            created_at=created_at,
            updated_at=updated_at,
            tags=data.get("tags", []),
            parent_voice_ids=data.get("parent_voice_ids", []),
            fusion_weights=data.get("fusion_weights", {}),
            ddsp_config=ddsp_config,
            index_tts_config=index_tts_config,
            weight_info=weight_info,
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "VoiceConfig":
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class VoicePreset:
    """音色预设

    用于快速创建常用音色配置的模板。
    """
    name: str
    description: str
    ddsp_template: Dict[str, Any]
    index_tts_template: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def create_voice_config(
        self,
        voice_name: str,
        model_paths: Dict[str, Dict[str, Any]]
    ) -> VoiceConfig:
        """基于预设创建音色配置"""
        # 更新模板中的模型路径
        ddsp_config_data = self.ddsp_template.copy()
        ddsp_paths = model_paths.get("ddsp", {})
        if isinstance(ddsp_paths, dict):
            ddsp_config_data.update(ddsp_paths)

        index_tts_config_data = self.index_tts_template.copy()
        index_tts_paths = model_paths.get("index_tts", {})
        if isinstance(index_tts_paths, dict):
            index_tts_config_data.update(index_tts_paths)

        return VoiceConfig(
            name=voice_name,
            description=f"基于预设 '{self.name}' 创建",
            tags=self.tags.copy(),
            ddsp_config=DDSPSVCConfig(**ddsp_config_data),
            index_tts_config=IndexTTSConfig(**index_tts_config_data),
        )
