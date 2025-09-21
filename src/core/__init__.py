"""核心模块

包含音色系统的核心功能和数据结构，包括音色继承和融合功能。
"""

from .models import (
    VoiceConfig,
    DDSPSVCConfig,
    IndexTTSConfig,
    WeightInfo,
    VoicePreset
)

from .weight_calculator import (
    WeightCalculator,
    WeightCalculationResult,
    WeightPreset
)

from .voice_manager import (
    VoiceManager,
    VoiceManagerError,
    VoiceNotFoundError,
    VoiceAlreadyExistsError
)

from .advanced_weight_calc import (
    AdvancedWeightCalculator,
    InheritanceWeightResult,
    FusionWeightResult,
    WeightVisualization
)

from .voice_inheritance import (
    VoiceInheritor,
    InheritanceConfig,
    InheritanceResult,
    InheritancePresetManager,
    VoiceInheritanceError,
    inherit_voice
)

from .voice_fusion import (
    VoiceFuser,
    FusionConfig,
    FusionSource,
    FusionResult,
    FusionPresetManager,
    FusionOptimizer,
    VoiceFusionError,
    fuse_voices_simple,
    create_fusion_chain
)

__all__ = [
    # 数据模型
    "VoiceConfig",
    "DDSPSVCConfig",
    "IndexTTSConfig",
    "WeightInfo",
    "VoicePreset",

    # 权重计算
    "WeightCalculator",
    "WeightCalculationResult",
    "WeightPreset",

    # 高级权重计算
    "AdvancedWeightCalculator",
    "InheritanceWeightResult",
    "FusionWeightResult",
    "WeightVisualization",

    # 音色管理
    "VoiceManager",
    "VoiceManagerError",
    "VoiceNotFoundError",
    "VoiceAlreadyExistsError",

    # 音色继承
    "VoiceInheritor",
    "InheritanceConfig",
    "InheritanceResult",
    "InheritancePresetManager",
    "VoiceInheritanceError",
    "inherit_voice",

    # 音色融合
    "VoiceFuser",
    "FusionConfig",
    "FusionSource",
    "FusionResult",
    "FusionPresetManager",
    "FusionOptimizer",
    "VoiceFusionError",
    "fuse_voices_simple",
    "create_fusion_chain",
]
