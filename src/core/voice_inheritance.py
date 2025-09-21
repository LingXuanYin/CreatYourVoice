
"""音色继承器

这个模块实现从现有音色创建新音色的功能。
设计原则：
1. 参数继承 - 从父音色继承DDSP和IndexTTS参数
2. 权重融合 - 按用户指定比例融合新旧参数
3. 元数据管理 - 维护继承关系和版本信息
4. 验证机制 - 确保继承后的配置有效性
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field

from .models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from .advanced_weight_calc import AdvancedWeightCalculator, InheritanceWeightResult
from .voice_manager import VoiceManager, VoiceNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class InheritanceConfig:
    """继承配置"""
    inheritance_ratio: float = 0.5  # 继承比例 (0.0-1.0)
    preserve_metadata: bool = True  # 是否保留元数据
    auto_generate_name: bool = True  # 是否自动生成名称
    copy_tags: bool = True  # 是否复制标签

    def __post_init__(self) -> None:
        """参数验证"""
        self.inheritance_ratio = max(0.0, min(1.0, self.inheritance_ratio))


@dataclass
class InheritanceResult:
    """继承结果"""
    new_voice_config: VoiceConfig
    inheritance_weights: InheritanceWeightResult
    inheritance_config: InheritanceConfig
    parent_voice_id: str
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """获取继承摘要"""
        return {
            "new_voice_id": self.new_voice_config.voice_id,
            "new_voice_name": self.new_voice_config.name,
            "parent_voice_id": self.parent_voice_id,
            "inheritance_ratio": self.inheritance_config.inheritance_ratio,
            "ddsp_speakers": list(self.inheritance_weights.ddsp_weights.keys()),
            "index_speakers": list(self.inheritance_weights.index_tts_weights.keys()),
            "processing_time": self.processing_time,
            "warnings_count": len(self.warnings)
        }


class VoiceInheritanceError(Exception):
    """音色继承异常"""
    pass


class VoiceInheritor:
    """音色继承器

    负责从现有音色创建新音色，支持参数继承和权重融合。
    核心功能：
    1. 参数继承 - 从父音色继承配置参数
    2. 权重计算 - 按比例融合新旧权重
    3. 配置验证 - 确保继承后配置的有效性
    4. 元数据管理 - 维护继承关系链
    """

    def __init__(self, voice_manager: VoiceManager):
        """初始化音色继承器

        Args:
            voice_manager: 音色管理器实例
        """
        self.voice_manager = voice_manager
        self.weight_calculator = AdvancedWeightCalculator()
        logger.info("音色继承器初始化完成")

    def inherit_from_voice(
        self,
        parent_voice_id: str,
        new_name: str,
        new_ddsp_config: DDSPSVCConfig,
        new_index_tts_config: IndexTTSConfig,
        inheritance_config: Optional[InheritanceConfig] = None
    ) -> InheritanceResult:
        """从现有音色继承创建新音色

        Args:
            parent_voice_id: 父音色ID
            new_name: 新音色名称
            new_ddsp_config: 新DDSP配置
            new_index_tts_config: 新IndexTTS配置
            inheritance_config: 继承配置

        Returns:
            InheritanceResult: 继承结果

        Raises:
            VoiceInheritanceError: 继承失败
        """
        import time
        start_time = time.time()

        if inheritance_config is None:
            inheritance_config = InheritanceConfig()

        try:
            logger.info(f"开始音色继承：{parent_voice_id} -> {new_name}")

            # 加载父音色
            parent_voice = self._load_parent_voice(parent_voice_id)

            # 计算继承权重
            inheritance_weights = self.weight_calculator.calculate_inheritance_weights(
                parent_voice,
                new_ddsp_config,
                new_index_tts_config,
                inheritance_config.inheritance_ratio
            )

            # 创建新音色配置
            new_voice_config = self._create_inherited_voice_config(
                parent_voice,
                new_name,
                new_ddsp_config,
                new_index_tts_config,
                inheritance_weights,
                inheritance_config
            )

            # 验证配置
            warnings = self._validate_inherited_config(new_voice_config, inheritance_weights)

            processing_time = time.time() - start_time

            result = InheritanceResult(
                new_voice_config=new_voice_config,
                inheritance_weights=inheritance_weights,
                inheritance_config=inheritance_config,
                parent_voice_id=parent_voice_id,
                processing_time=processing_time,
                warnings=warnings
            )

            logger.info(f"音色继承完成，耗时: {processing_time:.3f}s")
            return result

        except Exception as e:
            raise VoiceInheritanceError(f"音色继承失败: {e}")

    def inherit_from_voice_product(
        self,
        voice_product_path: Union[str, Path],
        new_name: str,
        new_ddsp_config: DDSPSVCConfig,
        new_index_tts_config: IndexTTSConfig,
        inheritance_config: Optional[InheritanceConfig] = None
    ) -> InheritanceResult:
        """从语音产物继承创建新音色

        Args:
            voice_product_path: 语音产物文件路径
            new_name: 新音色名称
            new_ddsp_config: 新DDSP配置
            new_index_tts_config: 新IndexTTS配置
            inheritance_config: 继承配置

        Returns:
            InheritanceResult: 继承结果
        """
        try:
            # 从语音产物文件加载配置
            voice_product_config = VoiceConfig.load_from_file(voice_product_path)

            # 临时保存到管理器（用于继承计算）
            temp_voice_id = f"temp_{uuid.uuid4().hex[:8]}"
            voice_product_config.voice_id = temp_voice_id

            try:
                self.voice_manager.save_voice(voice_product_config)

                # 执行继承
                result = self.inherit_from_voice(
                    temp_voice_id,
                    new_name,
                    new_ddsp_config,
                    new_index_tts_config,
                    inheritance_config
                )

                # 更新父音色ID为原始ID
                result.parent_voice_id = voice_product_config.voice_id
                result.new_voice_config.parent_voice_ids = [voice_product_config.voice_id]

                return result

            finally:
                # 清理临时音色
                try:
                    self.voice_manager.delete_voice(temp_voice_id)
                except:
                    pass

        except Exception as e:
            raise VoiceInheritanceError(f"从语音产物继承失败: {e}")

    def create_inheritance_chain(
        self,
        voice_configs: List[Tuple[VoiceConfig, float]],
        final_name: str,
        final_ddsp_config: DDSPSVCConfig,
        final_index_tts_config: IndexTTSConfig
    ) -> List[InheritanceResult]:
        """创建继承链

        Args:
            voice_configs: 音色配置和继承比例的列表
            final_name: 最终音色名称
            final_ddsp_config: 最终DDSP配置
            final_index_tts_config: 最终IndexTTS配置

        Returns:
            继承结果列表
        """
        if not voice_configs:
            raise VoiceInheritanceError("继承链不能为空")

        logger.info(f"创建继承链，长度: {len(voice_configs)}")

        results = []
        current_voice = voice_configs[0][0]

        # 保存第一个音色（如果还没保存）
        try:
            self.voice_manager.load_voice(current_voice.voice_id)
        except VoiceNotFoundError:
            self.voice_manager.save_voice(current_voice)

        # 逐步继承
        for i, (next_voice, inheritance_ratio) in enumerate(voice_configs[1:], 1):
            is_final = (i == len(voice_configs) - 1)

            if is_final:
                # 最后一步，使用用户提供的配置
                ddsp_config = final_ddsp_config
                index_config = final_index_tts_config
                name = final_name
            else:
                # 中间步骤，使用下一个音色的配置
                ddsp_config = next_voice.ddsp_config
                index_config = next_voice.index_tts_config
                name = f"{current_voice.name}_继承_{i}"

            inheritance_config = InheritanceConfig(inheritance_ratio=inheritance_ratio)

            result = self.inherit_from_voice(
                current_voice.voice_id,
                name,
                ddsp_config,
                index_config,
                inheritance_config
            )

            # 保存中间结果
            self.voice_manager.save_voice(result.new_voice_config)
            results.append(result)

            # 更新当前音色
            current_voice = result.new_voice_config

        logger.info(f"继承链创建完成，共 {len(results)} 步")
        return results

    def preview_inheritance(
        self,
        parent_voice_id: str,
        new_ddsp_config: DDSPSVCConfig,
        new_index_tts_config: IndexTTSConfig,
        inheritance_ratio: float
    ) -> Dict[str, Any]:
        """预览继承结果

        Args:
            parent_voice_id: 父音色ID
            new_ddsp_config: 新DDSP配置
            new_index_tts_config: 新IndexTTS配置
            inheritance_ratio: 继承比例

        Returns:
            预览信息字典
        """
        try:
            parent_voice = self._load_parent_voice(parent_voice_id)

            inheritance_weights = self.weight_calculator.calculate_inheritance_weights(
                parent_voice,
                new_ddsp_config,
                new_index_tts_config,
                inheritance_ratio
            )

            # 分析权重变化
            old_ddsp_weights = parent_voice.ddsp_config.spk_mix_dict or {}
            new_ddsp_weights = new_ddsp_config.spk_mix_dict or {}

            weight_changes = self._analyze_weight_changes(
                old_ddsp_weights,
                new_ddsp_weights,
                inheritance_weights.ddsp_weights
            )

            return {
                "parent_voice": {
                    "id": parent_voice.voice_id,
                    "name": parent_voice.name,
                    "ddsp_speakers": list(old_ddsp_weights.keys()),
                    "index_speaker": parent_voice.index_tts_config.speaker_name
                },
                "inheritance_ratio": inheritance_ratio,
                "resulting_weights": {
                    "ddsp": inheritance_weights.ddsp_weights,
                    "index_tts": inheritance_weights.index_tts_weights,
                    "combined": inheritance_weights.combined_weights
                },
                "weight_changes": weight_changes,
                "speaker_analysis": self._analyze_speaker_distribution(inheritance_weights)
            }

        except Exception as e:
            return {"error": f"预览失败: {e}"}

    def _load_parent_voice(self, parent_voice_id: str) -> VoiceConfig:
        """加载父音色"""
        try:
            return self.voice_manager.load_voice(parent_voice_id)
        except VoiceNotFoundError:
            raise VoiceInheritanceError(f"父音色不存在: {parent_voice_id}")

    def _create_inherited_voice_config(
        self,
        parent_voice: VoiceConfig,
        new_name: str,
        new_ddsp_config: DDSPSVCConfig,
        new_index_tts_config: IndexTTSConfig,
        inheritance_weights: InheritanceWeightResult,
        inheritance_config: InheritanceConfig
    ) -> VoiceConfig:
        """创建继承后的音色配置"""

        # 创建继承后的DDSP配置
        inherited_ddsp_config = self._create_inherited_ddsp_config(
            parent_voice.ddsp_config,
            new_ddsp_config,
            inheritance_weights,
            inheritance_config.inheritance_ratio
        )

        # 创建继承后的IndexTTS配置
        inherited_index_config = self._create_inherited_index_config(
            parent_voice.index_tts_config,
            new_index_tts_config,
            inheritance_config.inheritance_ratio
        )

        # 创建权重信息
        weight_info = WeightInfo(
            speaker_weights=inheritance_weights.combined_weights
        )

        # 生成新的音色ID
        new_voice_id = str(uuid.uuid4())

        # 处理标签
        tags = []
        if inheritance_config.copy_tags:
            tags.extend(parent_voice.tags)
        tags.append("继承音色")
        tags = list(set(tags))  # 去重

        # 创建新音色配置
        new_voice_config = VoiceConfig(
            voice_id=new_voice_id,
            name=new_name,
            description=f"继承自 '{parent_voice.name}' (比例: {inheritance_config.inheritance_ratio:.1%})",
            ddsp_config=inherited_ddsp_config,
            index_tts_config=inherited_index_config,
            weight_info=weight_info,
            tags=tags,
            parent_voice_ids=[parent_voice.voice_id],
            fusion_weights={parent_voice.voice_id: inheritance_config.inheritance_ratio}
        )

        return new_voice_config

    def _create_inherited_ddsp_config(
        self,
        parent_ddsp: DDSPSVCConfig,
        new_ddsp: DDSPSVCConfig,
        inheritance_weights: InheritanceWeightResult,
        inheritance_ratio: float
    ) -> DDSPSVCConfig:
        """创建继承后的DDSP配置"""

        # 继承基础参数
        inherited_config = DDSPSVCConfig(
            model_path=new_ddsp.model_path,  # 使用新的模型路径
            config_path=new_ddsp.config_path,  # 使用新的配置路径
            speaker_id=new_ddsp.speaker_id,  # 使用新的说话人ID
            f0_predictor=self._inherit_parameter(
                parent_ddsp.f0_predictor,
                new_ddsp.f0_predictor,
                inheritance_ratio,
                str
            ),
            f0_min=self._inherit_parameter(
                parent_ddsp.f0_min,
                new_ddsp.f0_min,
                inheritance_ratio,
                float
            ),
            f0_max=self._inherit_parameter(
                parent_ddsp.f0_max,
                new_ddsp.f0_max,
                inheritance_ratio,
                float
            ),
            threhold=self._inherit_parameter(
                parent_ddsp.threhold,
                new_ddsp.threhold,
                inheritance_ratio,
                float
            ),
            spk_mix_dict=inheritance_weights.ddsp_weights,
            use_spk_mix=len(inheritance_weights.ddsp_weights) > 1
        )

        return inherited_config

    def _create_inherited_index_config(
        self,
        parent_index: IndexTTSConfig,
        new_index: IndexTTSConfig,
        inheritance_ratio: float
    ) -> IndexTTSConfig:
        """创建继承后的IndexTTS配置"""

        inherited_config = IndexTTSConfig(
            model_path=new_index.model_path,  # 使用新的模型路径
            config_path=new_index.config_path,  # 使用新的配置路径
            speaker_name=new_index.speaker_name,  # 使用新的说话人名称
            emotion_reference=new_index.emotion_reference,  # 使用新的情感参考
            emotion_strength=self._inherit_parameter(
                parent_index.emotion_strength,
                new_index.emotion_strength,
                inheritance_ratio,
                float
            ),
            speed=self._inherit_parameter(
                parent_index.speed,
                new_index.speed,
                inheritance_ratio,
                float
            ),
            temperature=self._inherit_parameter(
                parent_index.temperature,
                new_index.temperature,
                inheritance_ratio,
                float
            ),
            top_k=self._inherit_parameter(
                parent_index.top_k,
                new_index.top_k,
                inheritance_ratio,
                int
            ),
            top_p=self._inherit_parameter(
                parent_index.top_p,
                new_index.top_p,
                inheritance_ratio,
                float
            )
        )

        return inherited_config

    def _inherit_parameter(
        self,
        old_value: Any,
        new_value: Any,
        inheritance_ratio: float,
        param_type: type
    ) -> Any:
        """继承单个参数

        Args:
            old_value: 旧值
            new_value: 新值
            inheritance_ratio: 继承比例
            param_type: 参数类型

        Returns:
            继承后的值
        """
        if param_type == str:
            # 字符串类型：根据继承比例选择
            return old_value if inheritance_ratio > 0.5 else new_value
        elif param_type in (int, float):
            # 数值类型：加权平均
            inherited_value = old_value * inheritance_ratio + new_value * (1.0 - inheritance_ratio)
            return param_type(inherited_value)
        else:
            # 其他类型：直接使用新值
            return new_value

    def _validate_inherited_config(
        self,
        voice_config: VoiceConfig,
        inheritance_weights: InheritanceWeightResult
    ) -> List[str]:
        """验证继承后的配置"""
        warnings = []

        # 验证权重一致性
        is_valid, errors = self.weight_calculator.validate_weights_consistency(
            inheritance_weights.ddsp_weights,
            inheritance_weights.index_tts_weights
        )

        if not is_valid:
            warnings.extend(errors)

        # 验证DDSP配置
        if not voice_config.ddsp_config.model_path:
            warnings.append("DDSP模型路径为空")

        if not voice_config.ddsp_config.config_path:
            warnings.append("DDSP配置路径为空")

        # 验证IndexTTS配置
        if not voice_config.index_tts_config.model_path:
            warnings.append("IndexTTS模型路径为空")

        if not voice_config.index_tts_config.speaker_name:
            warnings.append("IndexTTS说话人名称为空")

        # 验证权重分布
        if not inheritance_weights.ddsp_weights:
            warnings.append("DDSP权重为空")

        if len(inheritance_weights.ddsp_weights) > 10:
            warnings.append(f"DDSP说话人数量过多: {len(inheritance_weights.ddsp_weights)}")

        return warnings

    def _analyze_weight_changes(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        final_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """分析权重变化"""

        all_speakers = set(old_weights.keys()) | set(new_weights.keys()) | set(final_weights.keys())

        changes = {}
        for speaker in all_speakers:
            old_w = old_weights.get(speaker, 0.0)
            new_w = new_weights.get(speaker, 0.0)
            final_w = final_weights.get(speaker, 0.0)

            changes[speaker] = {
                "old": old_w,
                "new": new_w,
                "final": final_w,
                "change": final_w - old_w,
                "change_percent": ((final_w - old_w) / old_w * 100) if old_w > 0 else float('inf')
            }

        return {
            "speaker_changes": changes,
            "total_speakers": len(all_speakers),
            "new_speakers": len(set(final_weights.keys()) - set(old_weights.keys())),
            "removed_speakers": len(set(old_weights.keys()) - set(final_weights.keys()))
        }

    def _analyze_speaker_distribution(
        self,
        inheritance_weights: InheritanceWeightResult
    ) -> Dict[str, Any]:
        """分析说话人分布"""

        ddsp_weights = inheritance_weights.ddsp_weights
        combined_weights = inheritance_weights.combined_weights

        if not ddsp_weights:
            return {"error": "无权重数据"}

        # 计算分布统计
        weights_list = list(ddsp_weights.values())

        return {
            "speaker_count": len(ddsp_weights),
            "max_weight": max(weights_list),
            "min_weight": min(weights_list),
            "avg_weight": sum(weights_list) / len(weights_list),
            "dominant_speaker": max(ddsp_weights.items(), key=lambda x: x[1])[0],
            "weight_distribution": "均匀" if max(weights_list) - min(weights_list) < 0.3 else "不均匀",
            "entropy": self._calculate_entropy(weights_list)
        }

    def _calculate_entropy(self, weights: List[float]) -> float:
        """计算权重分布的熵"""
        import math

        if not weights:
            return 0.0

        # 归一化权重
        total = sum(weights)
        if total == 0:
            return 0.0

        normalized = [w / total for w in weights]

        # 计算熵
        entropy = 0.0
        for w in normalized:
            if w > 0:
                entropy -= w * math.log2(w)

        return entropy


class InheritancePresetManager:
    """继承预设管理器

    管理常用的继承配置预设。
    """

    @staticmethod
    def get_conservative_preset() -> InheritanceConfig:
        """保守继承预设（高继承比例）"""
        return InheritanceConfig(
            inheritance_ratio=0.8,
            preserve_metadata=True,
            auto_generate_name=True,
            copy_tags=True
        )

    @staticmethod
    def get_balanced_preset() -> InheritanceConfig:
        """平衡继承预设（中等继承比例）"""
        return InheritanceConfig(
            inheritance_ratio=0.5,
            preserve_metadata=True,
            auto_generate_name=True,
            copy_tags=True
        )

    @staticmethod
    def get_innovative_preset() -> InheritanceConfig:
        """创新继承预设（低继承比例）"""
        return InheritanceConfig(
            inheritance_ratio=0.2,
            preserve_metadata=False,
            auto_generate_name=True,
            copy_tags=False
        )

    @staticmethod
    def get_custom_preset(inheritance_ratio: float) -> InheritanceConfig:
        """自定义继承预设"""
        return InheritanceConfig(
            inheritance_ratio=inheritance_ratio,
            preserve_metadata=True,
            auto_generate_name=True,
            copy_tags=True
        )


# 便捷函数
def inherit_voice(
    voice_manager: VoiceManager,
    parent_voice_id: str,
    new_name: str,
    new_ddsp_config: DDSPSVCConfig,
    new_index_tts_config: IndexTTSConfig,
    inheritance_ratio: float = 0.5,
    save_result: bool = True
) -> InheritanceResult:
    """便捷的音色继承函数

    Args:
        voice_manager: 音色管理器
        parent_voice_id: 父音色ID
        new_name: 新音色名称
        new_ddsp_config: 新DDSP配置
        new_index_tts_config: 新IndexTTS配置
        inheritance_ratio: 继承比例
        save_result: 是否保存结果

    Returns:
        InheritanceResult: 继承结果
    """
    inheritor = VoiceInheritor(voice_manager)
    inheritance_config = InheritanceConfig(inheritance_ratio=inheritance_ratio)

    result = inheritor.inherit_from_voice(
        parent_voice_id,
        new_name,
        new_ddsp_config,
        new_index_tts_config,
        inheritance_config
    )

    if save_result:
        voice_manager.save_voice(result.new_voice_config)

    return result
