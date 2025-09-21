
"""音色融合器

这个模块实现多音色融合功能，支持复杂的权重计算和参数融合。
设计原则：
1. 多音色支持 - 支持任意数量音色的融合
2. 权重归一化 - 确保所有权重总和为1.0
3. 参数融合 - 智能融合DDSP和IndexTTS参数
4. 冲突解决 - 处理参数冲突和不兼容情况
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
import copy

from .models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from .advanced_weight_calc import AdvancedWeightCalculator, FusionWeightResult
from .voice_manager import VoiceManager, VoiceNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """融合配置"""
    auto_normalize_weights: bool = True  # 自动归一化权重
    resolve_conflicts: bool = True  # 自动解决参数冲突
    max_speakers: int = 10  # 最大说话人数量
    min_weight_threshold: float = 0.01  # 最小权重阈值
    preserve_dominant_config: bool = True  # 保留主导音色的配置

    def __post_init__(self) -> None:
        """参数验证"""
        self.max_speakers = max(1, self.max_speakers)
        self.min_weight_threshold = max(0.0, min(1.0, self.min_weight_threshold))


@dataclass
class FusionSource:
    """融合源"""
    voice_config: VoiceConfig
    weight: float
    priority: int = 0  # 优先级，数字越大优先级越高

    def __post_init__(self) -> None:
        """参数验证"""
        self.weight = max(0.0, self.weight)


@dataclass
class FusionResult:
    """融合结果"""
    fused_voice_config: VoiceConfig
    fusion_weights: FusionWeightResult
    fusion_config: FusionConfig
    source_voices: List[FusionSource]
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    conflicts_resolved: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """获取融合摘要"""
        return {
            "fused_voice_id": self.fused_voice_config.voice_id,
            "fused_voice_name": self.fused_voice_config.name,
            "source_count": len(self.source_voices),
            "source_voice_ids": [src.voice_config.voice_id for src in self.source_voices],
            "final_speakers": list(self.fusion_weights.combined_weights.keys()),
            "processing_time": self.processing_time,
            "warnings_count": len(self.warnings),
            "conflicts_count": len(self.conflicts_resolved)
        }


class VoiceFusionError(Exception):
    """音色融合异常"""
    pass


class VoiceFuser:
    """音色融合器

    负责多个音色的融合，支持复杂的权重计算和参数融合。
    核心功能：
    1. 多音色融合 - 支持任意数量音色的融合
    2. 权重计算 - 智能计算最终权重分布
    3. 参数融合 - 融合DDSP和IndexTTS参数
    4. 冲突解决 - 自动解决参数冲突
    """

    def __init__(self, voice_manager: VoiceManager):
        """初始化音色融合器

        Args:
            voice_manager: 音色管理器实例
        """
        self.voice_manager = voice_manager
        self.weight_calculator = AdvancedWeightCalculator()
        logger.info("音色融合器初始化完成")

    def fuse_voices(
        self,
        fusion_sources: List[FusionSource],
        fused_name: str,
        fusion_config: Optional[FusionConfig] = None
    ) -> FusionResult:
        """融合多个音色

        Args:
            fusion_sources: 融合源列表
            fused_name: 融合后音色名称
            fusion_config: 融合配置

        Returns:
            FusionResult: 融合结果

        Raises:
            VoiceFusionError: 融合失败
        """
        import time
        start_time = time.time()

        if fusion_config is None:
            fusion_config = FusionConfig()

        try:
            logger.info(f"开始音色融合：{len(fusion_sources)} 个音色 -> {fused_name}")

            # 验证融合源
            self._validate_fusion_sources(fusion_sources)

            # 归一化权重
            if fusion_config.auto_normalize_weights:
                fusion_sources = self._normalize_fusion_weights(fusion_sources)

            # 计算融合权重
            fusion_weights = self._calculate_fusion_weights(fusion_sources)

            # 创建融合配置
            fused_voice_config, warnings, conflicts = self._create_fused_voice_config(
                fusion_sources,
                fused_name,
                fusion_weights,
                fusion_config
            )

            processing_time = time.time() - start_time

            result = FusionResult(
                fused_voice_config=fused_voice_config,
                fusion_weights=fusion_weights,
                fusion_config=fusion_config,
                source_voices=fusion_sources,
                processing_time=processing_time,
                warnings=warnings,
                conflicts_resolved=conflicts
            )

            logger.info(f"音色融合完成，耗时: {processing_time:.3f}s")
            return result

        except Exception as e:
            raise VoiceFusionError(f"音色融合失败: {e}")

    def fuse_by_voice_ids(
        self,
        voice_ids_and_weights: Dict[str, float],
        fused_name: str,
        fusion_config: Optional[FusionConfig] = None
    ) -> FusionResult:
        """通过音色ID和权重融合

        Args:
            voice_ids_and_weights: 音色ID到权重的映射
            fused_name: 融合后音色名称
            fusion_config: 融合配置

        Returns:
            FusionResult: 融合结果
        """
        try:
            # 加载音色配置
            fusion_sources = []
            for voice_id, weight in voice_ids_and_weights.items():
                voice_config = self.voice_manager.load_voice(voice_id)
                fusion_sources.append(FusionSource(
                    voice_config=voice_config,
                    weight=weight
                ))

            return self.fuse_voices(fusion_sources, fused_name, fusion_config)

        except VoiceNotFoundError as e:
            raise VoiceFusionError(f"音色不存在: {e}")

    def fuse_voice_products(
        self,
        voice_product_paths: List[Tuple[Union[str, Path], float]],
        fused_name: str,
        fusion_config: Optional[FusionConfig] = None
    ) -> FusionResult:
        """融合语音产物文件

        Args:
            voice_product_paths: 语音产物路径和权重的列表
            fused_name: 融合后音色名称
            fusion_config: 融合配置

        Returns:
            FusionResult: 融合结果
        """
        try:
            fusion_sources = []

            for path, weight in voice_product_paths:
                voice_config = VoiceConfig.load_from_file(path)
                fusion_sources.append(FusionSource(
                    voice_config=voice_config,
                    weight=weight
                ))

            return self.fuse_voices(fusion_sources, fused_name, fusion_config)

        except Exception as e:
            raise VoiceFusionError(f"融合语音产物失败: {e}")

    def preview_fusion(
        self,
        fusion_sources: List[FusionSource],
        fusion_config: Optional[FusionConfig] = None
    ) -> Dict[str, Any]:
        """预览融合结果

        Args:
            fusion_sources: 融合源列表
            fusion_config: 融合配置

        Returns:
            预览信息字典
        """
        try:
            if fusion_config is None:
                fusion_config = FusionConfig()

            # 归一化权重
            if fusion_config.auto_normalize_weights:
                fusion_sources = self._normalize_fusion_weights(fusion_sources)

            # 计算融合权重
            fusion_weights = self._calculate_fusion_weights(fusion_sources)

            # 分析融合结果
            analysis = self._analyze_fusion_compatibility(fusion_sources)

            return {
                "source_voices": [
                    {
                        "id": src.voice_config.voice_id,
                        "name": src.voice_config.name,
                        "weight": src.weight,
                        "priority": src.priority,
                        "ddsp_speakers": list((src.voice_config.ddsp_config.spk_mix_dict or {}).keys()),
                        "index_speaker": src.voice_config.index_tts_config.speaker_name
                    }
                    for src in fusion_sources
                ],
                "fusion_weights": {
                    "ddsp": fusion_weights.ddsp_weights,
                    "index_tts": fusion_weights.index_tts_weights,
                    "combined": fusion_weights.combined_weights
                },
                "compatibility_analysis": analysis,
                "estimated_conflicts": self._estimate_conflicts(fusion_sources),
                "speaker_distribution": self._analyze_speaker_distribution(fusion_weights)
            }

        except Exception as e:
            return {"error": f"预览失败: {e}"}

    def create_fusion_template(
        self,
        template_name: str,
        fusion_sources: List[FusionSource],
        description: str = ""
    ) -> Dict[str, Any]:
        """创建融合模板

        Args:
            template_name: 模板名称
            fusion_sources: 融合源列表
            description: 模板描述

        Returns:
            融合模板字典
        """
        template = {
            "name": template_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "sources": [
                {
                    "voice_id": src.voice_config.voice_id,
                    "voice_name": src.voice_config.name,
                    "weight": src.weight,
                    "priority": src.priority
                }
                for src in fusion_sources
            ],
            "total_sources": len(fusion_sources),
            "total_weight": sum(src.weight for src in fusion_sources)
        }

        return template

    def _validate_fusion_sources(self, fusion_sources: List[FusionSource]) -> None:
        """验证融合源"""
        if not fusion_sources:
            raise VoiceFusionError("融合源列表不能为空")

        if len(fusion_sources) < 2:
            raise VoiceFusionError("至少需要两个音色进行融合")

        # 检查权重
        total_weight = sum(src.weight for src in fusion_sources)
        if total_weight <= 0:
            raise VoiceFusionError("权重总和必须大于0")

        # 检查音色配置
        for i, src in enumerate(fusion_sources):
            if not src.voice_config:
                raise VoiceFusionError(f"第{i+1}个融合源的音色配置为空")

            if not src.voice_config.ddsp_config:
                raise VoiceFusionError(f"第{i+1}个融合源缺少DDSP配置")

            if not src.voice_config.index_tts_config:
                raise VoiceFusionError(f"第{i+1}个融合源缺少IndexTTS配置")

    def _normalize_fusion_weights(self, fusion_sources: List[FusionSource]) -> List[FusionSource]:
        """归一化融合权重"""
        total_weight = sum(src.weight for src in fusion_sources)

        if total_weight <= 0:
            # 如果所有权重都是0，平均分配
            equal_weight = 1.0 / len(fusion_sources)
            normalized_sources = []
            for src in fusion_sources:
                new_src = copy.deepcopy(src)
                new_src.weight = equal_weight
                normalized_sources.append(new_src)
            return normalized_sources

        # 正常归一化
        normalized_sources = []
        for src in fusion_sources:
            new_src = copy.deepcopy(src)
            new_src.weight = src.weight / total_weight
            normalized_sources.append(new_src)

        return normalized_sources

    def _calculate_fusion_weights(self, fusion_sources: List[FusionSource]) -> FusionWeightResult:
        """计算融合权重"""
        # 构建音色配置字典和权重字典
        voice_configs = {
            src.voice_config.voice_id: src.voice_config
            for src in fusion_sources
        }

        fusion_weights = {
            src.voice_config.voice_id: src.weight
            for src in fusion_sources
        }

        return self.weight_calculator.calculate_fusion_weights(voice_configs, fusion_weights)

    def _create_fused_voice_config(
        self,
        fusion_sources: List[FusionSource],
        fused_name: str,
        fusion_weights: FusionWeightResult,
        fusion_config: FusionConfig
    ) -> Tuple[VoiceConfig, List[str], List[str]]:
        """创建融合后的音色配置"""
        warnings = []
        conflicts_resolved = []

        # 选择主导音色（权重最大的）
        dominant_source = max(fusion_sources, key=lambda x: x.weight)

        # 创建融合后的DDSP配置
        fused_ddsp_config, ddsp_warnings, ddsp_conflicts = self._create_fused_ddsp_config(
            fusion_sources, fusion_weights, fusion_config, dominant_source
        )
        warnings.extend(ddsp_warnings)
        conflicts_resolved.extend(ddsp_conflicts)

        # 创建融合后的IndexTTS配置
        fused_index_config, index_warnings, index_conflicts = self._create_fused_index_config(
            fusion_sources, fusion_weights, fusion_config, dominant_source
        )
        warnings.extend(index_warnings)
        conflicts_resolved.extend(index_conflicts)

        # 创建权重信息
        weight_info = WeightInfo(
            speaker_weights=fusion_weights.combined_weights
        )

        # 合并标签
        all_tags = set()
        for src in fusion_sources:
            all_tags.update(src.voice_config.tags)
        all_tags.add("融合音色")

        # 生成新的音色ID
        new_voice_id = str(uuid.uuid4())

        # 创建融合后的音色配置
        fused_voice_config = VoiceConfig(
            voice_id=new_voice_id,
            name=fused_name,
            description=f"融合音色，基于 {len(fusion_sources)} 个源音色",
            ddsp_config=fused_ddsp_config,
            index_tts_config=fused_index_config,
            weight_info=weight_info,
            tags=list(all_tags),
            parent_voice_ids=[src.voice_config.voice_id for src in fusion_sources],
            fusion_weights=fusion_weights.source_weights
        )

        return fused_voice_config, warnings, conflicts_resolved

    def _create_fused_ddsp_config(
        self,
        fusion_sources: List[FusionSource],
        fusion_weights: FusionWeightResult,
        fusion_config: FusionConfig,
        dominant_source: FusionSource
    ) -> Tuple[DDSPSVCConfig, List[str], List[str]]:
        """创建融合后的DDSP配置"""
        warnings = []
        conflicts_resolved = []

        # 使用主导音色的基础配置
        base_config = dominant_source.voice_config.ddsp_config

        # 融合数值参数
        fused_f0_min = self._fuse_numeric_parameter(
            [src.voice_config.ddsp_config.f0_min for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        fused_f0_max = self._fuse_numeric_parameter(
            [src.voice_config.ddsp_config.f0_max for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        fused_threshold = self._fuse_numeric_parameter(
            [src.voice_config.ddsp_config.threhold for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        # 选择F0预测器（使用主导音色的）
        f0_predictor = base_config.f0_predictor

        # 检查F0预测器冲突
        predictors = set(src.voice_config.ddsp_config.f0_predictor for src in fusion_sources)
        if len(predictors) > 1:
            conflicts_resolved.append(f"F0预测器冲突，使用主导音色的: {f0_predictor}")

        # 优化说话人权重
        optimized_weights = self.weight_calculator.optimize_speaker_selection(
            fusion_weights.ddsp_weights,
            fusion_config.max_speakers,
            fusion_config.min_weight_threshold
        )

        if len(optimized_weights) < len(fusion_weights.ddsp_weights):
            warnings.append(f"说话人数量优化：{len(fusion_weights.ddsp_weights)} -> {len(optimized_weights)}")

        fused_ddsp_config = DDSPSVCConfig(
            model_path=base_config.model_path,
            config_path=base_config.config_path,
            speaker_id=base_config.speaker_id,
            f0_predictor=f0_predictor,
            f0_min=fused_f0_min,
            f0_max=fused_f0_max,
            threhold=fused_threshold,
            spk_mix_dict=optimized_weights,
            use_spk_mix=len(optimized_weights) > 1
        )

        return fused_ddsp_config, warnings, conflicts_resolved

    def _create_fused_index_config(
        self,
        fusion_sources: List[FusionSource],
        fusion_weights: FusionWeightResult,
        fusion_config: FusionConfig,
        dominant_source: FusionSource
    ) -> Tuple[IndexTTSConfig, List[str], List[str]]:
        """创建融合后的IndexTTS配置"""
        warnings = []
        conflicts_resolved = []

        # 使用主导音色的基础配置
        base_config = dominant_source.voice_config.index_tts_config

        # 融合数值参数
        fused_emotion_strength = self._fuse_numeric_parameter(
            [src.voice_config.index_tts_config.emotion_strength for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        fused_speed = self._fuse_numeric_parameter(
            [src.voice_config.index_tts_config.speed for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        fused_temperature = self._fuse_numeric_parameter(
            [src.voice_config.index_tts_config.temperature for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        fused_top_k = int(self._fuse_numeric_parameter(
            [float(src.voice_config.index_tts_config.top_k) for src in fusion_sources],
            [src.weight for src in fusion_sources]
        ))

        fused_top_p = self._fuse_numeric_parameter(
            [src.voice_config.index_tts_config.top_p for src in fusion_sources],
            [src.weight for src in fusion_sources]
        )

        # 选择说话人名称（使用主导音色的）
        speaker_name = base_config.speaker_name

        # 检查说话人名称冲突
        speaker_names = set(src.voice_config.index_tts_config.speaker_name for src in fusion_sources)
        if len(speaker_names) > 1:
            conflicts_resolved.append(f"说话人名称冲突，使用主导音色的: {speaker_name}")

        fused_index_config = IndexTTSConfig(
            model_path=base_config.model_path,
            config_path=base_config.config_path,
            speaker_name=speaker_name,
            emotion_reference=base_config.emotion_reference,
            emotion_strength=fused_emotion_strength,
            speed=fused_speed,
            temperature=fused_temperature,
            top_k=fused_top_k,
            top_p=fused_top_p
        )

        return fused_index_config, warnings, conflicts_resolved

    def _fuse_numeric_parameter(
        self,
        values: List[Union[int, float]],
        weights: List[float]
    ) -> float:
        """融合数值参数

        Args:
            values: 参数值列表
            weights: 对应的权重列表

        Returns:
            融合后的参数值
        """
        if len(values) != len(weights):
            raise ValueError("参数值和权重列表长度不匹配")

        if not values:
            return 0.0

        # 加权平均
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        weight_sum = sum(weights)

        if weight_sum == 0:
            return sum(values) / len(values)  # 简单平均

        return weighted_sum / weight_sum

    def _analyze_fusion_compatibility(self, fusion_sources: List[FusionSource]) -> Dict[str, Any]:
        """分析融合兼容性"""
        compatibility = {
            "model_compatibility": True,
            "parameter_conflicts": [],
            "speaker_overlap": {},
            "config_differences": []
        }

        # 检查模型兼容性
        ddsp_models = set(src.voice_config.ddsp_config.model_path for src in fusion_sources)
        index_models = set(src.voice_config.index_tts_config.model_path for src in fusion_sources)

        if len(ddsp_models) > 1:
            compatibility["parameter_conflicts"].append("DDSP模型路径不一致")
            compatibility["model_compatibility"] = False

        if len(index_models) > 1:
            compatibility["parameter_conflicts"].append("IndexTTS模型路径不一致")
            compatibility["model_compatibility"] = False

        # 检查说话人重叠
        all_speakers = set()
        for src in fusion_sources:
            speakers = set((src.voice_config.ddsp_config.spk_mix_dict or {}).keys())
            overlap = speakers & all_speakers
            if overlap:
                compatibility["speaker_overlap"][src.voice_config.voice_id] = list(overlap)
            all_speakers.update(speakers)

        # 检查配置差异
        f0_predictors = set(src.voice_config.ddsp_config.f0_predictor for src in fusion_sources)
        if len(f0_predictors) > 1:
            compatibility["config_differences"].append(f"F0预测器不一致: {list(f0_predictors)}")

        return compatibility

    def _estimate_conflicts(self, fusion_sources: List[FusionSource]) -> List[str]:
        """估计可能的冲突"""
        conflicts = []

        # 检查模型路径冲突
        ddsp_paths = [src.voice_config.ddsp_config.model_path for src in fusion_sources]
        if len(set(ddsp_paths)) > 1:
            conflicts.append("DDSP模型路径冲突")

        index_paths = [src.voice_config.index_tts_config.model_path for src in fusion_sources]
        if len(set(index_paths)) > 1:
            conflicts.append("IndexTTS模型路径冲突")

        # 检查参数范围冲突
        f0_mins = [src.voice_config.ddsp_config.f0_min for src in fusion_sources]
        f0_maxs = [src.voice_config.ddsp_config.f0_max for src in fusion_sources]

        if max(f0_mins) - min(f0_mins) > 50:
            conflicts.append("F0最小值差异较大")

        if max(f0_maxs) - min(f0_maxs) > 200:
            conflicts.append("F0最大值差异较大")

        return conflicts

    def _analyze_speaker_distribution(self, fusion_weights: FusionWeightResult) -> Dict[str, Any]:
        """分析说话人分布"""
        ddsp_weights = fusion_weights.ddsp_weights

        if not ddsp_weights:
            return {"error": "无权重数据"}

        weights_list = list(ddsp_weights.values())

        return {
            "total_speakers": len(ddsp_weights),
            "max_weight": max(weights_list),
            "min_weight": min(weights_list),
            "avg_weight": sum(weights_list) / len(weights_list),
            "dominant_speaker": max(ddsp_weights.items(), key=lambda x: x[1])[0],
            "weight_variance": sum((w - sum(weights_list)/len(weights_list))**2 for w in weights_list) / len(weights_list),
            "distribution_type": self._classify_distribution(weights_list)
        }

    def _classify_distribution(self, weights: List[float]) -> str:
        """分类权重分布类型"""
        if not weights:
            return "空"

        max_w = max(weights)
        min_w = min(weights)
        avg_w = sum(weights) / len(weights)

        if max_w - min_w < 0.1:
            return "均匀分布"
        elif max_w > 0.7:
            return "主导型分布"
        elif max_w - avg_w < 0.2:
            return "平衡型分布"
        else:
            return "不均匀分布"


class FusionPresetManager:
    """融合预设管理器

    管理常用的融合配置预设。
    """

    @staticmethod
    def get_balanced_preset() -> FusionConfig:
        """平衡融合预设"""
        return FusionConfig(
            auto_normalize_weights=True,
            resolve_conflicts=True,
            max_speakers=8,
            min_weight_threshold=0.05,
            preserve_dominant_config=True
        )

    @staticmethod
    def get_conservative_preset() -> FusionConfig:
        """保守融合预设（保留更多细节）"""
        return FusionConfig(
            auto_normalize_weights=True,
            resolve_conflicts=True,
            max_speakers=12,
            min_weight_threshold=0.01,
            preserve_dominant_config=True
        )

    @staticmethod
    def get_aggressive_preset() -> FusionConfig:
        """激进融合预设（简化结果）"""
        return FusionConfig(
            auto_normalize_weights=True,
            resolve_conflicts=True,
            max_speakers=5,
            min_weight_threshold=0.1,
            preserve_dominant_config=False
        )

    @staticmethod
    def get_custom_preset(
        max_speakers: int = 8,
        min_weight_threshold: float = 0.05
    ) -> FusionConfig:
        """自定义融合预设"""
        return FusionConfig(
            auto_normalize_weights=True,
            resolve_conflicts=True,
            max_speakers=max_speakers,
            min_weight_threshold=min_weight_threshold,
            preserve_dominant_config=True
        )


class FusionOptimizer:
    """融合优化器

    提供融合结果的优化建议和自动调整功能。
    """

    def __init__(self, weight_calculator: AdvancedWeightCalculator):
        """初始化融合优化器"""
        self.weight_calculator = weight_calculator

    def optimize_fusion_weights(
        self,
        fusion_sources: List[FusionSource],
        target_speakers: int = 6
    ) -> List[FusionSource]:
        """优化融合权重

        Args:
            fusion_sources: 原始融合源
            target_speakers: 目标说话人数量

        Returns:
            优化后的融合源列表
        """
        # 计算当前权重分布
        current_weights = {src.voice_config.voice_id: src.weight for src in fusion_sources}

        # 使用权重计算器优化
        available_speakers = [src.voice_config.voice_id for src in fusion_sources]
        optimized_weights = self.weight_calculator.optimize_speaker_selection(
            current_weights,
            target_speakers
        )

        # 更新融合源权重
        optimized_sources = []
        for src in fusion_sources:
            if src.voice_config.voice_id in optimized_weights:
                new_src = copy.deepcopy(src)
                new_src.weight = optimized_weights[src.voice_config.voice_id]
                optimized_sources.append(new_src)

        return optimized_sources

    def suggest_fusion_improvements(
        self,
        fusion_result: FusionResult
    ) -> List[str]:
        """建议融合改进"""
        suggestions = []

        # 分析权重分布
        weights = list(fusion_result.fusion_weights.combined_weights.values())
        if not weights:
            return ["无权重数据，无法提供建议"]

        max_weight = max(weights)
        min_weight = min(weights)

        # 权重分布建议
        if max_weight > 0.8:
            suggestions.append("主导说话人权重过高，考虑降低以获得更好的融合效果")

        if min_weight < 0.01:
            suggestions.append("存在权重过低的说话人，考虑移除或增加权重")

        if len(weights) > 10:
            suggestions.append("说话人数量较多，考虑减少以提高性能")

        # 冲突建议
        if fusion_result.conflicts_resolved:
            suggestions.append("存在参数冲突，建议检查源音色的兼容性")

        # 警告建议
        if fusion_result.warnings:
            suggestions.append("存在警告信息，建议检查配置的有效性")

        return suggestions


# 便捷函数
def fuse_voices_simple(
    voice_manager: VoiceManager,
    voice_ids_and_weights: Dict[str, float],
    fused_name: str,
    save_result: bool = True
) -> FusionResult:
    """简单的音色融合函数

    Args:
        voice_manager: 音色管理器
        voice_ids_and_weights: 音色ID到权重的映射
        fused_name: 融合后音色名称
        save_result: 是否保存结果

    Returns:
        FusionResult: 融合结果
    """
    fuser = VoiceFuser(voice_manager)

    result = fuser.fuse_by_voice_ids(
        voice_ids_and_weights,
        fused_name,
        FusionPresetManager.get_balanced_preset()
    )

    if save_result:
        voice_manager.save_voice(result.fused_voice_config)

    return result


def create_fusion_chain(
    voice_manager: VoiceManager,
    fusion_steps: List[Dict[str, Any]],
    final_name: str
) -> List[FusionResult]:
    """创建融合链

    Args:
        voice_manager: 音色管理器
        fusion_steps: 融合步骤列表，每个步骤包含voice_ids_and_weights
        final_name: 最终音色名称

    Returns:
        融合结果列表
    """
    fuser = VoiceFuser(voice_manager)
    results = []

    current_voice_id = None

    for i, step in enumerate(fusion_steps):
        is_final = (i == len(fusion_steps) - 1)
        step_name = final_name if is_final else f"{final_name}_step_{i+1}"

        # 如果有前一步的结果，加入当前步骤
        voice_weights = step["voice_ids_and_weights"].copy()
        if current_voice_id:
            voice_weights[current_voice_id] = step.get("previous_weight", 0.5)

        result = fuser.fuse_by_voice_ids(
            voice_weights,
            step_name,
            step.get("fusion_config", FusionPresetManager.get_balanced_preset())
        )

        # 保存中间结果
        voice_manager.save_voice(result.fused_voice_config)
        results.append(result)

        current_voice_id = result.fused_voice_config.voice_id

    return results
