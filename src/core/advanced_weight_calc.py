"""高级权重计算引擎

这个模块实现音色融合和继承的复杂权重计算逻辑。
设计原则：
1. 精确算法 - 严格按照用户描述的权重计算公式实现
2. 多层级支持 - 支持继承链和多音色融合
3. 归一化保证 - 确保所有权重总和为1.0
4. 说话人分组 - 按说话人ID正确分组和计算
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import copy
from .models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from .weight_calculator import WeightCalculator, WeightCalculationResult

logger = logging.getLogger(__name__)


@dataclass
class InheritanceWeightResult:
    """继承权重计算结果"""
    ddsp_weights: Dict[str, float] = field(default_factory=dict)
    index_tts_weights: Dict[str, float] = field(default_factory=dict)
    combined_weights: Dict[str, float] = field(default_factory=dict)
    inheritance_ratio: float = 0.0
    source_voice_id: Optional[str] = None

    def __post_init__(self) -> None:
        """验证权重一致性"""
        for weights_dict in [self.ddsp_weights, self.index_tts_weights, self.combined_weights]:
            if weights_dict:
                total = sum(weights_dict.values())
                if abs(total - 1.0) > 1e-6:
                    logger.warning(f"权重总和不为1.0: {total}")


@dataclass
class FusionWeightResult:
    """融合权重计算结果"""
    ddsp_weights: Dict[str, float] = field(default_factory=dict)
    index_tts_weights: Dict[str, float] = field(default_factory=dict)
    combined_weights: Dict[str, float] = field(default_factory=dict)
    source_weights: Dict[str, float] = field(default_factory=dict)
    source_voice_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证权重一致性"""
        for weights_dict in [self.ddsp_weights, self.index_tts_weights, self.combined_weights]:
            if weights_dict:
                total = sum(weights_dict.values())
                if abs(total - 1.0) > 1e-6:
                    logger.warning(f"权重总和不为1.0: {total}")


class AdvancedWeightCalculator:
    """高级权重计算器

    实现音色继承和融合的复杂权重计算算法。
    核心算法：
    1. 继承算法：旧权重 * 继承比例 + 新权重 * (1 - 继承比例)
    2. 融合算法：多个音色按权重加权平均
    3. 分组归一化：按说话人ID分组后分别归一化
    """

    def __init__(self):
        """初始化高级权重计算器"""
        self.base_calculator = WeightCalculator()
        logger.info("高级权重计算器初始化完成")

    def calculate_inheritance_weights(
        self,
        old_voice_config: VoiceConfig,
        new_ddsp_config: DDSPSVCConfig,
        new_index_tts_config: IndexTTSConfig,
        inheritance_ratio: float
    ) -> InheritanceWeightResult:
        """计算继承权重

        Args:
            old_voice_config: 旧音色配置
            new_ddsp_config: 新DDSP配置
            new_index_tts_config: 新IndexTTS配置
            inheritance_ratio: 继承比例 (0.0-1.0)

        Returns:
            InheritanceWeightResult: 继承权重计算结果

        算法实现：
        1. 旧权重 = 原权重 * 继承比例
        2. 新权重 = 新权重 * (1 - 继承比例)
        3. 最终权重 = 旧权重 + 新权重
        4. 按说话人ID分组归一化
        """
        inheritance_ratio = max(0.0, min(1.0, inheritance_ratio))
        new_ratio = 1.0 - inheritance_ratio

        logger.info(f"计算继承权重，继承比例: {inheritance_ratio:.3f}")

        # 提取旧配置的权重
        old_ddsp_weights = old_voice_config.ddsp_config.spk_mix_dict or {}
        old_index_weights = self._extract_index_tts_weights(old_voice_config.index_tts_config)

        # 提取新配置的权重
        new_ddsp_weights = new_ddsp_config.spk_mix_dict or {}
        new_index_weights = self._extract_index_tts_weights(new_index_tts_config)

        # 计算DDSP权重继承
        ddsp_result = self._calculate_inheritance_for_weights(
            old_ddsp_weights, new_ddsp_weights, inheritance_ratio
        )

        # 计算IndexTTS权重继承
        index_result = self._calculate_inheritance_for_weights(
            old_index_weights, new_index_weights, inheritance_ratio
        )

        # 计算组合权重（取DDSP权重为主，IndexTTS为辅）
        combined_weights = self._merge_ddsp_and_index_weights(
            ddsp_result.normalized_weights,
            index_result.normalized_weights
        )

        return InheritanceWeightResult(
            ddsp_weights=ddsp_result.normalized_weights,
            index_tts_weights=index_result.normalized_weights,
            combined_weights=combined_weights,
            inheritance_ratio=inheritance_ratio,
            source_voice_id=old_voice_config.voice_id
        )

    def calculate_fusion_weights(
        self,
        source_voice_configs: Dict[str, VoiceConfig],
        fusion_weights: Dict[str, float]
    ) -> FusionWeightResult:
        """计算融合权重

        Args:
            source_voice_configs: 源音色配置字典 {voice_id: VoiceConfig}
            fusion_weights: 融合权重字典 {voice_id: weight}

        Returns:
            FusionWeightResult: 融合权重计算结果

        算法实现：
        1. 归一化融合权重
        2. 对每个音色的说话人权重按融合权重加权
        3. 按说话人ID分组求和
        4. 最终归一化
        """
        if not source_voice_configs or not fusion_weights:
            raise ValueError("源音色配置和融合权重不能为空")

        logger.info(f"计算融合权重，源音色数量: {len(source_voice_configs)}")

        # 归一化融合权重
        fusion_result = self.base_calculator.normalize_weights(fusion_weights)
        normalized_fusion_weights = fusion_result.normalized_weights

        # 收集所有说话人权重
        ddsp_speaker_weights: Dict[str, float] = {}
        index_speaker_weights: Dict[str, float] = {}

        for voice_id, voice_config in source_voice_configs.items():
            if voice_id not in normalized_fusion_weights:
                continue

            voice_weight = normalized_fusion_weights[voice_id]

            # 处理DDSP权重
            ddsp_weights = voice_config.ddsp_config.spk_mix_dict or {}
            if ddsp_weights:
                # 归一化当前音色的DDSP权重
                ddsp_norm_result = self.base_calculator.normalize_weights(ddsp_weights)
                for speaker_id, speaker_weight in ddsp_norm_result.normalized_weights.items():
                    weighted_value = speaker_weight * voice_weight
                    ddsp_speaker_weights[speaker_id] = ddsp_speaker_weights.get(speaker_id, 0.0) + weighted_value

            # 处理IndexTTS权重
            index_weights = self._extract_index_tts_weights(voice_config.index_tts_config)
            if index_weights:
                # 归一化当前音色的IndexTTS权重
                index_norm_result = self.base_calculator.normalize_weights(index_weights)
                for speaker_id, speaker_weight in index_norm_result.normalized_weights.items():
                    weighted_value = speaker_weight * voice_weight
                    index_speaker_weights[speaker_id] = index_speaker_weights.get(speaker_id, 0.0) + weighted_value

        # 最终归一化
        final_ddsp_result = self.base_calculator.normalize_weights(ddsp_speaker_weights)
        final_index_result = self.base_calculator.normalize_weights(index_speaker_weights)

        # 计算组合权重
        combined_weights = self._merge_ddsp_and_index_weights(
            final_ddsp_result.normalized_weights,
            final_index_result.normalized_weights
        )

        return FusionWeightResult(
            ddsp_weights=final_ddsp_result.normalized_weights,
            index_tts_weights=final_index_result.normalized_weights,
            combined_weights=combined_weights,
            source_weights=normalized_fusion_weights,
            source_voice_ids=list(source_voice_configs.keys())
        )

    def calculate_multi_level_fusion(
        self,
        voice_configs: List[VoiceConfig],
        weights: List[float],
        max_inheritance_depth: int = 5
    ) -> FusionWeightResult:
        """计算多层级融合权重

        处理复杂的继承链，避免循环依赖。

        Args:
            voice_configs: 音色配置列表
            weights: 对应的权重列表
            max_inheritance_depth: 最大继承深度

        Returns:
            FusionWeightResult: 多层级融合结果
        """
        if len(voice_configs) != len(weights):
            raise ValueError("音色配置数量与权重数量不匹配")

        logger.info(f"计算多层级融合，音色数量: {len(voice_configs)}")

        # 构建继承图并检测循环
        inheritance_graph = self._build_inheritance_graph(voice_configs)
        if self._has_circular_dependency(inheritance_graph):
            logger.warning("检测到循环继承依赖，将忽略继承关系")
            # 如果有循环依赖，直接按当前配置融合
            return self._direct_fusion(voice_configs, weights)

        # 按继承深度排序
        sorted_configs = self._sort_by_inheritance_depth(voice_configs, inheritance_graph)

        # 逐层计算融合权重
        return self._calculate_layered_fusion(sorted_configs, weights)

    def _calculate_inheritance_for_weights(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        inheritance_ratio: float
    ) -> WeightCalculationResult:
        """为单组权重计算继承"""
        # 获取所有说话人ID
        all_speakers = set(old_weights.keys()) | set(new_weights.keys())

        # 计算继承权重
        inherited_weights = {}
        for speaker in all_speakers:
            old_w = old_weights.get(speaker, 0.0)
            new_w = new_weights.get(speaker, 0.0)

            # 核心算法：旧权重 * 继承比例 + 新权重 * (1 - 继承比例)
            inherited_w = old_w * inheritance_ratio + new_w * (1.0 - inheritance_ratio)

            if inherited_w > 0:
                inherited_weights[speaker] = inherited_w

        # 归一化
        return self.base_calculator.normalize_weights(inherited_weights)

    def _extract_index_tts_weights(self, config: IndexTTSConfig) -> Dict[str, float]:
        """从IndexTTS配置提取权重

        IndexTTS主要通过speaker_name控制，这里简化处理
        """
        if config.speaker_name:
            return {config.speaker_name: 1.0}
        return {}

    def _merge_ddsp_and_index_weights(
        self,
        ddsp_weights: Dict[str, float],
        index_weights: Dict[str, float],
        ddsp_priority: float = 0.7
    ) -> Dict[str, float]:
        """合并DDSP和IndexTTS权重

        Args:
            ddsp_weights: DDSP权重
            index_weights: IndexTTS权重
            ddsp_priority: DDSP权重优先级
        """
        if not ddsp_weights and not index_weights:
            return {}

        if not ddsp_weights:
            return index_weights.copy()

        if not index_weights:
            return ddsp_weights.copy()

        # 合并权重，DDSP优先
        all_speakers = set(ddsp_weights.keys()) | set(index_weights.keys())
        merged_weights = {}

        for speaker in all_speakers:
            ddsp_w = ddsp_weights.get(speaker, 0.0)
            index_w = index_weights.get(speaker, 0.0)

            # 加权平均
            merged_w = ddsp_w * ddsp_priority + index_w * (1.0 - ddsp_priority)
            if merged_w > 0:
                merged_weights[speaker] = merged_w

        # 归一化
        result = self.base_calculator.normalize_weights(merged_weights)
        return result.normalized_weights

    def _build_inheritance_graph(self, voice_configs: List[VoiceConfig]) -> Dict[str, List[str]]:
        """构建继承关系图"""
        graph = {}
        for config in voice_configs:
            graph[config.voice_id] = config.parent_voice_ids.copy()
        return graph

    def _has_circular_dependency(self, graph: Dict[str, List[str]]) -> bool:
        """检测循环依赖"""
        def dfs(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        for node in graph:
            if node not in visited:
                if dfs(node, visited, set()):
                    return True
        return False

    def _sort_by_inheritance_depth(
        self,
        voice_configs: List[VoiceConfig],
        graph: Dict[str, List[str]]
    ) -> List[VoiceConfig]:
        """按继承深度排序"""
        def get_depth(voice_id: str, memo: Dict[str, int] = {}) -> int:
            if voice_id in memo:
                return memo[voice_id]

            parents = graph.get(voice_id, [])
            if not parents:
                memo[voice_id] = 0
                return 0

            max_parent_depth = max(get_depth(parent, memo) for parent in parents)
            depth = max_parent_depth + 1
            memo[voice_id] = depth
            return depth

        return sorted(voice_configs, key=lambda config: get_depth(config.voice_id))

    def _direct_fusion(self, voice_configs: List[VoiceConfig], weights: List[float]) -> FusionWeightResult:
        """直接融合（忽略继承关系）"""
        config_dict = {config.voice_id: config for config in voice_configs}
        weight_dict = {config.voice_id: weight for config, weight in zip(voice_configs, weights)}

        return self.calculate_fusion_weights(config_dict, weight_dict)

    def _calculate_layered_fusion(
        self,
        sorted_configs: List[VoiceConfig],
        weights: List[float]
    ) -> FusionWeightResult:
        """分层计算融合权重"""
        # 简化实现：直接融合
        # 在实际应用中，这里可以实现更复杂的分层逻辑
        return self._direct_fusion(sorted_configs, weights)

    def validate_weights_consistency(
        self,
        ddsp_weights: Dict[str, float],
        index_weights: Dict[str, float],
        tolerance: float = 1e-6
    ) -> Tuple[bool, List[str]]:
        """验证权重一致性

        Args:
            ddsp_weights: DDSP权重
            index_weights: IndexTTS权重
            tolerance: 容差

        Returns:
            (是否一致, 错误信息列表)
        """
        errors = []

        # 检查权重总和
        ddsp_total = sum(ddsp_weights.values()) if ddsp_weights else 0.0
        index_total = sum(index_weights.values()) if index_weights else 0.0

        if ddsp_weights and abs(ddsp_total - 1.0) > tolerance:
            errors.append(f"DDSP权重总和不为1.0: {ddsp_total:.6f}")

        if index_weights and abs(index_total - 1.0) > tolerance:
            errors.append(f"IndexTTS权重总和不为1.0: {index_total:.6f}")

        # 检查权重值范围
        for weights_dict, name in [(ddsp_weights, "DDSP"), (index_weights, "IndexTTS")]:
            for speaker, weight in weights_dict.items():
                if weight < 0:
                    errors.append(f"{name}权重为负数: {speaker}={weight}")
                if weight > 1.0 + tolerance:
                    errors.append(f"{name}权重超过1.0: {speaker}={weight}")

        return len(errors) == 0, errors

    def optimize_speaker_selection(
        self,
        weights: Dict[str, float],
        max_speakers: int = 5,
        min_weight_threshold: float = 0.01
    ) -> Dict[str, float]:
        """优化说话人选择

        Args:
            weights: 原始权重
            max_speakers: 最大说话人数量
            min_weight_threshold: 最小权重阈值

        Returns:
            优化后的权重字典
        """
        if not weights:
            return {}

        # 过滤低权重说话人
        filtered_weights = {
            speaker: weight
            for speaker, weight in weights.items()
            if weight >= min_weight_threshold
        }

        if len(filtered_weights) <= max_speakers:
            result = self.base_calculator.normalize_weights(filtered_weights)
            return result.normalized_weights

        # 选择权重最大的说话人
        sorted_speakers = sorted(
            filtered_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected_weights = dict(sorted_speakers[:max_speakers])
        result = self.base_calculator.normalize_weights(selected_weights)

        logger.info(f"优化说话人选择：{len(weights)} -> {len(selected_weights)}")
        return result.normalized_weights


class WeightVisualization:
    """权重可视化工具

    提供权重分布的可视化和分析功能。
    """

    @staticmethod
    def format_weights_table(weights: Dict[str, float]) -> str:
        """格式化权重表格"""
        if not weights:
            return "无权重数据"

        lines = ["说话人ID | 权重 | 百分比"]
        lines.append("-" * 30)

        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for speaker, weight in sorted_weights:
            percentage = weight * 100
            lines.append(f"{speaker:10} | {weight:.4f} | {percentage:6.2f}%")

        total = sum(weights.values())
        lines.append("-" * 30)
        lines.append(f"{'总计':10} | {total:.4f} | {total*100:6.2f}%")

        return "\n".join(lines)

    @staticmethod
    def analyze_weight_distribution(weights: Dict[str, float]) -> Dict[str, Any]:
        """分析权重分布"""
        if not weights:
            return {"error": "无权重数据"}

        values = list(weights.values())
        return {
            "speaker_count": len(weights),
            "total_weight": sum(values),
            "max_weight": max(values),
            "min_weight": min(values),
            "avg_weight": sum(values) / len(values),
            "dominant_speaker": max(weights.items(), key=lambda x: x[1])[0],
            "weight_variance": sum((w - sum(values)/len(values))**2 for w in values) / len(values)
        }
