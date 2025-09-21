"""权重计算系统

这个模块实现音色融合的权重计算逻辑。
设计原则：
1. 用户输入任意数字，系统自动归一化
2. 支持多层级权重计算
3. 权重总和始终为1.0，消除特殊情况
"""

from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WeightCalculationResult:
    """权重计算结果"""
    normalized_weights: Dict[str, float]
    original_weights: Dict[str, float]
    total_original: float
    scaling_factor: float

    def __post_init__(self) -> None:
        """验证结果的一致性"""
        total_normalized = sum(self.normalized_weights.values())
        if abs(total_normalized - 1.0) > 1e-6:
            logger.warning(f"归一化权重总和不为1.0: {total_normalized}")


class WeightCalculator:
    """权重计算器

    核心功能：
    1. 将任意数字归一化为权重
    2. 支持多层级权重合并
    3. 处理零权重和负权重
    """

    @staticmethod
    def normalize_weights(weights: Dict[str, float]) -> WeightCalculationResult:
        """归一化权重

        Args:
            weights: 原始权重字典，值可以是任意正数

        Returns:
            WeightCalculationResult: 包含归一化结果的对象

        设计原则：
        - 负数权重自动设为0
        - 所有权重为0时，平均分配
        - 权重总和始终为1.0
        """
        if not weights:
            return WeightCalculationResult({}, {}, 0.0, 0.0)

        # 处理负权重：设为0
        cleaned_weights = {k: max(0.0, v) for k, v in weights.items()}
        total = sum(cleaned_weights.values())

        # 特殊情况：所有权重都是0，平均分配
        if total == 0.0:
            count = len(cleaned_weights)
            normalized = {k: 1.0 / count for k in cleaned_weights.keys()}
            return WeightCalculationResult(
                normalized_weights=normalized,
                original_weights=weights.copy(),
                total_original=0.0,
                scaling_factor=float('inf')  # 表示从0缩放到1
            )

        # 正常情况：按比例归一化
        scaling_factor = 1.0 / total
        normalized = {k: v * scaling_factor for k, v in cleaned_weights.items()}

        return WeightCalculationResult(
            normalized_weights=normalized,
            original_weights=weights.copy(),
            total_original=total,
            scaling_factor=scaling_factor
        )

    @staticmethod
    def merge_weights(
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        merge_ratio: float = 0.5
    ) -> WeightCalculationResult:
        """合并两组权重

        Args:
            old_weights: 旧权重字典
            new_weights: 新权重字典
            merge_ratio: 新权重的比例 (0.0-1.0)

        Returns:
            合并后的权重计算结果

        算法：
        merged_weight = old_weight * (1 - merge_ratio) + new_weight * merge_ratio
        """
        merge_ratio = max(0.0, min(1.0, merge_ratio))  # 限制在[0,1]范围

        # 获取所有说话人ID
        all_speakers = set(old_weights.keys()) | set(new_weights.keys())

        # 计算合并权重
        merged = {}
        for speaker in all_speakers:
            old_w = old_weights.get(speaker, 0.0)
            new_w = new_weights.get(speaker, 0.0)
            merged[speaker] = old_w * (1.0 - merge_ratio) + new_w * merge_ratio

        return WeightCalculator.normalize_weights(merged)

    @staticmethod
    def calculate_speaker_groups(
        speaker_weights: Dict[str, float],
        group_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """按组计算说话人权重

        Args:
            speaker_weights: 说话人权重
            group_mapping: 说话人到组的映射，如果为None则每个说话人一组

        Returns:
            按组分类的权重字典
        """
        if group_mapping is None:
            # 每个说话人单独一组
            return {speaker: {speaker: weight}
                   for speaker, weight in speaker_weights.items()}

        groups: Dict[str, Dict[str, float]] = {}
        for speaker, weight in speaker_weights.items():
            group = group_mapping.get(speaker, speaker)  # 默认组名为说话人名
            if group not in groups:
                groups[group] = {}
            groups[group][speaker] = weight

        return groups

    @staticmethod
    def interpolate_weights(
        start_weights: Dict[str, float],
        end_weights: Dict[str, float],
        steps: int
    ) -> List[WeightCalculationResult]:
        """权重插值

        在两组权重之间生成平滑过渡的权重序列。
        用于动态权重变化或权重动画。

        Args:
            start_weights: 起始权重
            end_weights: 结束权重
            steps: 插值步数

        Returns:
            权重插值序列
        """
        if steps < 2:
            return [WeightCalculator.normalize_weights(start_weights)]

        results = []
        for i in range(steps):
            ratio = i / (steps - 1)  # 0.0 到 1.0
            interpolated = WeightCalculator.merge_weights(
                start_weights, end_weights, ratio
            )
            results.append(interpolated)

        return results

    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> Tuple[bool, List[str]]:
        """验证权重的有效性

        Args:
            weights: 待验证的权重字典

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []

        if not weights:
            errors.append("权重字典为空")
            return False, errors

        # 检查权重值
        for speaker, weight in weights.items():
            if not isinstance(weight, (int, float)):
                errors.append(f"说话人 '{speaker}' 的权重不是数字: {type(weight)}")
            elif weight < 0:
                errors.append(f"说话人 '{speaker}' 的权重为负数: {weight}")

        # 检查说话人ID
        for speaker in weights.keys():
            if not isinstance(speaker, str) or not speaker.strip():
                errors.append(f"无效的说话人ID: '{speaker}'")

        return len(errors) == 0, errors

    @staticmethod
    def optimize_weights(
        target_weights: Dict[str, float],
        available_speakers: List[str],
        max_speakers: int = 5
    ) -> WeightCalculationResult:
        """优化权重分配

        当目标权重中的说话人数量超过限制时，选择权重最大的说话人。

        Args:
            target_weights: 目标权重
            available_speakers: 可用的说话人列表
            max_speakers: 最大说话人数量

        Returns:
            优化后的权重结果
        """
        # 过滤出可用的说话人
        filtered_weights = {
            speaker: weight
            for speaker, weight in target_weights.items()
            if speaker in available_speakers and weight > 0
        }

        if len(filtered_weights) <= max_speakers:
            return WeightCalculator.normalize_weights(filtered_weights)

        # 选择权重最大的说话人
        sorted_speakers = sorted(
            filtered_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        optimized_weights = dict(sorted_speakers[:max_speakers])
        return WeightCalculator.normalize_weights(optimized_weights)


class WeightPreset:
    """权重预设管理

    提供常用的权重配置模板。
    """

    @staticmethod
    def equal_weights(speakers: List[str]) -> Dict[str, float]:
        """等权重分配"""
        if not speakers:
            return {}
        weight = 1.0 / len(speakers)
        return {speaker: weight for speaker in speakers}

    @staticmethod
    def dominant_speaker(
        dominant: str,
        others: List[str],
        dominant_ratio: float = 0.7
    ) -> Dict[str, float]:
        """主导说话人权重分配

        Args:
            dominant: 主导说话人
            others: 其他说话人
            dominant_ratio: 主导说话人的权重比例
        """
        if not others:
            return {dominant: 1.0}

        other_ratio = (1.0 - dominant_ratio) / len(others)
        weights = {dominant: dominant_ratio}
        weights.update({speaker: other_ratio for speaker in others})

        return weights

    @staticmethod
    def gradual_fade(
        speakers: List[str],
        fade_factor: float = 0.7
    ) -> Dict[str, float]:
        """渐变权重分配

        第一个说话人权重最高，后续按fade_factor递减。
        """
        if not speakers:
            return {}

        weights = {}
        current_weight = 1.0
        total = 0.0

        # 计算原始权重
        for speaker in speakers:
            weights[speaker] = current_weight
            total += current_weight
            current_weight *= fade_factor

        # 归一化
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
