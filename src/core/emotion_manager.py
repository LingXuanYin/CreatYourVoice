"""情感参数管理器

这个模块负责情感参数的管理和转换。
设计原则：
1. 简洁直接 - 8维情感向量作为核心数据结构
2. 无特殊情况 - 所有情感控制都转换为向量
3. 缓存优化 - 避免重复处理相同的参考音频
"""

import os
import sys
import json
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# 添加IndexTTS到Python路径
INDEX_TTS_PATH = Path(__file__).parent.parent.parent / "index-tts"
if str(INDEX_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(INDEX_TTS_PATH))

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError as e:
    logging.warning(f"无法导入IndexTTS模块，情感分析功能将受限: {e}")
    IndexTTS2 = None

logger = logging.getLogger(__name__)


@dataclass
class EmotionVector:
    """8维情感向量

    IndexTTS v2的标准情感维度：
    [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    """
    happy: float = 0.0
    angry: float = 0.0
    sad: float = 0.0
    afraid: float = 0.0
    disgusted: float = 0.0
    melancholic: float = 0.0
    surprised: float = 0.0
    calm: float = 0.0

    def __post_init__(self):
        """确保所有值在有效范围内"""
        for field_name in ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']:
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(1.0, float(value))))

    def to_list(self) -> List[float]:
        """转换为列表格式"""
        return [self.happy, self.angry, self.sad, self.afraid,
                self.disgusted, self.melancholic, self.surprised, self.calm]

    @classmethod
    def from_list(cls, values: List[float]) -> "EmotionVector":
        """从列表创建情感向量"""
        if len(values) != 8:
            raise ValueError(f"情感向量必须包含8个值，当前为{len(values)}")

        return cls(
            happy=values[0], angry=values[1], sad=values[2], afraid=values[3],
            disgusted=values[4], melancholic=values[5], surprised=values[6], calm=values[7]
        )

    def normalize(self) -> "EmotionVector":
        """归一化情感向量，确保总和为1.0"""
        values = self.to_list()
        total = sum(values)

        if total == 0:
            # 如果所有值都为0，设置为平静状态
            return EmotionVector(calm=1.0)

        normalized_values = [v / total for v in values]
        return EmotionVector.from_list(normalized_values)

    def blend(self, other: "EmotionVector", weight: float = 0.5) -> "EmotionVector":
        """与另一个情感向量混合"""
        weight = max(0.0, min(1.0, weight))
        self_values = self.to_list()
        other_values = other.to_list()

        blended_values = [
            self_values[i] * (1 - weight) + other_values[i] * weight
            for i in range(8)
        ]

        return EmotionVector.from_list(blended_values)


@dataclass
class EmotionPreset:
    """情感预设"""
    name: str
    description: str
    emotion_vector: EmotionVector
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class EmotionManagerError(Exception):
    """情感管理器异常基类"""
    pass


class EmotionAnalysisError(EmotionManagerError):
    """情感分析异常"""
    pass


class EmotionManager:
    """情感参数管理器

    负责情感向量的创建、转换和管理。
    核心功能：
    1. 情感参考音频转向量
    2. 情感描述文本转向量
    3. 情感预设管理
    4. 情感向量缓存
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        presets_file: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """初始化情感管理器

        Args:
            model_dir: IndexTTS模型目录
            presets_file: 情感预设文件路径
            cache_dir: 缓存目录
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.presets_file = Path(presets_file) if presets_file else Path("src/data/emotion_presets.yaml")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/emotions")

        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # IndexTTS模型实例（延迟加载）
        self._model: Optional[Any] = None
        self._model_loaded = False

        # 情感预设缓存
        self._presets: Dict[str, EmotionPreset] = {}
        self._presets_loaded = False

        # 音频情感向量缓存
        self._audio_cache: Dict[str, EmotionVector] = {}

        logger.info(f"情感管理器初始化完成，缓存目录: {self.cache_dir}")

    def _load_model(self) -> None:
        """加载IndexTTS模型"""
        if self._model_loaded or not self.model_dir:
            return

        if IndexTTS2 is None:
            raise EmotionManagerError("IndexTTS模块未安装，无法进行情感分析")

        try:
            logger.info(f"加载IndexTTS模型: {self.model_dir}")

            self._model = IndexTTS2(
                cfg_path=str(self.model_dir / "config.yaml"),
                model_dir=str(self.model_dir),
                use_fp16=False,
                device=None,
                use_cuda_kernel=False,
                use_deepspeed=False
            )

            self._model_loaded = True
            logger.info("IndexTTS模型加载成功")

        except Exception as e:
            raise EmotionManagerError(f"加载IndexTTS模型失败: {e}")

    def _load_presets(self) -> None:
        """加载情感预设"""
        if self._presets_loaded:
            return

        if not self.presets_file.exists():
            logger.warning(f"情感预设文件不存在: {self.presets_file}")
            self._presets_loaded = True
            return

        try:
            import yaml
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            for preset_data in data.get('presets', []):
                emotion_vector = EmotionVector.from_list(preset_data['emotion_vector'])
                preset = EmotionPreset(
                    name=preset_data['name'],
                    description=preset_data['description'],
                    emotion_vector=emotion_vector,
                    tags=preset_data.get('tags', [])
                )
                self._presets[preset.name] = preset

            self._presets_loaded = True
            logger.info(f"加载了 {len(self._presets)} 个情感预设")

        except Exception as e:
            logger.error(f"加载情感预设失败: {e}")
            self._presets_loaded = True

    def _get_audio_cache_key(self, audio_path: Union[str, Path]) -> str:
        """生成音频缓存键"""
        audio_path = Path(audio_path)
        # 使用文件路径和修改时间作为缓存键
        mtime = audio_path.stat().st_mtime if audio_path.exists() else 0
        return f"{audio_path.name}_{int(mtime)}"

    def extract_emotion_from_audio(
        self,
        audio_path: Union[str, Path],
        use_cache: bool = True
    ) -> EmotionVector:
        """从音频提取情感向量

        Args:
            audio_path: 音频文件路径
            use_cache: 是否使用缓存

        Returns:
            情感向量

        Raises:
            EmotionAnalysisError: 情感分析失败
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise EmotionAnalysisError(f"音频文件不存在: {audio_path}")

        # 检查缓存
        cache_key = self._get_audio_cache_key(audio_path)
        if use_cache and cache_key in self._audio_cache:
            logger.debug(f"使用缓存的情感向量: {audio_path}")
            return self._audio_cache[cache_key]

        try:
            # 加载模型
            self._load_model()
            if not self._model:
                raise EmotionAnalysisError("IndexTTS模型未加载")

            logger.info(f"分析音频情感: {audio_path}")

            # 使用IndexTTS提取情感特征
            # 这里需要调用IndexTTS的情感分析功能
            if hasattr(self._model, 'extract_emotion_vector'):
                emotion_values = self._model.extract_emotion_vector(str(audio_path))
            else:
                # 如果没有直接的情感提取方法，使用推理方法间接获取
                # 这是一个简化的实现，实际可能需要更复杂的处理
                logger.warning("使用简化的情感分析方法")
                emotion_values = self._analyze_audio_simple(audio_path)

            emotion_vector = EmotionVector.from_list(emotion_values)

            # 缓存结果
            if use_cache:
                self._audio_cache[cache_key] = emotion_vector

            logger.info(f"音频情感分析完成: {emotion_vector.to_list()}")
            return emotion_vector

        except Exception as e:
            raise EmotionAnalysisError(f"音频情感分析失败: {audio_path}, 错误: {e}")

    def _analyze_audio_simple(self, audio_path: Path) -> List[float]:
        """简化的音频情感分析

        这是一个备用方法，当IndexTTS没有直接的情感提取功能时使用。
        基于音频的基本特征进行简单的情感估计。
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=22050)

            # 提取基本特征
            # 音调特征
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0

            # 能量特征
            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms)
            energy_var = np.var(rms)

            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_mean = np.mean(spectral_centroids)

            # 基于特征的简单情感映射
            # 这是一个非常简化的映射，实际应用中需要更复杂的模型
            emotion_values = [0.0] * 8  # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

            # 高音调 + 高能量 -> 高兴/惊讶
            if pitch_mean > 200 and energy_mean > 0.1:
                emotion_values[0] = 0.6  # happy
                emotion_values[6] = 0.3  # surprised
            # 低音调 + 低能量 -> 悲伤/忧郁
            elif pitch_mean < 150 and energy_mean < 0.05:
                emotion_values[2] = 0.5  # sad
                emotion_values[5] = 0.4  # melancholic
            # 高能量变化 -> 愤怒
            elif energy_var > 0.01:
                emotion_values[1] = 0.7  # angry
            # 默认平静
            else:
                emotion_values[7] = 0.8  # calm

            # 归一化
            total = sum(emotion_values)
            if total > 0:
                emotion_values = [float(v) / total for v in emotion_values]
            else:
                emotion_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 默认平静

            return emotion_values

        except Exception as e:
            logger.warning(f"简化情感分析失败: {e}")
            # 返回默认的平静状态
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def analyze_emotion_from_text(self, text: str) -> EmotionVector:
        """从文本分析情感

        Args:
            text: 输入文本

        Returns:
            情感向量

        Raises:
            EmotionAnalysisError: 情感分析失败
        """
        try:
            # 加载模型
            self._load_model()
            if not self._model:
                raise EmotionAnalysisError("IndexTTS模型未加载")

            logger.info(f"分析文本情感: {text[:50]}...")

            # 使用IndexTTS的文本情感分析
            if hasattr(self._model, 'qwen_emo') and hasattr(self._model.qwen_emo, 'inference'):
                emotion_dict = self._model.qwen_emo.inference(text)

                # 转换为8维向量
                emotion_values = [
                    emotion_dict.get('happy', 0.0),
                    emotion_dict.get('angry', 0.0),
                    emotion_dict.get('sad', 0.0),
                    emotion_dict.get('afraid', 0.0),
                    emotion_dict.get('disgusted', 0.0),
                    emotion_dict.get('melancholic', 0.0),
                    emotion_dict.get('surprised', 0.0),
                    emotion_dict.get('calm', 0.0)
                ]
            else:
                # 简化的文本情感分析
                emotion_values = self._analyze_text_simple(text)

            emotion_vector = EmotionVector.from_list(emotion_values)
            logger.info(f"文本情感分析完成: {emotion_vector.to_list()}")
            return emotion_vector

        except Exception as e:
            raise EmotionAnalysisError(f"文本情感分析失败: {e}")

    def _analyze_text_simple(self, text: str) -> List[float]:
        """简化的文本情感分析

        基于关键词的简单情感分析，作为备用方法。
        """
        text_lower = text.lower()

        # 情感关键词字典
        emotion_keywords = {
            'happy': ['开心', '高兴', '快乐', '愉快', '兴奋', '欢乐', '喜悦', 'happy', 'joy', 'excited'],
            'angry': ['愤怒', '生气', '恼火', '愤慨', '怒', 'angry', 'mad', 'furious'],
            'sad': ['悲伤', '难过', '伤心', '痛苦', '哀伤', 'sad', 'sorrow', 'grief'],
            'afraid': ['害怕', '恐惧', '担心', '紧张', '焦虑', 'afraid', 'fear', 'anxious'],
            'disgusted': ['厌恶', '恶心', '讨厌', '反感', 'disgusted', 'disgusting'],
            'melancholic': ['忧郁', '沮丧', '低落', '郁闷', 'melancholic', 'depressed'],
            'surprised': ['惊讶', '震惊', '意外', '吃惊', 'surprised', 'shocked'],
            'calm': ['平静', '冷静', '安静', '淡定', 'calm', 'peaceful', 'quiet']
        }

        # 计算每种情感的得分
        emotion_scores = [0.0] * 8
        emotion_names = ['happy', 'angry', 'sad', 'afraid', 'disgusted', 'melancholic', 'surprised', 'calm']

        for i, emotion_name in enumerate(emotion_names):
            keywords = emotion_keywords.get(emotion_name, [])
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[i] += 1.0

        # 如果没有匹配到任何关键词，默认为平静
        if sum(emotion_scores) == 0:
            emotion_scores[7] = 1.0  # calm
        else:
            # 归一化
            total = sum(emotion_scores)
            emotion_scores = [score / total for score in emotion_scores]

        return emotion_scores

    def get_preset(self, name: str) -> Optional[EmotionPreset]:
        """获取情感预设

        Args:
            name: 预设名称

        Returns:
            情感预设，如果不存在则返回None
        """
        self._load_presets()
        return self._presets.get(name)

    def list_presets(self) -> List[EmotionPreset]:
        """列出所有情感预设

        Returns:
            情感预设列表
        """
        self._load_presets()
        return list(self._presets.values())

    def add_preset(self, preset: EmotionPreset) -> None:
        """添加情感预设

        Args:
            preset: 情感预设
        """
        self._load_presets()
        self._presets[preset.name] = preset
        self._save_presets()

    def remove_preset(self, name: str) -> bool:
        """移除情感预设

        Args:
            name: 预设名称

        Returns:
            是否成功移除
        """
        self._load_presets()
        if name in self._presets:
            del self._presets[name]
            self._save_presets()
            return True
        return False

    def _save_presets(self) -> None:
        """保存情感预设到文件"""
        try:
            import yaml

            # 准备数据
            presets_data = {
                'presets': []
            }

            for preset in self._presets.values():
                preset_data = {
                    'name': preset.name,
                    'description': preset.description,
                    'emotion_vector': preset.emotion_vector.to_list(),
                    'tags': preset.tags
                }
                presets_data['presets'].append(preset_data)

            # 确保目录存在
            self.presets_file.parent.mkdir(parents=True, exist_ok=True)

            # 保存文件
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                yaml.dump(presets_data, f, default_flow_style=False, allow_unicode=True, indent=2)

            logger.info(f"情感预设已保存: {self.presets_file}")

        except Exception as e:
            logger.error(f"保存情感预设失败: {e}")

    def create_emotion_vector(
        self,
        happy: float = 0.0,
        angry: float = 0.0,
        sad: float = 0.0,
        afraid: float = 0.0,
        disgusted: float = 0.0,
        melancholic: float = 0.0,
        surprised: float = 0.0,
        calm: float = 0.0,
        normalize: bool = True
    ) -> EmotionVector:
        """创建情感向量

        Args:
            happy: 高兴程度 (0.0-1.0)
            angry: 愤怒程度 (0.0-1.0)
            sad: 悲伤程度 (0.0-1.0)
            afraid: 恐惧程度 (0.0-1.0)
            disgusted: 厌恶程度 (0.0-1.0)
            melancholic: 忧郁程度 (0.0-1.0)
            surprised: 惊讶程度 (0.0-1.0)
            calm: 平静程度 (0.0-1.0)
            normalize: 是否归一化

        Returns:
            情感向量
        """
        emotion_vector = EmotionVector(
            happy=happy, angry=angry, sad=sad, afraid=afraid,
            disgusted=disgusted, melancholic=melancholic,
            surprised=surprised, calm=calm
        )

        if normalize:
            emotion_vector = emotion_vector.normalize()

        return emotion_vector

    def blend_emotions(
        self,
        emotions: List[Tuple[EmotionVector, float]]
    ) -> EmotionVector:
        """混合多个情感向量

        Args:
            emotions: 情感向量和权重的列表 [(emotion_vector, weight), ...]

        Returns:
            混合后的情感向量
        """
        if not emotions:
            return EmotionVector(calm=1.0)

        if len(emotions) == 1:
            return emotions[0][0]

        # 归一化权重
        total_weight = sum(weight for _, weight in emotions)
        if total_weight == 0:
            return EmotionVector(calm=1.0)

        # 计算加权平均
        result_values = [0.0] * 8
        for emotion_vector, weight in emotions:
            normalized_weight = weight / total_weight
            emotion_values = emotion_vector.to_list()
            for i in range(8):
                result_values[i] += emotion_values[i] * normalized_weight

        return EmotionVector.from_list(result_values)

    def clear_cache(self) -> None:
        """清理缓存"""
        self._audio_cache.clear()
        logger.info("情感管理器缓存已清理")

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息

        Returns:
            缓存信息字典
        """
        return {
            "audio_cache_size": len(self._audio_cache),
            "presets_count": len(self._presets),
            "model_loaded": self._model_loaded,
            "cache_dir": str(self.cache_dir)
        }


# 便捷函数
def create_emotion_vector(
    happy: float = 0.0,
    angry: float = 0.0,
    sad: float = 0.0,
    afraid: float = 0.0,
    disgusted: float = 0.0,
    melancholic: float = 0.0,
    surprised: float = 0.0,
    calm: float = 0.0
) -> EmotionVector:
    """便捷的情感向量创建函数"""
    return EmotionVector(
        happy=happy, angry=angry, sad=sad, afraid=afraid,
        disgusted=disgusted, melancholic=melancholic,
        surprised=surprised, calm=calm
    )


def extract_emotion_from_audio(
    audio_path: Union[str, Path],
    model_dir: Optional[Union[str, Path]] = None
) -> EmotionVector:
    """便捷的音频情感提取函数"""
    manager = EmotionManager(model_dir=model_dir)
    return manager.extract_emotion_from_audio(audio_path)


def analyze_emotion_from_text(
    text: str,
    model_dir: Optional[Union[str, Path]] = None
) -> EmotionVector:
    """便捷的文本情感分析函数"""
    manager = EmotionManager(model_dir=model_dir)
    return manager.analyze_emotion_from_text(text)
