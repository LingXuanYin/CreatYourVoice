"""语音合成器

这个模块是音色使用和语音合成的核心组件。
设计原则：
1. 简洁工作流 - 音色选择 → 文本输入 → 情感控制 → 语音合成
2. 统一接口 - 隐藏复杂的情感处理和分句逻辑
3. 完整记录 - 保存所有合成参数，支持重现和调整
"""

import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .models import VoiceConfig
from .voice_manager import VoiceManager, VoiceNotFoundError
from .emotion_manager import EmotionManager, EmotionVector, EmotionManagerError
from ..integrations.index_tts import IndexTTSIntegration, IndexTTSResult, InferenceError
from ..utils.audio_utils import AudioProcessor, AudioProcessingError

logger = logging.getLogger(__name__)


@dataclass
class SynthesisParams:
    """语音合成参数"""
    # 基础参数
    text: str
    voice_id: str

    # 情感控制参数
    emotion_mode: str = "speaker"  # "speaker", "reference", "vector", "text", "preset"
    emotion_reference_audio: Optional[str] = None
    emotion_vector: Optional[List[float]] = None
    emotion_text: Optional[str] = None
    emotion_preset: Optional[str] = None
    emotion_weight: float = 0.65

    # 生成参数
    speed: float = 1.0
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    max_text_tokens_per_segment: int = 120
    interval_silence: int = 200

    # 音频处理参数
    normalize_audio: bool = True
    trim_silence: bool = True
    apply_fade: bool = True
    fade_duration: float = 0.1


@dataclass
class SynthesisResult:
    """语音合成结果"""
    # 基础信息
    synthesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = False
    error_message: Optional[str] = None

    # 音频结果
    audio_path: Optional[str] = None
    audio_data: Optional[Tuple[int, Any]] = None  # (sample_rate, audio_array)

    # 处理信息
    processing_time: float = 0.0
    segments_count: int = 0
    text_length: int = 0

    # 参数记录
    synthesis_params: Optional[SynthesisParams] = None
    voice_config: Optional[VoiceConfig] = None
    final_emotion_vector: Optional[List[float]] = None

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)


class VoiceSynthesizerError(Exception):
    """语音合成器异常基类"""
    pass


class TextProcessingError(VoiceSynthesizerError):
    """文本处理异常"""
    pass


class VoiceSynthesizer:
    """语音合成器

    整合音色管理、情感控制和语音合成的核心组件。
    工作流程：
    1. 加载音色配置
    2. 处理情感参数
    3. 分句处理长文本
    4. 执行语音合成
    5. 后处理音频
    6. 保存结果和参数
    """

    def __init__(
        self,
        voice_manager: Optional[VoiceManager] = None,
        emotion_manager: Optional[EmotionManager] = None,
        index_tts_integration: Optional[IndexTTSIntegration] = None,
        output_dir: Union[str, Path] = "outputs/synthesis",
        temp_dir: Union[str, Path] = "temp/synthesis"
    ):
        """初始化语音合成器

        Args:
            voice_manager: 音色管理器
            emotion_manager: 情感管理器
            index_tts_integration: IndexTTS集成
            output_dir: 输出目录
            temp_dir: 临时目录
        """
        self.voice_manager = voice_manager or VoiceManager()
        self.emotion_manager = emotion_manager or EmotionManager()
        self.index_tts = index_tts_integration or IndexTTSIntegration()

        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 音频处理器
        self.audio_processor = AudioProcessor()

        logger.info(f"语音合成器初始化完成，输出目录: {self.output_dir}")

    def synthesize(
        self,
        params: SynthesisParams,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> SynthesisResult:
        """执行语音合成

        Args:
            params: 合成参数
            output_path: 输出路径，None则自动生成
            progress_callback: 进度回调函数

        Returns:
            合成结果
        """
        start_time = time.time()
        result = SynthesisResult(synthesis_params=params, text_length=len(params.text))

        try:
            if progress_callback:
                progress_callback(0.1, "加载音色配置...")

            # 1. 加载音色配置
            voice_config = self._load_voice_config(params.voice_id)
            result.voice_config = voice_config

            if progress_callback:
                progress_callback(0.2, "处理情感参数...")

            # 2. 处理情感参数
            emotion_vector = self._process_emotion_params(params)
            result.final_emotion_vector = emotion_vector.to_list()

            if progress_callback:
                progress_callback(0.3, "处理文本分句...")

            # 3. 文本预处理和分句
            text_segments = self._process_text(params.text, params.max_text_tokens_per_segment)
            result.segments_count = len(text_segments)

            if progress_callback:
                progress_callback(0.4, "准备音频参考...")

            # 4. 准备音频参考
            speaker_audio = self._get_speaker_audio(voice_config)
            emotion_audio = self._get_emotion_audio(params, emotion_vector)

            if progress_callback:
                progress_callback(0.5, "开始语音合成...")

            # 5. 执行合成
            audio_results = []
            for i, segment in enumerate(text_segments):
                if progress_callback:
                    segment_progress = 0.5 + 0.4 * (i + 1) / len(text_segments)
                    progress_callback(segment_progress, f"合成第 {i+1}/{len(text_segments)} 段...")

                segment_result = self._synthesize_segment(
                    segment, speaker_audio, emotion_audio, emotion_vector, params
                )
                audio_results.append(segment_result)

            if progress_callback:
                progress_callback(0.9, "后处理音频...")

            # 6. 拼接和后处理音频
            final_audio = self._post_process_audio(audio_results, params)

            # 7. 保存音频
            if output_path is None:
                output_path = self._generate_output_path(params)

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if final_audio is not None:
                sample_rate, audio_data = final_audio
                self.audio_processor.save_audio(audio_data, output_path, sample_rate)
                result.audio_path = str(output_path)
                result.audio_data = final_audio

            result.success = True
            result.processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, "合成完成！")

            logger.info(f"语音合成成功: {output_path}, 处理时间: {result.processing_time:.2f}s")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"语音合成失败: {e}")

            if progress_callback:
                progress_callback(1.0, f"合成失败: {e}")

        return result

    def _load_voice_config(self, voice_id: str) -> VoiceConfig:
        """加载音色配置"""
        try:
            return self.voice_manager.load_voice(voice_id)
        except VoiceNotFoundError:
            raise VoiceSynthesizerError(f"音色不存在: {voice_id}")
        except Exception as e:
            raise VoiceSynthesizerError(f"加载音色配置失败: {e}")

    def _process_emotion_params(self, params: SynthesisParams) -> EmotionVector:
        """处理情感参数，统一转换为情感向量"""
        try:
            if params.emotion_mode == "vector" and params.emotion_vector:
                # 直接使用情感向量
                return EmotionVector.from_list(params.emotion_vector)

            elif params.emotion_mode == "preset" and params.emotion_preset:
                # 使用情感预设
                preset = self.emotion_manager.get_preset(params.emotion_preset)
                if preset:
                    return preset.emotion_vector
                else:
                    logger.warning(f"情感预设不存在: {params.emotion_preset}，使用默认平静状态")
                    return EmotionVector(calm=1.0)

            elif params.emotion_mode == "reference" and params.emotion_reference_audio:
                # 从参考音频提取情感
                return self.emotion_manager.extract_emotion_from_audio(params.emotion_reference_audio)

            elif params.emotion_mode == "text" and params.emotion_text:
                # 从文本分析情感
                return self.emotion_manager.analyze_emotion_from_text(params.emotion_text)

            else:
                # 默认使用平静状态（speaker模式或其他情况）
                return EmotionVector(calm=1.0)

        except EmotionManagerError as e:
            logger.warning(f"情感处理失败: {e}，使用默认平静状态")
            return EmotionVector(calm=1.0)

    def _process_text(self, text: str, max_tokens_per_segment: int) -> List[str]:
        """文本预处理和分句"""
        if not text or not text.strip():
            raise TextProcessingError("文本不能为空")

        # 清理文本
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 合并多个空白字符

        # 简单分句逻辑
        # 按句号、问号、感叹号分句
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 如果单句太长，进一步分割
        segments = []
        for sentence in sentences:
            if len(sentence) <= max_tokens_per_segment * 2:  # 简化的token估算
                segments.append(sentence)
            else:
                # 按逗号、分号进一步分割
                sub_segments = re.split(r'[，；,;]+', sentence)
                current_segment = ""

                for sub_segment in sub_segments:
                    sub_segment = sub_segment.strip()
                    if not sub_segment:
                        continue

                    if len(current_segment + sub_segment) <= max_tokens_per_segment * 2:
                        current_segment += sub_segment + "，"
                    else:
                        if current_segment:
                            segments.append(current_segment.rstrip("，"))
                        current_segment = sub_segment + "，"

                if current_segment:
                    segments.append(current_segment.rstrip("，"))

        if not segments:
            segments = [text]  # 如果分句失败，使用原文本

        logger.info(f"文本分句完成: {len(text)} 字符 -> {len(segments)} 段")
        return segments

    def _get_speaker_audio(self, voice_config: VoiceConfig) -> str:
        """获取说话人参考音频路径"""
        # 这里需要根据voice_config获取对应的音频文件
        # 简化实现：假设音频文件存储在特定目录
        audio_dir = Path("voices") / voice_config.voice_id

        # 查找音频文件
        for ext in ['.wav', '.mp3', '.flac']:
            audio_path = audio_dir / f"reference{ext}"
            if audio_path.exists():
                return str(audio_path)

        # 如果没有找到，抛出异常
        raise VoiceSynthesizerError(f"找不到音色参考音频: {voice_config.voice_id}")

    def _get_emotion_audio(self, params: SynthesisParams, emotion_vector: EmotionVector) -> Optional[str]:
        """获取情感参考音频路径"""
        if params.emotion_mode == "reference" and params.emotion_reference_audio:
            if Path(params.emotion_reference_audio).exists():
                return params.emotion_reference_audio
            else:
                logger.warning(f"情感参考音频不存在: {params.emotion_reference_audio}")

        return None

    def _synthesize_segment(
        self,
        text: str,
        speaker_audio: str,
        emotion_audio: Optional[str],
        emotion_vector: EmotionVector,
        params: SynthesisParams
    ) -> IndexTTSResult:
        """合成单个文本段"""
        try:
            # 准备IndexTTS参数
            generation_kwargs = {
                "do_sample": True,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "temperature": params.temperature,
                "length_penalty": 0.0,
                "num_beams": 3,
                "repetition_penalty": 10.0,
                "max_mel_tokens": 1500,
            }

            # 执行合成
            result = self.index_tts.infer(
                text=text,
                speaker_audio=speaker_audio,
                output_path=None,  # 不直接保存，返回音频数据
                emotion_control_method=params.emotion_mode if params.emotion_mode != "preset" else "vector",
                emotion_audio=emotion_audio,
                emotion_weight=params.emotion_weight,
                emotion_vector=emotion_vector.to_list(),
                emotion_text=params.emotion_text,
                use_emotion_random=False,
                max_text_tokens_per_segment=params.max_text_tokens_per_segment,
                interval_silence=params.interval_silence,
                **generation_kwargs
            )

            return result

        except InferenceError as e:
            raise VoiceSynthesizerError(f"语音合成失败: {e}")

    def _post_process_audio(
        self,
        audio_results: List[IndexTTSResult],
        params: SynthesisParams
    ) -> Optional[Tuple[int, Any]]:
        """后处理音频：拼接、归一化、淡入淡出等"""
        if not audio_results:
            return None

        try:
            # 提取音频数据
            audio_segments = []
            sample_rate = None

            for result in audio_results:
                if result.audio_data:
                    sr, audio_data = result.audio_data
                    if sample_rate is None:
                        sample_rate = sr
                    elif sample_rate != sr:
                        # 重采样到统一采样率
                        audio_data = self.audio_processor.resample_audio(audio_data, sr, sample_rate)

                    audio_segments.append(audio_data)

            if not audio_segments or sample_rate is None:
                return None

            # 拼接音频
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                # 添加段间静音
                silence_duration = params.interval_silence / 1000.0  # 转换为秒
                final_audio = self.audio_processor.concatenate_audio(
                    audio_segments,
                    crossfade_duration=0.05,  # 50ms交叉淡化
                    sample_rate=sample_rate
                )

            # 后处理
            if params.trim_silence:
                final_audio = self.audio_processor.trim_silence(final_audio, sample_rate)

            if params.normalize_audio:
                final_audio = self.audio_processor.normalize_audio(final_audio, method="peak")

            if params.apply_fade:
                final_audio = self.audio_processor.apply_fade(
                    final_audio,
                    fade_in_duration=params.fade_duration,
                    fade_out_duration=params.fade_duration,
                    sample_rate=sample_rate
                )

            return (sample_rate, final_audio)

        except AudioProcessingError as e:
            raise VoiceSynthesizerError(f"音频后处理失败: {e}")

    def _generate_output_path(self, params: SynthesisParams) -> Path:
        """生成输出文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthesis_{timestamp}_{params.voice_id[:8]}.wav"
        return self.output_dir / filename

    def list_available_voices(self) -> List[VoiceConfig]:
        """列出可用的音色"""
        return self.voice_manager.list_voices()

    def get_voice_info(self, voice_id: str) -> Optional[VoiceConfig]:
        """获取音色信息"""
        try:
            return self.voice_manager.load_voice(voice_id)
        except VoiceNotFoundError:
            return None

    def list_emotion_presets(self) -> List[str]:
        """列出可用的情感预设"""
        presets = self.emotion_manager.list_presets()
        return [preset.name for preset in presets]

    def get_emotion_preset_info(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """获取情感预设信息"""
        preset = self.emotion_manager.get_preset(preset_name)
        if preset:
            return {
                "name": preset.name,
                "description": preset.description,
                "emotion_vector": preset.emotion_vector.to_list(),
                "tags": preset.tags
            }
        return None

    def validate_params(self, params: SynthesisParams) -> List[str]:
        """验证合成参数"""
        errors = []

        # 验证文本
        if not params.text or not params.text.strip():
            errors.append("文本不能为空")

        # 验证音色
        try:
            self.voice_manager.load_voice(params.voice_id)
        except VoiceNotFoundError:
            errors.append(f"音色不存在: {params.voice_id}")

        # 验证情感参数
        if params.emotion_mode == "vector" and params.emotion_vector:
            if len(params.emotion_vector) != 8:
                errors.append("情感向量必须包含8个值")
            if any(v < 0 or v > 1 for v in params.emotion_vector):
                errors.append("情感向量值必须在0-1之间")

        elif params.emotion_mode == "reference" and params.emotion_reference_audio:
            if not Path(params.emotion_reference_audio).exists():
                errors.append(f"情感参考音频不存在: {params.emotion_reference_audio}")

        elif params.emotion_mode == "preset" and params.emotion_preset:
            if not self.emotion_manager.get_preset(params.emotion_preset):
                errors.append(f"情感预设不存在: {params.emotion_preset}")

        # 验证数值参数
        if not 0.1 <= params.speed <= 3.0:
            errors.append("语速必须在0.1-3.0之间")

        if not 0.1 <= params.temperature <= 2.0:
            errors.append("温度参数必须在0.1-2.0之间")

        return errors


# 便捷函数
def synthesize_speech(
    text: str,
    voice_id: str,
    emotion_mode: str = "speaker",
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> SynthesisResult:
    """便捷的语音合成函数"""
    synthesizer = VoiceSynthesizer()

    params = SynthesisParams(
        text=text,
        voice_id=voice_id,
        emotion_mode=emotion_mode,
        **kwargs
    )

    return synthesizer.synthesize(params, output_path)
