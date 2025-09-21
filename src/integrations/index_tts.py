"""IndexTTS集成模块

这个模块封装IndexTTS v2的推理接口，提供情感语音合成功能。
设计原则：
1. 简化接口 - 隐藏复杂的情感控制逻辑
2. 缓存优化 - 复用参考音频处理结果
3. 错误恢复 - 提供有意义的错误信息和降级方案
"""

import os
import sys
import time
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple, Callable, Any
import logging
from dataclasses import dataclass

# 添加IndexTTS到Python路径
INDEX_TTS_PATH = Path(__file__).parent.parent.parent / "index-tts"
if str(INDEX_TTS_PATH) not in sys.path:
    sys.path.insert(0, str(INDEX_TTS_PATH))

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError as e:
    logging.error(f"无法导入IndexTTS模块: {e}")
    raise

logger = logging.getLogger(__name__)


@dataclass
class IndexTTSResult:
    """IndexTTS推理结果"""
    audio_path: Optional[str]
    audio_data: Optional[Tuple[int, np.ndarray]]  # (sample_rate, audio_array)
    processing_time: float
    segments_count: int
    emotion_info: Optional[Dict[str, Any]] = None


class IndexTTSError(Exception):
    """IndexTTS异常基类"""
    pass


class ModelLoadError(IndexTTSError):
    """模型加载异常"""
    pass


class InferenceError(IndexTTSError):
    """推理异常"""
    pass


class IndexTTSIntegration:
    """IndexTTS集成类

    提供IndexTTS v2模型的加载和推理功能。
    设计原则：
    1. 延迟加载 - 只有在需要时才加载模型
    2. 情感控制 - 支持多种情感控制方式
    3. 缓存机制 - 避免重复处理相同的参考音频
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "checkpoints",
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        use_cuda_kernel: bool = False,
        use_deepspeed: bool = False
    ):
        """初始化IndexTTS集成

        Args:
            model_dir: 模型目录
            config_path: 配置文件路径
            device: 计算设备
            use_fp16: 是否使用FP16
            use_cuda_kernel: 是否使用CUDA内核
            use_deepspeed: 是否使用DeepSpeed
        """
        self.model_dir = Path(model_dir)
        self.config_path = config_path or (self.model_dir / "config.yaml")
        self.device = device
        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed

        # 模型实例
        self._model: Optional[Any] = None
        self._model_loaded = False

        logger.info(f"IndexTTS集成初始化完成，模型目录: {self.model_dir}")

    def load_model(self) -> None:
        """加载IndexTTS模型

        Raises:
            ModelLoadError: 模型加载失败
        """
        if self._model_loaded and self._model is not None:
            logger.debug("模型已加载")
            return

        try:
            logger.info(f"加载IndexTTS模型: {self.model_dir}")

            # 检查必要文件
            required_files = [
                "bpe.model",
                "gpt.pth",
                "config.yaml",
                "s2mel.pth",
                "wav2vec2bert_stats.pt"
            ]

            for file in required_files:
                file_path = self.model_dir / file
                if not file_path.exists():
                    raise ModelLoadError(f"必需文件不存在: {file_path}")

            # 创建模型实例
            self._model = IndexTTS2(
                cfg_path=str(self.config_path),
                model_dir=str(self.model_dir),
                use_fp16=self.use_fp16,
                device=self.device,
                use_cuda_kernel=self.use_cuda_kernel,
                use_deepspeed=self.use_deepspeed
            )

            self._model_loaded = True
            logger.info(f"IndexTTS模型加载成功")

        except Exception as e:
            raise ModelLoadError(f"加载IndexTTS模型失败: {e}")

    def infer(
        self,
        text: str,
        speaker_audio: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        emotion_control_method: str = "speaker",
        emotion_audio: Optional[Union[str, Path]] = None,
        emotion_weight: float = 0.65,
        emotion_vector: Optional[List[float]] = None,
        emotion_text: Optional[str] = None,
        use_emotion_random: bool = False,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **generation_kwargs
    ) -> IndexTTSResult:
        """执行IndexTTS推理

        Args:
            text: 目标文本
            speaker_audio: 说话人参考音频路径
            output_path: 输出音频路径（None则返回音频数据）
            emotion_control_method: 情感控制方式
                - "speaker": 使用说话人音频的情感
                - "reference": 使用情感参考音频
                - "vector": 使用情感向量
                - "text": 使用情感描述文本
            emotion_audio: 情感参考音频路径
            emotion_weight: 情感权重 (0.0-1.0)
            emotion_vector: 情感向量 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emotion_text: 情感描述文本
            use_emotion_random: 是否使用情感随机采样
            max_text_tokens_per_segment: 每段最大token数
            interval_silence: 段间静音时长(ms)
            progress_callback: 进度回调函数
            **generation_kwargs: 生成参数

        Returns:
            IndexTTSResult: 推理结果

        Raises:
            InferenceError: 推理失败
        """
        if not self._model_loaded or self._model is None:
            self.load_model()

        try:
            start_time = time.time()

            # 设置进度回调
            if progress_callback:
                def gr_progress(value, desc):
                    progress_callback(value, desc)
                if hasattr(self._model, 'gr_progress'):
                    self._model.gr_progress = gr_progress

            # 验证输入
            if not text or not text.strip():
                raise InferenceError("文本不能为空")

            if not os.path.exists(speaker_audio):
                raise InferenceError(f"说话人音频文件不存在: {speaker_audio}")

            # 处理情感控制参数
            emo_control_mode = self._parse_emotion_control_method(emotion_control_method)

            # 验证情感控制参数
            if emo_control_mode == 1 and (not emotion_audio or not os.path.exists(emotion_audio)):
                logger.warning("情感参考音频不存在，回退到说话人情感模式")
                emo_control_mode = 0
                emotion_audio = None

            if emo_control_mode == 2 and not emotion_vector:
                logger.warning("情感向量为空，回退到说话人情感模式")
                emo_control_mode = 0
                emotion_vector = None

            # 处理情感向量
            if emotion_vector and emo_control_mode == 2:
                # 确保向量长度为8
                if len(emotion_vector) != 8:
                    raise InferenceError(f"情感向量长度必须为8，当前为{len(emotion_vector)}")

                # 归一化情感向量
                if hasattr(self._model, 'normalize_emo_vec'):
                    emotion_vector = self._model.normalize_emo_vec(emotion_vector, apply_bias=True)

            # 设置生成参数默认值
            generation_params = {
                "do_sample": True,
                "top_p": 0.8,
                "top_k": 30,
                "temperature": 0.8,
                "length_penalty": 0.0,
                "num_beams": 3,
                "repetition_penalty": 10.0,
                "max_mel_tokens": 1500,
            }
            generation_params.update(generation_kwargs)

            logger.info(f"开始IndexTTS推理，文本长度: {len(text)}, 情感控制: {emotion_control_method}")

            # 执行推理
            if not hasattr(self._model, 'infer'):
                raise InferenceError("模型不支持推理方法")

            result = self._model.infer(
                spk_audio_prompt=str(speaker_audio),
                text=text,
                output_path=str(output_path) if output_path else None,
                emo_audio_prompt=str(emotion_audio) if emotion_audio else None,
                emo_alpha=emotion_weight,
                emo_vector=emotion_vector,
                use_emo_text=(emo_control_mode == 3),
                emo_text=emotion_text,
                use_random=use_emotion_random,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                verbose=False,
                **generation_params
            )

            processing_time = time.time() - start_time

            # 计算分段数（简化估算）
            segments_count = 1
            if hasattr(self._model, 'tokenizer'):
                try:
                    text_tokens = self._model.tokenizer.tokenize(text)
                    segments = self._model.tokenizer.split_segments(text_tokens, max_text_tokens_per_segment)
                    segments_count = len(segments)
                except Exception:
                    # 如果分段失败，使用默认值
                    segments_count = max(1, len(text) // (max_text_tokens_per_segment * 2))

            # 构建情感信息
            emotion_info = {
                "control_method": emotion_control_method,
                "emotion_weight": emotion_weight,
                "emotion_vector": emotion_vector,
                "emotion_text": emotion_text,
                "use_random": use_emotion_random
            }

            logger.info(f"IndexTTS推理完成，处理时间: {processing_time:.2f}s，分段数: {segments_count}")

            if output_path:
                return IndexTTSResult(
                    audio_path=str(output_path),
                    audio_data=None,
                    processing_time=processing_time,
                    segments_count=segments_count,
                    emotion_info=emotion_info
                )
            else:
                return IndexTTSResult(
                    audio_path=None,
                    audio_data=result,  # (sample_rate, audio_array)
                    processing_time=processing_time,
                    segments_count=segments_count,
                    emotion_info=emotion_info
                )

        except Exception as e:
            raise InferenceError(f"IndexTTS推理失败: {e}")

    def _parse_emotion_control_method(self, method: str) -> int:
        """解析情感控制方式"""
        method_map = {
            "speaker": 0,      # 与音色参考音频相同
            "reference": 1,    # 使用情感参考音频
            "vector": 2,       # 使用情感向量控制
            "text": 3          # 使用情感描述文本控制
        }

        if method not in method_map:
            logger.warning(f"未知的情感控制方式: {method}，使用默认值 'speaker'")
            return 0

        return method_map[method]

    def create_emotion_vector(
        self,
        happy: float = 0.0,
        angry: float = 0.0,
        sad: float = 0.0,
        afraid: float = 0.0,
        disgusted: float = 0.0,
        melancholic: float = 0.0,
        surprised: float = 0.0,
        calm: float = 0.0
    ) -> List[float]:
        """创建情感向量

        Args:
            happy: 高兴 (0.0-1.0)
            angry: 愤怒 (0.0-1.0)
            sad: 悲伤 (0.0-1.0)
            afraid: 恐惧 (0.0-1.0)
            disgusted: 厌恶 (0.0-1.0)
            melancholic: 低落 (0.0-1.0)
            surprised: 惊讶 (0.0-1.0)
            calm: 平静 (0.0-1.0)

        Returns:
            情感向量列表
        """
        return [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

    def analyze_emotion_from_text(self, text: str) -> Dict[str, float]:
        """从文本分析情感

        Args:
            text: 输入文本

        Returns:
            情感分析结果字典

        Raises:
            InferenceError: 分析失败
        """
        if not self._model_loaded or self._model is None:
            self.load_model()

        try:
            # 使用QwenEmotion分析情感
            if hasattr(self._model, 'qwen_emo') and hasattr(self._model.qwen_emo, 'inference'):
                emotion_dict = self._model.qwen_emo.inference(text)
                logger.info(f"文本情感分析结果: {emotion_dict}")
                return emotion_dict
            else:
                raise InferenceError("模型不支持情感分析功能")

        except Exception as e:
            raise InferenceError(f"文本情感分析失败: {e}")

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        if not self._model_loaded or self._model is None:
            return None

        return {
            "model_dir": str(self.model_dir),
            "config_path": str(self.config_path),
            "device": getattr(self._model, 'device', None),
            "use_fp16": getattr(self._model, 'use_fp16', None),
            "use_cuda_kernel": getattr(self._model, 'use_cuda_kernel', None),
            "model_version": getattr(self._model, 'model_version', None),
            "max_text_tokens": getattr(getattr(self._model, 'cfg', None), 'gpt', {}).get('max_text_tokens', None) if hasattr(self._model, 'cfg') else None,
            "max_mel_tokens": getattr(getattr(self._model, 'cfg', None), 'gpt', {}).get('max_mel_tokens', None) if hasattr(self._model, 'cfg') else None
        }

    def clear_cache(self) -> None:
        """清理缓存"""
        if self._model is not None:
            # 清理IndexTTS内部缓存
            self._model.cache_spk_cond = None
            self._model.cache_s2mel_style = None
            self._model.cache_s2mel_prompt = None
            self._model.cache_spk_audio_prompt = None
            self._model.cache_emo_cond = None
            self._model.cache_emo_audio_prompt = None
            self._model.cache_mel = None

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("IndexTTS缓存已清理")

    def preload_speaker(self, speaker_audio: Union[str, Path]) -> None:
        """预加载说话人音频

        Args:
            speaker_audio: 说话人音频路径
        """
        if not self._model_loaded or self._model is None:
            self.load_model()

        try:
            # 通过一次简单推理来预加载说话人特征
            self._model.infer(
                spk_audio_prompt=str(speaker_audio),
                text="测试",  # 简短文本
                output_path=None,
                max_text_tokens_per_segment=20,
                max_mel_tokens=50  # 限制生成长度
            )
            logger.info(f"说话人音频预加载完成: {speaker_audio}")

        except Exception as e:
            logger.warning(f"说话人音频预加载失败: {e}")

    def __del__(self):
        """析构函数"""
        self.clear_cache()


# 便捷函数
def synthesize_speech(
    text: str,
    speaker_audio: Union[str, Path],
    output_path: Union[str, Path],
    model_dir: Union[str, Path] = "checkpoints",
    emotion_control: str = "speaker",
    **kwargs
) -> IndexTTSResult:
    """便捷的语音合成函数

    Args:
        text: 目标文本
        speaker_audio: 说话人音频路径
        output_path: 输出音频路径
        model_dir: 模型目录
        emotion_control: 情感控制方式
        **kwargs: 其他推理参数

    Returns:
        IndexTTSResult: 推理结果
    """
    integration = IndexTTSIntegration(model_dir=model_dir)
    integration.load_model()

    result = integration.infer(
        text=text,
        speaker_audio=speaker_audio,
        output_path=output_path,
        emotion_control_method=emotion_control,
        **kwargs
    )

    return result
