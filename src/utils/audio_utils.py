"""音频处理工具

这个模块提供音频处理的基础功能。
设计原则：
1. 单一职责 - 每个函数只做一件事
2. 无副作用 - 不修改输入数据
3. 简单直接 - 避免过度抽象
"""

import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """音频处理异常"""
    pass


class AudioProcessor:
    """音频处理器

    提供常用的音频处理功能。
    设计原则：
    1. 静态方法优先 - 大部分操作不需要状态
    2. 标准化接口 - 统一的输入输出格式
    3. 错误恢复 - 提供有意义的错误信息
    """

    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        sample_rate: Optional[int] = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """加载音频文件

        Args:
            file_path: 音频文件路径
            sample_rate: 目标采样率，None表示保持原始采样率
            mono: 是否转换为单声道
            offset: 开始时间偏移（秒）
            duration: 加载时长（秒），None表示加载全部

        Returns:
            (audio_data, sample_rate): 音频数据和采样率

        Raises:
            AudioProcessingError: 加载失败
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AudioProcessingError(f"音频文件不存在: {file_path}")

            audio, sr = librosa.load(
                file_path,
                sr=sample_rate,
                mono=mono,
                offset=offset,
                duration=duration
            )

            logger.debug(f"音频加载成功: {file_path}, 形状: {audio.shape}, 采样率: {sr}")
            return audio, sr

        except Exception as e:
            raise AudioProcessingError(f"加载音频文件失败: {file_path}, 错误: {e}")

    @staticmethod
    def save_audio(
        audio: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: int,
        format: Optional[str] = None
    ) -> None:
        """保存音频文件

        Args:
            audio: 音频数据
            file_path: 输出文件路径
            sample_rate: 采样率
            format: 音频格式，None表示根据文件扩展名自动判断

        Raises:
            AudioProcessingError: 保存失败
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 确保音频数据在有效范围内
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = np.clip(audio, -1.0, 1.0)

            sf.write(file_path, audio, sample_rate, format=format)
            logger.debug(f"音频保存成功: {file_path}")

        except Exception as e:
            raise AudioProcessingError(f"保存音频文件失败: {file_path}, 错误: {e}")

    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """重采样音频

        Args:
            audio: 原始音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率

        Returns:
            重采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio.copy()

        try:
            resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            logger.debug(f"音频重采样: {orig_sr}Hz -> {target_sr}Hz")
            return resampled

        except Exception as e:
            raise AudioProcessingError(f"音频重采样失败: {e}")

    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        method: str = "peak",
        target_level: float = 0.95
    ) -> np.ndarray:
        """音频归一化

        Args:
            audio: 音频数据
            method: 归一化方法 ("peak", "rms")
            target_level: 目标电平

        Returns:
            归一化后的音频数据
        """
        if len(audio) == 0:
            return audio.copy()

        try:
            if method == "peak":
                # 峰值归一化
                peak = np.max(np.abs(audio))
                if peak > 0:
                    normalized = audio * (target_level / peak)
                else:
                    normalized = audio.copy()

            elif method == "rms":
                # RMS归一化
                rms = np.sqrt(np.mean(audio ** 2))
                if rms > 0:
                    normalized = audio * (target_level / rms)
                else:
                    normalized = audio.copy()
            else:
                raise AudioProcessingError(f"未知的归一化方法: {method}")

            # 防止削波
            normalized = np.clip(normalized, -1.0, 1.0)
            return normalized

        except Exception as e:
            raise AudioProcessingError(f"音频归一化失败: {e}")

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sample_rate: int,
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """去除音频首尾静音

        Args:
            audio: 音频数据
            sample_rate: 采样率
            threshold_db: 静音阈值（dB）
            frame_length: 帧长度
            hop_length: 跳跃长度

        Returns:
            去除静音后的音频数据
        """
        try:
            # 使用librosa的trim函数
            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=-threshold_db,
                frame_length=frame_length,
                hop_length=hop_length
            )

            logger.debug(f"静音去除: {len(audio)} -> {len(trimmed)} 样本")
            return trimmed

        except Exception as e:
            raise AudioProcessingError(f"去除静音失败: {e}")

    @staticmethod
    def split_audio(
        audio: np.ndarray,
        sample_rate: int,
        segment_length: float,
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """分割音频

        Args:
            audio: 音频数据
            sample_rate: 采样率
            segment_length: 分段长度（秒）
            overlap: 重叠长度（秒）

        Returns:
            音频分段列表
        """
        try:
            segment_samples = int(segment_length * sample_rate)
            overlap_samples = int(overlap * sample_rate)
            step_samples = segment_samples - overlap_samples

            if step_samples <= 0:
                raise AudioProcessingError("重叠长度不能大于等于分段长度")

            segments = []
            start = 0

            while start < len(audio):
                end = min(start + segment_samples, len(audio))
                segment = audio[start:end]

                # 如果分段太短，跳过
                if len(segment) >= segment_samples // 2:
                    segments.append(segment)

                start += step_samples

            logger.debug(f"音频分割: {len(audio)} 样本 -> {len(segments)} 段")
            return segments

        except Exception as e:
            raise AudioProcessingError(f"音频分割失败: {e}")

    @staticmethod
    def concatenate_audio(
        audio_list: List[np.ndarray],
        crossfade_duration: float = 0.0,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """拼接音频

        Args:
            audio_list: 音频数据列表
            crossfade_duration: 交叉淡化时长（秒）
            sample_rate: 采样率（用于计算交叉淡化样本数）

        Returns:
            拼接后的音频数据
        """
        if not audio_list:
            return np.array([])

        if len(audio_list) == 1:
            return audio_list[0].copy()

        try:
            result = audio_list[0].copy()

            for i in range(1, len(audio_list)):
                current_audio = audio_list[i]

                if crossfade_duration > 0 and sample_rate is not None:
                    # 应用交叉淡化
                    fade_samples = int(crossfade_duration * sample_rate)
                    fade_samples = min(fade_samples, len(result), len(current_audio))

                    if fade_samples > 0:
                        # 创建淡化窗口
                        fade_out = np.linspace(1.0, 0.0, fade_samples)
                        fade_in = np.linspace(0.0, 1.0, fade_samples)

                        # 应用交叉淡化
                        result[-fade_samples:] *= fade_out
                        current_audio[:fade_samples] *= fade_in
                        result[-fade_samples:] += current_audio[:fade_samples]

                        # 拼接剩余部分
                        result = np.concatenate([result, current_audio[fade_samples:]])
                    else:
                        result = np.concatenate([result, current_audio])
                else:
                    # 直接拼接
                    result = np.concatenate([result, current_audio])

            logger.debug(f"音频拼接: {len(audio_list)} 段 -> {len(result)} 样本")
            return result

        except Exception as e:
            raise AudioProcessingError(f"音频拼接失败: {e}")

    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        fade_in_duration: float = 0.0,
        fade_out_duration: float = 0.0,
        sample_rate: int = 22050
    ) -> np.ndarray:
        """应用淡入淡出效果

        Args:
            audio: 音频数据
            fade_in_duration: 淡入时长（秒）
            fade_out_duration: 淡出时长（秒）
            sample_rate: 采样率

        Returns:
            应用淡化效果后的音频数据
        """
        if len(audio) == 0:
            return audio.copy()

        try:
            result = audio.copy()

            # 淡入
            if fade_in_duration > 0:
                fade_in_samples = int(fade_in_duration * sample_rate)
                fade_in_samples = min(fade_in_samples, len(result))

                if fade_in_samples > 0:
                    fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples)
                    result[:fade_in_samples] *= fade_in_curve

            # 淡出
            if fade_out_duration > 0:
                fade_out_samples = int(fade_out_duration * sample_rate)
                fade_out_samples = min(fade_out_samples, len(result))

                if fade_out_samples > 0:
                    fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples)
                    result[-fade_out_samples:] *= fade_out_curve

            return result

        except Exception as e:
            raise AudioProcessingError(f"应用淡化效果失败: {e}")

    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> dict:
        """获取音频文件信息

        Args:
            file_path: 音频文件路径

        Returns:
            音频信息字典
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AudioProcessingError(f"音频文件不存在: {file_path}")

            # 获取基本信息
            info = sf.info(file_path)

            # 计算时长
            duration = info.frames / info.samplerate

            return {
                "file_path": str(file_path),
                "duration": duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
                "file_size": file_path.stat().st_size
            }

        except Exception as e:
            raise AudioProcessingError(f"获取音频信息失败: {file_path}, 错误: {e}")

    @staticmethod
    def convert_format(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "wav",
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None
    ) -> None:
        """转换音频格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_format: 目标格式
            target_sample_rate: 目标采样率
            target_channels: 目标声道数
        """
        try:
            # 加载音频
            audio, sr = AudioProcessor.load_audio(input_path, mono=(target_channels == 1))

            # 重采样
            if target_sample_rate and target_sample_rate != sr:
                audio = AudioProcessor.resample_audio(audio, sr, target_sample_rate)
                sr = target_sample_rate

            # 调整声道
            if target_channels:
                if target_channels == 1 and len(audio.shape) > 1:
                    # 转为单声道
                    audio = np.mean(audio, axis=0)
                elif target_channels == 2 and len(audio.shape) == 1:
                    # 转为立体声
                    audio = np.stack([audio, audio], axis=0)

            # 保存
            AudioProcessor.save_audio(audio, output_path, sr, format=target_format)
            logger.info(f"格式转换完成: {input_path} -> {output_path}")

        except Exception as e:
            raise AudioProcessingError(f"格式转换失败: {e}")


# 便捷函数
def load_audio(file_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, int]:
    """便捷的音频加载函数"""
    return AudioProcessor.load_audio(file_path, **kwargs)


def save_audio(audio: np.ndarray, file_path: Union[str, Path], sample_rate: int, **kwargs) -> None:
    """便捷的音频保存函数"""
    return AudioProcessor.save_audio(audio, file_path, sample_rate, **kwargs)


def normalize_audio(audio: np.ndarray, **kwargs) -> np.ndarray:
    """便捷的音频归一化函数"""
    return AudioProcessor.normalize_audio(audio, **kwargs)


def trim_silence(audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
    """便捷的静音去除函数"""
    return AudioProcessor.trim_silence(audio, sample_rate, **kwargs)
