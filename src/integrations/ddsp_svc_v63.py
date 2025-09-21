"""DDSP-SVC 6.3版本适配器

这个模块实现DDSP-SVC 6.3版本的推理接口适配。
6.3版本特点：
1. Volume_Extractor需要hop_size和win_size两个参数
2. 音量掩码直接使用upsample处理
3. 模型返回mel，需要单独调用vocoder推理
4. 支持声域偏移功能
5. 默认t_start为0.0
"""

import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, Callable, Any
import logging
from dataclasses import dataclass

from ..utils.version_detector import VersionInfo, DDSPSVCVersion

logger = logging.getLogger(__name__)


@dataclass
class DDSPSVCv63Result:
    """DDSP-SVC 6.3推理结果"""
    audio: np.ndarray
    sample_rate: int
    processing_time: float
    segments_count: int


class DDSPSVCv63Error(Exception):
    """DDSP-SVC 6.3异常基类"""
    pass


class DDSPSVCv63Adapter:
    """DDSP-SVC 6.3版本适配器

    适配6.3版本的推理接口，处理与6.1版本的差异。
    """

    def __init__(self, ddsp_svc_path: Path, version_info: Optional[VersionInfo] = None, device: Optional[str] = None):
        """初始化6.3版本适配器

        Args:
            ddsp_svc_path: DDSP-SVC项目路径
            version_info: 版本信息
            device: 计算设备
        """
        self.ddsp_svc_path = Path(ddsp_svc_path)
        self.version_info = version_info
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 添加DDSP-SVC到Python路径
        if str(self.ddsp_svc_path) not in sys.path:
            sys.path.insert(0, str(self.ddsp_svc_path))

        # 模型缓存
        self._model: Optional[Any] = None
        self._vocoder: Optional[Any] = None
        self._args: Optional[Any] = None
        self._current_model_path: Optional[str] = None

        # 编码器缓存
        self._units_encoder: Optional[Any] = None
        self._encoder_config: Optional[Tuple] = None

        logger.info(f"DDSP-SVC 6.3适配器初始化完成，设备: {self.device}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """加载DDSP-SVC 6.3模型

        Args:
            model_path: 模型文件路径

        Raises:
            DDSPSVCv63Error: 模型加载失败
        """
        model_path = str(model_path)

        # 检查缓存
        if model_path == self._current_model_path and self._model is not None:
            logger.debug(f"模型已加载: {model_path}")
            return

        try:
            logger.info(f"加载DDSP-SVC 6.3模型: {model_path}")

            # 检查文件存在
            if not os.path.exists(model_path):
                raise DDSPSVCv63Error(f"模型文件不存在: {model_path}")

            # 导入6.3版本的模块
            from reflow.vocoder import load_model_vocoder

            # 加载模型
            model, vocoder, args = load_model_vocoder(model_path, device=self.device)

            # 缓存模型
            self._model = model
            self._vocoder = vocoder
            self._args = args
            self._current_model_path = model_path

            # 重置编码器缓存
            self._units_encoder = None
            self._encoder_config = None

            logger.info(f"6.3版本模型加载成功: {model_path}")

        except Exception as e:
            raise DDSPSVCv63Error(f"加载6.3版本模型失败: {model_path}, 错误: {e}")

    def _get_units_encoder(self) -> Any:
        """获取单元编码器（延迟加载）"""
        if self._model is None or self._args is None:
            raise DDSPSVCv63Error("模型未加载")

        # 检查是否需要重新创建编码器
        current_config = (
            self._args.data.encoder,
            self._args.data.encoder_ckpt,
            self._args.data.encoder_sample_rate,
            self._args.data.encoder_hop_size
        )

        if self._units_encoder is None or self._encoder_config != current_config:
            logger.debug("创建6.3版本单元编码器")

            from ddsp.vocoder import Units_Encoder

            # 获取cnhubertsoft_gate参数
            if self._args.data.encoder == 'cnhubertsoftfish':
                cnhubertsoft_gate = self._args.data.cnhubertsoft_gate
            else:
                cnhubertsoft_gate = 10

            self._units_encoder = Units_Encoder(
                self._args.data.encoder,
                self._args.data.encoder_ckpt,
                self._args.data.encoder_sample_rate,
                self._args.data.encoder_hop_size,
                cnhubertsoft_gate=cnhubertsoft_gate,
                device=self.device
            )
            self._encoder_config = current_config

        return self._units_encoder

    def infer(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
        speaker_id: int = 1,
        spk_mix_dict: Optional[Dict[str, float]] = None,
        f0_predictor: str = "rmvpe",
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
        threshold: float = -60.0,
        key_shift: float = 0.0,
        formant_shift: float = 0.0,
        vocal_register_shift: float = 0.0,  # 6.3版本新增
        infer_step: Optional[int] = None,
        method: str = "auto",
        t_start: float = 0.0,  # 6.3版本默认值
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DDSPSVCv63Result:
        """执行DDSP-SVC 6.3推理

        Args:
            audio: 输入音频
            sample_rate: 音频采样率
            speaker_id: 说话人ID
            spk_mix_dict: 说话人混合字典
            f0_predictor: F0预测器类型
            f0_min: 最小F0
            f0_max: 最大F0
            threshold: 响应阈值(dB)
            key_shift: 音调偏移（半音）
            formant_shift: 共振峰偏移（半音）
            vocal_register_shift: 声域偏移（半音，6.3版本新增）
            infer_step: 推理步数
            method: 采样方法
            t_start: 起始时间（6.3版本默认0.0）
            progress_callback: 进度回调函数

        Returns:
            DDSPSVCv63Result: 推理结果

        Raises:
            DDSPSVCv63Error: 推理失败
        """
        if self._model is None or self._args is None or self._vocoder is None:
            raise DDSPSVCv63Error("模型未加载，请先调用load_model()")

        try:
            import time
            from ddsp.vocoder import F0_Extractor, Volume_Extractor
            from ddsp.core import upsample
            from slicer import Slicer

            start_time = time.time()

            # 加载音频
            if isinstance(audio, (str, Path)):
                audio_data, input_sr = librosa.load(audio, sr=None)
                if len(audio_data.shape) > 1:
                    audio_data = librosa.to_mono(audio_data)
            else:
                audio_data = audio.copy()
                input_sr = sample_rate
                if input_sr is None:
                    raise DDSPSVCv63Error("当audio为numpy数组时，必须提供sample_rate")

            # 计算参数
            hop_size = self._args.data.block_size * input_sr / self._args.data.sampling_rate
            win_size = self._args.data.volume_smooth_size * input_sr / self._args.data.sampling_rate

            # 提取F0
            logger.debug(f"提取F0，预测器: {f0_predictor}")
            pitch_extractor = F0_Extractor(
                f0_predictor,
                input_sr,
                hop_size,
                f0_min,
                f0_max
            )
            f0 = pitch_extractor.extract(audio_data, uv_interp=True, device=self.device)
            f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)

            # 音调偏移
            f0 = f0 * 2 ** (key_shift / 12)

            # 共振峰偏移
            formant_shift_tensor = torch.from_numpy(
                np.array([[formant_shift]])
            ).float().to(self.device)

            # 声域偏移（6.3版本新增功能）
            try:
                if hasattr(self._vocoder, 'vocoder') and hasattr(self._vocoder.vocoder, 'h') and hasattr(self._vocoder.vocoder.h, 'pc_aug'):
                    if self._vocoder.vocoder.h.pc_aug:
                        vocal_register_factor = 2 ** (vocal_register_shift / 12)
                    else:
                        if vocal_register_shift != 0:
                            logger.warning("当前声码器不支持声域偏移")
                        vocal_register_factor = 1.0
                else:
                    if vocal_register_shift != 0:
                        logger.warning("当前声码器不支持声域偏移")
                    vocal_register_factor = 1.0
            except AttributeError:
                if vocal_register_shift != 0:
                    logger.warning("当前声码器不支持声域偏移")
                vocal_register_factor = 1.0

            # 提取音量（6.3版本特有的处理方式）
            logger.debug("提取音量包络（6.3版本）")
            volume_extractor = Volume_Extractor(hop_size, win_size)  # 6.3版本需要两个参数
            volume = volume_extractor.extract(audio_data)

            # 6.3版本特有的掩码处理
            mask = (volume > 10 ** (threshold / 20)).astype('float')
            mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
            mask = upsample(mask, self._args.data.block_size).squeeze(-1)
            volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

            # 获取单元编码器
            units_encoder = self._get_units_encoder()

            # 说话人配置
            spk_id = torch.LongTensor(np.array([[speaker_id]])).to(self.device)

            # 推理参数
            if method == 'auto':
                method = getattr(self._args.infer, 'method', 'euler')
            if infer_step is None:
                infer_step = getattr(self._args.infer, 'infer_step', 20)
            if t_start == 0.0 and hasattr(self._args, 'model') and hasattr(self._args.model, 't_start') and self._args.model.t_start is not None:
                t_start = float(self._args.model.t_start)

            # 分段处理
            logger.debug("开始分段推理（6.3版本）")
            segments = self._split_audio(audio_data, input_sr, hop_size)

            if progress_callback:
                progress_callback(0, len(segments))

            result = np.zeros(0)
            current_length = 0

            with torch.no_grad():
                for i, segment in enumerate(segments):
                    start_frame = segment[0]
                    seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)

                    # 编码单元
                    seg_units = units_encoder.encode(seg_input, input_sr, hop_size)

                    # 提取对应的F0和音量
                    seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
                    seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]

                    # 6.3版本模型推理（返回mel）
                    seg_mel = self._model(
                        seg_units,
                        seg_f0 / vocal_register_factor,
                        seg_volume,
                        spk_id=spk_id,
                        spk_mix_dict=spk_mix_dict,
                        aug_shift=formant_shift_tensor,
                        vocoder=self._vocoder,
                        infer_step=infer_step,
                        method=method,
                        t_start=t_start
                    )

                    # 声码器合成（6.3版本分离的步骤）
                    seg_output = self._vocoder.infer(seg_mel, seg_f0)

                    # 应用音量掩码
                    mask_slice = mask[:, start_frame * self._args.data.block_size :
                                    (start_frame + seg_units.size(1)) * self._args.data.block_size]
                    seg_output *= mask_slice
                    seg_output = seg_output.squeeze().cpu().numpy()

                    # 拼接音频
                    silent_length = round(start_frame * self._args.data.block_size) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, seg_output)
                    else:
                        result = self._cross_fade(result, seg_output, current_length + silent_length)

                    current_length = current_length + silent_length + len(seg_output)


                    if progress_callback:
                        progress_callback(i + 1, len(segments))

            processing_time = time.time() - start_time

            logger.info(f"6.3版本推理完成，处理时间: {processing_time:.2f}s，分段数: {len(segments)}")

            return DDSPSVCv63Result(
                audio=result,
                sample_rate=self._args.data.sampling_rate,
                processing_time=processing_time,
                segments_count=len(segments)
            )

        except Exception as e:
            raise DDSPSVCv63Error(f"6.3版本推理失败: {e}")

    def _split_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: float,
        db_thresh: float = -40,
        min_len: int = 5000
    ) -> list:
        """分割音频"""
        from slicer import Slicer

        slicer = Slicer(
            sr=sample_rate,
            threshold=db_thresh,
            min_length=min_len
        )
        chunks = dict(slicer.slice(audio))
        result = []

        for k, v in chunks.items():
            tag = v["split_time"].split(",")
            if tag[0] != tag[1]:
                start_frame = int(int(tag[0]) // hop_size)
                end_frame = int(int(tag[1]) // hop_size)
                if end_frame > start_frame:
                    result.append((
                        start_frame,
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]
                    ))

        return result

    def _cross_fade(self, a: np.ndarray, b: np.ndarray, idx: int) -> np.ndarray:
        """交叉淡化"""
        result = np.zeros(idx + b.shape[0])
        fade_len = a.shape[0] - idx
        np.copyto(dst=result[:idx], src=a[:idx])
        k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
        result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
        np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
        return result

    def save_audio(
        self,
        result: DDSPSVCv63Result,
        output_path: Union[str, Path]
    ) -> None:
        """保存音频结果

        Args:
            result: 推理结果
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(output_path, result.audio, result.sample_rate)
        logger.info(f"6.3版本音频保存成功: {output_path}")

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        if self._args is None:
            return None

        return {
            "version": "6.3",
            "model_path": self._current_model_path,
            "sampling_rate": getattr(self._args.data, 'sampling_rate', None),
            "block_size": getattr(self._args.data, 'block_size', None),
            "encoder": getattr(self._args.data, 'encoder', None),
            "encoder_ckpt": getattr(self._args.data, 'encoder_ckpt', None),
            "device": self.device,
            "supports_vocal_register": True,
            "default_t_start": 0.0
        }

    def clear_cache(self) -> None:
        """清理缓存"""
        self._model = None
        self._vocoder = None
        self._args = None
        self._units_encoder = None
        self._encoder_config = None
        self._current_model_path = None

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("6.3版本适配器缓存已清理")
