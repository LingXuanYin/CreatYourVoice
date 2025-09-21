"""DDSP-SVC集成模块（多版本支持）

这个模块封装DDSP-SVC的推理接口，提供简洁的音色转换功能。
现在支持DDSP-SVC 6.1和6.3两个版本的自动检测和适配。

设计原则：
1. 隐藏复杂性 - 用户只需要提供音频和配置
2. 统一接口 - 无论单说话人还是多说话人混合都用同一接口
3. 资源管理 - 自动管理模型加载和GPU内存
4. 版本无关 - 自动检测和适配不同版本的DDSP-SVC
5. 向后兼容 - 保持与现有代码的兼容性

版本支持：
- DDSP-SVC 6.1: 基础功能，不支持声域偏移
- DDSP-SVC 6.3: 完整功能，支持声域偏移和改进的音量处理
"""

import os
import sys
import torch
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, Callable, Any, List
import logging
from dataclasses import dataclass

# 导入版本管理相关模块
try:
    from .ddsp_svc_unified import DDSPSVCUnified, DDSPSVCUnifiedResult, convert_voice as unified_convert_voice
    from ..utils.version_detector import DDSPSVCVersion, get_ddsp_svc_version
    from .version_manager import get_version_manager
except ImportError as e:
    logging.error(f"无法导入版本管理模块: {e}")
    # 回退到原有实现
    DDSPSVCUnified = None

logger = logging.getLogger(__name__)

# 添加DDSP-SVC到Python路径（向后兼容）
DDSP_SVC_PATH = Path(__file__).parent.parent.parent / "DDSP-SVC"
if str(DDSP_SVC_PATH) not in sys.path:
    sys.path.insert(0, str(DDSP_SVC_PATH))


@dataclass
class DDSPSVCResult:
    """DDSP-SVC推理结果（向后兼容）"""
    audio: np.ndarray
    sample_rate: int
    processing_time: float
    segments_count: int
    version: Optional[str] = None  # 新增版本信息
    adapter_info: Optional[Dict[str, Any]] = None  # 新增适配器信息


class DDSPSVCError(Exception):
    """DDSP-SVC异常基类"""
    pass


class ModelLoadError(DDSPSVCError):
    """模型加载异常"""
    pass


class InferenceError(DDSPSVCError):
    """推理异常"""
    pass


class DDSPSVCIntegration:
    """DDSP-SVC集成类（多版本支持）

    提供DDSP-SVC模型的加载和推理功能，支持6.1和6.3版本自动检测和适配。
    设计原则：
    1. 延迟加载 - 只有在需要时才加载模型
    2. 缓存机制 - 避免重复加载相同模型
    3. 错误恢复 - 推理失败时提供有意义的错误信息
    4. 版本无关 - 自动适配不同版本的DDSP-SVC
    5. 向后兼容 - 保持与现有代码的兼容性
    """

    def __init__(self, device: Optional[str] = None, version: Union[str, DDSPSVCVersion] = "auto"):
        """初始化DDSP-SVC集成

        Args:
            device: 计算设备，None表示自动选择
            version: DDSP-SVC版本，"auto"表示自动检测
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 尝试使用新的统一接口
        if DDSPSVCUnified is not None:
            try:
                self._unified = DDSPSVCUnified(version=version, device=self.device)
                self._use_unified = True
                logger.info(f"DDSP-SVC集成初始化完成（统一接口），设备: {self.device}")
                return
            except Exception as e:
                logger.warning(f"统一接口初始化失败，回退到原有实现: {e}")

        # 回退到原有实现
        self._use_unified = False
        self._init_legacy()

    def _init_legacy(self):
        """初始化原有实现（向后兼容）"""
        try:
            from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
            from ddsp.core import upsample
            from reflow.vocoder import load_model_vocoder
            from slicer import Slicer
        except ImportError as e:
            logger.error(f"无法导入DDSP-SVC模块: {e}")
            raise

        # 模型缓存
        self._model_cache: Dict[str, Tuple] = {}
        self._current_model_path: Optional[str] = None
        self._model: Optional[Any] = None
        self._vocoder: Optional[Any] = None
        self._args: Optional[Any] = None

        # 编码器缓存
        self._units_encoder: Optional[Any] = None
        self._encoder_config: Optional[Tuple] = None

        logger.info(f"DDSP-SVC集成初始化完成（原有实现），设备: {self.device}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """加载DDSP-SVC模型

        Args:
            model_path: 模型文件路径

        Raises:
            ModelLoadError: 模型加载失败
        """
        if self._use_unified:
            # 使用统一接口
            try:
                self._unified.load_model(model_path)
                logger.info(f"模型加载成功（统一接口）: {model_path}")
            except Exception as e:
                raise ModelLoadError(f"加载模型失败: {model_path}, 错误: {e}")
        else:
            # 使用原有实现
            self._load_model_legacy(model_path)

    def _load_model_legacy(self, model_path: Union[str, Path]) -> None:
        """加载DDSP-SVC模型（原有实现）"""
        from reflow.vocoder import load_model_vocoder

        model_path = str(model_path)

        # 检查缓存
        if model_path == self._current_model_path and self._model is not None:
            logger.debug(f"模型已加载: {model_path}")
            return

        try:
            logger.info(f"加载DDSP-SVC模型: {model_path}")

            # 检查文件存在
            if not os.path.exists(model_path):
                raise ModelLoadError(f"模型文件不存在: {model_path}")

            # 加载模型
            model, vocoder, args = load_model_vocoder(model_path, device=self.device)

            # 缓存模型
            self._model = model
            self._vocoder = vocoder
            self._args = args
            self._current_model_path = model_path

            # 重置编码器缓存（因为可能需要不同的编码器）
            self._units_encoder = None
            self._encoder_config = None

            logger.info(f"模型加载成功: {model_path}")

        except Exception as e:
            raise ModelLoadError(f"加载模型失败: {model_path}, 错误: {e}")

    def _get_units_encoder(self) -> Any:
        """获取单元编码器（延迟加载）"""
        from ddsp.vocoder import Units_Encoder

        if self._model is None or self._args is None:
            raise InferenceError("模型未加载")

        # 检查是否需要重新创建编码器
        current_config = (
            self._args.data.encoder,
            self._args.data.encoder_ckpt,
            self._args.data.encoder_sample_rate,
            self._args.data.encoder_hop_size
        )

        if self._units_encoder is None or self._encoder_config != current_config:
            logger.debug("创建单元编码器")

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
        vocal_register_shift: float = 0.0,
        infer_step: Optional[int] = None,
        method: str = "auto",
        t_start: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DDSPSVCResult:
        """执行DDSP-SVC推理

        Args:
            audio: 输入音频（numpy数组或文件路径）
            sample_rate: 音频采样率（如果audio是numpy数组）
            speaker_id: 说话人ID
            spk_mix_dict: 说话人混合字典
            f0_predictor: F0预测器类型
            f0_min: 最小F0
            f0_max: 最大F0
            threshold: 响应阈值(dB)
            key_shift: 音调偏移（半音）
            formant_shift: 共振峰偏移（半音）
            vocal_register_shift: 声域偏移（半音）
            infer_step: 推理步数
            method: 采样方法
            t_start: 起始时间
            progress_callback: 进度回调函数

        Returns:
            DDSPSVCResult: 推理结果

        Raises:
            InferenceError: 推理失败
        """
        if self._model is None or self._args is None or self._vocoder is None:
            raise InferenceError("模型未加载，请先调用load_model()")

        try:
            import time
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
                    raise InferenceError("当audio为numpy数组时，必须提供sample_rate")

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

            # 声域偏移
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

            # 提取音量
            logger.debug("提取音量包络")
            volume_extractor = Volume_Extractor(hop_size, win_size)
            volume = volume_extractor.extract(audio_data)
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
            logger.debug("开始分段推理")
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

                    # 模型推理
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

                    # 声码器合成
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

            logger.info(f"推理完成，处理时间: {processing_time:.2f}s，分段数: {len(segments)}")

            return DDSPSVCResult(
                audio=result,
                sample_rate=self._args.data.sampling_rate,
                processing_time=processing_time,
                segments_count=len(segments)
            )

        except Exception as e:
            raise InferenceError(f"推理失败: {e}")

    def _split_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        hop_size: float,
        db_thresh: float = -40,
        min_len: int = 5000
    ) -> list:
        """分割音频"""
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
        result: DDSPSVCResult,
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
        logger.info(f"音频保存成功: {output_path}")

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取当前模型信息"""
        if self._use_unified:
            # 使用统一接口获取模型信息
            if hasattr(self, '_unified') and self._unified:
                return self._unified.get_model_info()
            return None

        if self._args is None:
            return None

        return {
            "model_path": self._current_model_path,
            "sampling_rate": getattr(self._args.data, 'sampling_rate', None),
            "block_size": getattr(self._args.data, 'block_size', None),
            "encoder": getattr(self._args.data, 'encoder', None),
            "encoder_ckpt": getattr(self._args.data, 'encoder_ckpt', None),
            "device": self.device,
            "speakers": self.get_available_speakers()
        }

    def get_available_speakers(self) -> List[Dict[str, Any]]:
        """获取模型中可用的speaker列表

        Returns:
            List[Dict[str, Any]]: speaker信息列表，每个包含id和name
        """
        if self._use_unified:
            # 使用统一接口获取speaker列表
            if hasattr(self, '_unified') and self._unified:
                return self._unified.get_available_speakers()
            return []

        if self._model is None or self._args is None:
            return []

        try:
            speakers = []

            # 尝试从模型配置中获取speaker信息
            if hasattr(self._args, 'model') and hasattr(self._args.model, 'n_spk'):
                n_speakers = self._args.model.n_spk
                for i in range(n_speakers):
                    speakers.append({
                        "id": i,
                        "name": f"Speaker_{i}"
                    })

            # 尝试从模型状态字典中获取更详细的speaker信息
            if hasattr(self._model, 'state_dict'):
                state_dict = self._model.state_dict()
                # 查找speaker相关的参数
                for key in state_dict.keys():
                    if 'spk' in key.lower() and 'embed' in key.lower():
                        # 从embedding层推断speaker数量
                        if len(state_dict[key].shape) > 0:
                            n_speakers = state_dict[key].shape[0]
                            speakers = []
                            for i in range(n_speakers):
                                speakers.append({
                                    "id": i,
                                    "name": f"Speaker_{i}"
                                })
                            break

            logger.info(f"检测到 {len(speakers)} 个speaker")
            return speakers

        except Exception as e:
            logger.error(f"获取speaker列表失败: {e}")
            return []

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        if self._use_unified:
            if hasattr(self, '_unified') and self._unified:
                return self._unified.is_model_loaded()
            return False

        return self._model is not None and self._args is not None

    def unload_model(self) -> None:
        """卸载当前模型"""
        if self._use_unified:
            if hasattr(self, '_unified') and self._unified:
                self._unified.unload_model()
            return

        self.clear_cache()
        logger.info("DDSP-SVC模型已卸载")

    def clear_cache(self) -> None:
        """清理缓存"""
        self._model_cache.clear()
        self._model = None
        self._vocoder = None
        self._args = None
        self._units_encoder = None
        self._encoder_config = None
        self._current_model_path = None

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("缓存已清理")

    def __del__(self):
        """析构函数"""
        try:
            if self._use_unified:
                # 使用统一接口的清理
                if hasattr(self, '_unified') and self._unified:
                    self._unified.clear_cache()
            else:
                # 使用原有实现的清理
                if hasattr(self, '_model_cache'):
                    self.clear_cache()
        except Exception as e:
            # 析构函数中不应该抛出异常
            logger.debug(f"析构函数清理缓存时出错: {e}")


# 便捷函数
def convert_voice(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_path: Union[str, Path],
    speaker_id: int = 1,
    spk_mix_dict: Optional[Dict[str, float]] = None,
    **kwargs
) -> DDSPSVCResult:
    """便捷的音色转换函数

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        model_path: 模型路径
        speaker_id: 说话人ID
        spk_mix_dict: 说话人混合字典
        **kwargs: 其他推理参数

    Returns:
        DDSPSVCResult: 推理结果
    """
    integration = DDSPSVCIntegration()
    integration.load_model(model_path)

    result = integration.infer(
        audio=input_path,
        speaker_id=speaker_id,
        spk_mix_dict=spk_mix_dict,
        **kwargs
    )

    integration.save_audio(result, output_path)
    return result
