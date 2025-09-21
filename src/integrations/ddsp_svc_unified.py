"""DDSP-SVC统一接口

这个模块提供版本无关的统一DDSP-SVC推理接口。
设计原则：
1. 统一API - 对外提供一致的接口，隐藏版本差异
2. 自动适配 - 根据检测到的版本自动选择合适的适配器
3. 向后兼容 - 保持与现有代码的兼容性
4. 简化使用 - 用户无需关心版本差异
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Union, Tuple, Callable, Any, List
from dataclasses import dataclass

from ..utils.version_detector import DDSPSVCVersion, VersionInfo, get_ddsp_svc_version
from .version_manager import get_version_manager, DDSPSVCVersionManager
from .ddsp_svc_v61 import DDSPSVCv61Adapter, DDSPSVCv61Result
from .ddsp_svc_v63 import DDSPSVCv63Adapter, DDSPSVCv63Result

logger = logging.getLogger(__name__)


@dataclass
class DDSPSVCUnifiedResult:
    """DDSP-SVC统一推理结果"""
    audio: Any  # np.ndarray
    sample_rate: int
    processing_time: float
    segments_count: int
    version: str
    adapter_info: Dict[str, Any]


class DDSPSVCUnifiedError(Exception):
    """DDSP-SVC统一接口异常"""
    pass


class DDSPSVCUnified:
    """DDSP-SVC统一接口

    提供版本无关的统一推理接口，自动处理6.1和6.3版本的差异。
    """

    def __init__(
        self,
        version: Union[str, DDSPSVCVersion] = "auto",
        ddsp_svc_path: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """初始化DDSP-SVC统一接口

        Args:
            version: 指定版本，"auto"表示自动检测
            ddsp_svc_path: DDSP-SVC项目路径
            device: 计算设备
        """
        self.device = device
        self.ddsp_svc_path = ddsp_svc_path

        # 版本管理器
        self.version_manager = get_version_manager()

        # 当前适配器
        self._current_adapter: Optional[Any] = None
        self._current_version: Optional[DDSPSVCVersion] = None

        # 设置版本
        if version == "auto":
            self._auto_detect_version()
        else:
            self._set_version(version)

        logger.info(f"DDSP-SVC统一接口初始化完成，版本: {self._current_version.value if self._current_version else 'unknown'}")

    def _auto_detect_version(self) -> None:
        """自动检测版本"""
        try:
            version_info = self.version_manager.detect_and_set_version()
            self._current_version = version_info.version

            if self._current_version == DDSPSVCVersion.UNKNOWN:
                logger.warning("无法检测版本，使用默认版本6.3")
                self._current_version = DDSPSVCVersion.V6_3

            self._current_adapter = self.version_manager.get_adapter(self._current_version)

        except Exception as e:
            logger.error(f"版本检测失败: {e}")
            # 回退到6.3版本
            self._current_version = DDSPSVCVersion.V6_3
            self._current_adapter = self.version_manager.get_adapter(self._current_version)

    def _set_version(self, version: Union[str, DDSPSVCVersion]) -> None:
        """设置指定版本"""
        if isinstance(version, str):
            if version == "6.1":
                version = DDSPSVCVersion.V6_1
            elif version == "6.3":
                version = DDSPSVCVersion.V6_3
            else:
                raise DDSPSVCUnifiedError(f"不支持的版本: {version}")

        self._current_version = version

        # 尝试切换版本
        if not self.version_manager.switch_version(version):
            logger.warning(f"版本切换失败，继续使用版本: {version.value}")

        self._current_adapter = self.version_manager.get_adapter(version)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """加载DDSP-SVC模型

        Args:
            model_path: 模型文件路径

        Raises:
            DDSPSVCUnifiedError: 模型加载失败
        """
        if self._current_adapter is None:
            raise DDSPSVCUnifiedError("适配器未初始化")

        try:
            self._current_adapter.load_model(model_path)
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            raise DDSPSVCUnifiedError(f"模型加载失败: {e}")

    def infer(
        self,
        audio: Union[Any, str, Path],  # np.ndarray or file path
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
        t_start: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DDSPSVCUnifiedResult:
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
            vocal_register_shift: 声域偏移（半音，仅6.3版本支持）
            infer_step: 推理步数
            method: 采样方法
            t_start: 起始时间（None表示使用版本默认值）
            progress_callback: 进度回调函数

        Returns:
            DDSPSVCUnifiedResult: 统一推理结果

        Raises:
            DDSPSVCUnifiedError: 推理失败
        """
        if self._current_adapter is None:
            raise DDSPSVCUnifiedError("适配器未初始化")

        try:
            # 处理版本特定的参数
            kwargs = {
                "audio": audio,
                "sample_rate": sample_rate,
                "speaker_id": speaker_id,
                "spk_mix_dict": spk_mix_dict,
                "f0_predictor": f0_predictor,
                "f0_min": f0_min,
                "f0_max": f0_max,
                "threshold": threshold,
                "key_shift": key_shift,
                "formant_shift": formant_shift,
                "infer_step": infer_step,
                "method": method,
                "progress_callback": progress_callback
            }

            # 处理t_start参数
            if t_start is None:
                # 使用版本默认值
                if self._current_version == DDSPSVCVersion.V6_1:
                    kwargs["t_start"] = 0.7
                else:  # V6_3
                    kwargs["t_start"] = 0.0
            else:
                kwargs["t_start"] = t_start

            # 处理声域偏移（仅6.3版本支持）
            if self._current_version == DDSPSVCVersion.V6_3:
                kwargs["vocal_register_shift"] = vocal_register_shift
            elif vocal_register_shift != 0.0:
                version_str = self._current_version.value if self._current_version else "unknown"
                logger.warning(f"版本{version_str}不支持声域偏移，忽略该参数")

            # 执行推理
            result = self._current_adapter.infer(**kwargs)

            # 转换为统一结果格式
            version_str = self._current_version.value if self._current_version else "unknown"
            return DDSPSVCUnifiedResult(
                audio=result.audio,
                sample_rate=result.sample_rate,
                processing_time=result.processing_time,
                segments_count=result.segments_count,
                version=version_str,
                adapter_info=self._current_adapter.get_model_info() or {}
            )

        except Exception as e:
            raise DDSPSVCUnifiedError(f"推理失败: {e}")

    def save_audio(
        self,
        result: DDSPSVCUnifiedResult,
        output_path: Union[str, Path]
    ) -> None:
        """保存音频结果

        Args:
            result: 推理结果
            output_path: 输出路径
        """
        if self._current_adapter is None:
            raise DDSPSVCUnifiedError("适配器未初始化")

        # 转换为适配器特定的结果格式
        if self._current_version == DDSPSVCVersion.V6_1:
            adapter_result = DDSPSVCv61Result(
                audio=result.audio,
                sample_rate=result.sample_rate,
                processing_time=result.processing_time,
                segments_count=result.segments_count
            )
        else:  # V6_3
            adapter_result = DDSPSVCv63Result(
                audio=result.audio,
                sample_rate=result.sample_rate,
                processing_time=result.processing_time,
                segments_count=result.segments_count
            )

        self._current_adapter.save_audio(adapter_result, output_path)

    def get_version_info(self) -> Dict[str, Any]:
        """获取版本信息

        Returns:
            Dict[str, Any]: 版本信息
        """
        version_info = self.version_manager.get_current_version()
        model_info = self._current_adapter.get_model_info() if self._current_adapter else {}

        return {
            "current_version": self._current_version.value if self._current_version else "unknown",
            "version_info": {
                "version": version_info.version.value if version_info else "unknown",
                "branch": version_info.branch if version_info else None,
                "commit_hash": version_info.commit_hash if version_info else None,
                "path": str(version_info.path) if version_info else None,
                "features": version_info.features if version_info else {}
            },
            "model_info": model_info,
            "supported_versions": [v.value for v in self.version_manager.get_supported_versions()],
            "device": self.device
        }

    def switch_version(self, version: Union[str, DDSPSVCVersion]) -> bool:
        """切换版本

        Args:
            version: 目标版本

        Returns:
            bool: 切换是否成功
        """
        try:
            self._set_version(version)
            version_str = self._current_version.value if self._current_version else "unknown"
            logger.info(f"成功切换到版本: {version_str}")
            return True
        except Exception as e:
            logger.error(f"版本切换失败: {e}")
            return False

    def get_supported_features(self) -> Dict[str, bool]:
        """获取当前版本支持的功能

        Returns:
            Dict[str, bool]: 功能支持情况
        """
        if self._current_version == DDSPSVCVersion.V6_1:
            return {
                "vocal_register_shift": False,
                "win_size_volume": False,
                "return_wav_direct": True,
                "mask_padding": True
            }
        elif self._current_version == DDSPSVCVersion.V6_3:
            return {
                "vocal_register_shift": True,
                "win_size_volume": True,
                "return_wav_direct": False,
                "mask_padding": False
            }
        else:
            return {}

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载

        Returns:
            bool: 模型是否已加载
        """
        if self._current_adapter is None:
            return False

        # 检查适配器是否有is_model_loaded方法
        if hasattr(self._current_adapter, 'is_model_loaded'):
            return self._current_adapter.is_model_loaded()

        # 回退检查：检查是否有模型信息
        model_info = self._current_adapter.get_model_info()
        return model_info is not None and model_info.get('model_path') is not None

    def get_available_speakers(self) -> List[Dict[str, Any]]:
        """获取可用的说话人列表

        Returns:
            List[Dict[str, Any]]: 说话人信息列表
        """
        if self._current_adapter is None:
            return []

        # 检查适配器是否有get_available_speakers方法
        if hasattr(self._current_adapter, 'get_available_speakers'):
            result = self._current_adapter.get_available_speakers()
            # 确保返回列表格式
            if isinstance(result, dict):
                return [{"id": k, "name": v} for k, v in result.items()]
            elif isinstance(result, list):
                return result
            else:
                return []

        # 回退方案：从模型信息中获取
        model_info = self._current_adapter.get_model_info()
        if model_info and 'speakers' in model_info:
            speakers = model_info['speakers']
            if isinstance(speakers, dict):
                return [{"id": k, "name": v} for k, v in speakers.items()]
            elif isinstance(speakers, list):
                return speakers

        # 默认返回空列表
        return []

    def unload_model(self) -> None:
        """卸载模型"""
        if self._current_adapter is None:
            return

        # 检查适配器是否有unload_model方法
        if hasattr(self._current_adapter, 'unload_model'):
            self._current_adapter.unload_model()
        else:
            # 回退方案：清理缓存
            self._current_adapter.clear_cache()

        logger.info("模型已卸载")

    def clear_cache(self) -> None:
        """清理缓存"""
        if self._current_adapter:
            self._current_adapter.clear_cache()

        self.version_manager.clear_cache()

        logger.info("统一接口缓存已清理")

    def __del__(self):
        """析构函数"""
        try:
            self.clear_cache()
        except Exception as e:
            # 析构函数中不应该抛出异常
            logger.debug(f"析构函数清理缓存时出错: {e}")


# 便捷函数
def convert_voice(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_path: Union[str, Path],
    version: Union[str, DDSPSVCVersion] = "auto",
    speaker_id: int = 1,
    spk_mix_dict: Optional[Dict[str, float]] = None,
    **kwargs
) -> DDSPSVCUnifiedResult:
    """便捷的音色转换函数

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        model_path: 模型路径
        version: DDSP-SVC版本
        speaker_id: 说话人ID
        spk_mix_dict: 说话人混合字典
        **kwargs: 其他推理参数

    Returns:
        DDSPSVCUnifiedResult: 推理结果
    """
    unified = DDSPSVCUnified(version=version)
    unified.load_model(model_path)

    result = unified.infer(
        audio=input_path,
        speaker_id=speaker_id,
        spk_mix_dict=spk_mix_dict,
        **kwargs
    )

    unified.save_audio(result, output_path)
    return result


def get_ddsp_svc_unified(version: Union[str, DDSPSVCVersion] = "auto") -> DDSPSVCUnified:
    """获取DDSP-SVC统一接口实例

    Args:
        version: 指定版本

    Returns:
        DDSPSVCUnified: 统一接口实例
    """
    return DDSPSVCUnified(version=version)
