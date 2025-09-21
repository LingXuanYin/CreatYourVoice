"""GPU模型管理器

这个模块提供统一的GPU模型加载、卸载和管理功能。
设计原则：
1. 统一接口 - 所有模型都通过同一个管理器
2. 引用计数 - 自动管理模型生命周期
3. 内存优化 - 智能加载和卸载策略
"""

import os
import gc
import time
import threading
from typing import Dict, Optional, Any, Set, Callable, Union, List
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

from ..utils.gpu_utils import GPUUtils
from ..utils.memory_monitor import get_memory_monitor, MemoryAlert
from ..utils.error_handler import handle_errors, ErrorSeverity, get_error_handler
from ..integrations.ddsp_svc import DDSPSVCIntegration
from ..integrations.index_tts import IndexTTSIntegration

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型枚举"""
    DDSP_SVC = "ddsp_svc"
    INDEX_TTS = "index_tts"


class ModelStatus(Enum):
    """模型状态枚举"""
    UNLOADED = "unloaded"      # 未加载
    LOADING = "loading"        # 加载中
    LOADED = "loaded"          # 已加载
    UNLOADING = "unloading"    # 卸载中
    ERROR = "error"            # 错误状态


@dataclass
class ModelInfo:
    """模型信息"""
    model_id: str
    model_type: ModelType
    model_path: str
    device: str
    status: ModelStatus = ModelStatus.UNLOADED
    reference_count: int = 0
    memory_usage_mb: int = 0
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    error_message: Optional[str] = None

    # 模型实例
    instance: Optional[Any] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.last_used is None:
            self.last_used = time.time()


@dataclass
class LoadRequest:
    """加载请求"""
    model_id: str
    model_type: ModelType
    model_path: str
    device: str = "auto"
    priority: int = 0  # 优先级，数字越大优先级越高
    callback: Optional[Callable[[bool, str], None]] = None


class GPUModelManager:
    """GPU模型管理器

    统一管理所有GPU模型的加载、卸载和生命周期。
    设计原则：
    1. 单例模式 - 全局唯一的模型管理器
    2. 线程安全 - 支持多线程并发访问
    3. 自动清理 - 基于引用计数和内存压力自动卸载
    """

    _instance: Optional['GPUModelManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'GPUModelManager':
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化模型管理器"""
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return

        # 模型存储
        self._models: Dict[str, ModelInfo] = {}
        self._model_lock = threading.RLock()

        # 加载队列
        self._load_queue: List[LoadRequest] = []
        self._queue_lock = threading.Lock()

        # 配置参数
        self.max_memory_usage_percent = 85.0  # 最大内存使用率
        self.auto_cleanup_enabled = True
        self.cleanup_threshold_percent = 90.0  # 自动清理阈值
        self.idle_timeout_minutes = 30  # 空闲超时时间

        # 监控和回调
        self._memory_monitor = get_memory_monitor()
        self._memory_monitor.add_alert_callback(self._on_memory_alert)

        # 状态
        self._initialized = True

        logger.info("GPU模型管理器初始化完成")

    def load_ddsp_model(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        model_id: Optional[str] = None
    ) -> str:
        """加载DDSP-SVC模型

        Args:
            model_path: 模型文件路径
            device: 计算设备
            model_id: 模型ID，None则自动生成

        Returns:
            模型ID

        Raises:
            RuntimeError: 加载失败
        """
        model_path = str(model_path)
        if model_id is None:
            model_id = f"ddsp_{Path(model_path).stem}_{int(time.time())}"

        return self._load_model(
            model_id=model_id,
            model_type=ModelType.DDSP_SVC,
            model_path=model_path,
            device=device
        )

    def load_index_tts_model(
        self,
        model_dir: Union[str, Path],
        device: str = "auto",
        model_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """加载IndexTTS模型

        Args:
            model_dir: 模型目录路径
            device: 计算设备
            model_id: 模型ID，None则自动生成
            **kwargs: IndexTTS初始化参数

        Returns:
            模型ID

        Raises:
            RuntimeError: 加载失败
        """
        model_dir = str(model_dir)
        if model_id is None:
            model_id = f"index_tts_{Path(model_dir).name}_{int(time.time())}"

        return self._load_model(
            model_id=model_id,
            model_type=ModelType.INDEX_TTS,
            model_path=model_dir,
            device=device,
            **kwargs
        )

    def _load_model(
        self,
        model_id: str,
        model_type: ModelType,
        model_path: str,
        device: str = "auto",
        **kwargs
    ) -> str:
        """内部模型加载方法"""
        with self._model_lock:
            # 检查模型是否已存在
            if model_id in self._models:
                model_info = self._models[model_id]
                if model_info.status == ModelStatus.LOADED:
                    model_info.reference_count += 1
                    model_info.last_used = time.time()
                    logger.info(f"模型已加载，增加引用计数: {model_id} (refs: {model_info.reference_count})")
                    return model_id
                elif model_info.status == ModelStatus.LOADING:
                    # 等待加载完成
                    self._wait_for_loading(model_id)
                    return model_id

            # 检查设备和内存
            if device == "auto":
                device = GPUUtils.get_optimal_device()

            # 估算内存需求
            estimated_memory = GPUUtils.estimate_model_memory(model_path)
            if not GPUUtils.check_memory_requirement(estimated_memory, device):
                # 尝试清理内存
                self._cleanup_unused_models()
                if not GPUUtils.check_memory_requirement(estimated_memory, device):
                    raise RuntimeError(f"内存不足，无法加载模型 {model_id}")

            # 创建模型信息
            model_info = ModelInfo(
                model_id=model_id,
                model_type=model_type,
                model_path=model_path,
                device=device,
                status=ModelStatus.LOADING,
                reference_count=1
            )
            self._models[model_id] = model_info

            try:
                # 执行加载
                start_time = time.time()
                instance = self._create_model_instance(model_type, model_path, device, **kwargs)
                load_time = time.time() - start_time

                # 更新模型信息
                model_info.instance = instance
                model_info.status = ModelStatus.LOADED
                model_info.load_time = load_time
                model_info.memory_usage_mb = self._estimate_instance_memory(instance, device)

                logger.info(f"模型加载成功: {model_id} (耗时: {load_time:.2f}s, 内存: {model_info.memory_usage_mb}MB)")
                return model_id

            except Exception as e:
                # 加载失败，更新状态
                model_info.status = ModelStatus.ERROR
                model_info.error_message = str(e)
                logger.error(f"模型加载失败: {model_id}, 错误: {e}")
                raise RuntimeError(f"加载模型失败: {e}")

    def _create_model_instance(
        self,
        model_type: ModelType,
        model_path: str,
        device: str,
        **kwargs
    ) -> Any:
        """创建模型实例"""
        if model_type == ModelType.DDSP_SVC:
            integration = DDSPSVCIntegration(device=device)
            integration.load_model(model_path)
            return integration

        elif model_type == ModelType.INDEX_TTS:
            integration = IndexTTSIntegration(
                model_dir=model_path,
                device=device,
                **kwargs
            )
            integration.load_model()
            return integration

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _estimate_instance_memory(self, instance: Any, device: str) -> int:
        """估算实例内存使用"""
        if device.startswith("cuda"):
            try:
                device_id = int(device.split(":")[1]) if ":" in device else 0
                memory_info = GPUUtils.get_gpu_memory_info(device_id)
                return memory_info.get("used", 0)
            except Exception:
                pass

        # 默认估算值
        return 1024  # 1GB

    def _wait_for_loading(self, model_id: str, timeout: float = 60.0) -> None:
        """等待模型加载完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            model_info = self._models.get(model_id)
            if not model_info:
                raise RuntimeError(f"模型不存在: {model_id}")

            if model_info.status == ModelStatus.LOADED:
                return
            elif model_info.status == ModelStatus.ERROR:
                raise RuntimeError(f"模型加载失败: {model_info.error_message}")

            time.sleep(0.1)

        raise RuntimeError(f"模型加载超时: {model_id}")

    def unload_model(self, model_id: str, force: bool = False) -> bool:
        """卸载模型

        Args:
            model_id: 模型ID
            force: 是否强制卸载（忽略引用计数）

        Returns:
            是否成功卸载
        """
        with self._model_lock:
            if model_id not in self._models:
                logger.warning(f"模型不存在: {model_id}")
                return False

            model_info = self._models[model_id]

            if not force:
                # 减少引用计数
                model_info.reference_count = max(0, model_info.reference_count - 1)
                if model_info.reference_count > 0:
                    logger.info(f"减少模型引用计数: {model_id} (refs: {model_info.reference_count})")
                    return True

            # 执行卸载
            return self._unload_model_instance(model_id)

    def _unload_model_instance(self, model_id: str) -> bool:
        """卸载模型实例"""
        model_info = self._models[model_id]

        try:
            model_info.status = ModelStatus.UNLOADING

            # 清理模型实例
            if model_info.instance:
                if hasattr(model_info.instance, 'clear_cache'):
                    model_info.instance.clear_cache()
                model_info.instance = None

            # 清理GPU缓存
            if model_info.device.startswith("cuda"):
                device_id = int(model_info.device.split(":")[1]) if ":" in model_info.device else 0
                GPUUtils.clear_gpu_cache(device_id)

            # 强制垃圾回收
            gc.collect()

            # 更新状态
            model_info.status = ModelStatus.UNLOADED
            model_info.memory_usage_mb = 0

            logger.info(f"模型卸载成功: {model_id}")
            return True

        except Exception as e:
            model_info.status = ModelStatus.ERROR
            model_info.error_message = f"卸载失败: {e}"
            logger.error(f"模型卸载失败: {model_id}, 错误: {e}")
            return False

    def unload_all_models(self, force: bool = False) -> Dict[str, bool]:
        """卸载所有模型

        Args:
            force: 是否强制卸载

        Returns:
            卸载结果字典 {model_id: success}
        """
        results = {}

        with self._model_lock:
            model_ids = list(self._models.keys())

        for model_id in model_ids:
            results[model_id] = self.unload_model(model_id, force=force)

        logger.info(f"批量卸载完成，成功: {sum(results.values())}, 失败: {len(results) - sum(results.values())}")
        return results

    def get_model(self, model_id: str) -> Optional[Any]:
        """获取模型实例

        Args:
            model_id: 模型ID

        Returns:
            模型实例，如果不存在或未加载则返回None
        """
        with self._model_lock:
            if model_id not in self._models:
                return None

            model_info = self._models[model_id]
            if model_info.status != ModelStatus.LOADED:
                return None

            # 更新最后使用时间
            model_info.last_used = time.time()
            return model_info.instance

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        with self._model_lock:
            if model_id not in self._models:
                return None

            model_info = self._models[model_id]
            return {
                "model_id": model_info.model_id,
                "model_type": model_info.model_type.value,
                "model_path": model_info.model_path,
                "device": model_info.device,
                "status": model_info.status.value,
                "reference_count": model_info.reference_count,
                "memory_usage_mb": model_info.memory_usage_mb,
                "load_time": model_info.load_time,
                "last_used": model_info.last_used,
                "error_message": model_info.error_message
            }

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        with self._model_lock:
            result = []
            for model_id in self._models.keys():
                info = self.get_model_info(model_id)
                if info is not None:
                    result.append(info)
            return result

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        with self._model_lock:
            total_memory = 0
            loaded_models = 0

            for model_info in self._models.values():
                if model_info.status == ModelStatus.LOADED:
                    total_memory += model_info.memory_usage_mb
                    loaded_models += 1

            # 获取系统内存状态
            memory_status = self._memory_monitor.get_current_status()

            return {
                "total_model_memory_mb": total_memory,
                "loaded_model_count": loaded_models,
                "total_model_count": len(self._models),
                "system_memory": memory_status.get("system_memory", {}),
                "gpu_memory": memory_status.get("gpu_memory", [])
            }

    def _cleanup_unused_models(self) -> int:
        """清理未使用的模型

        Returns:
            清理的模型数量
        """
        cleaned_count = 0
        current_time = time.time()

        with self._model_lock:
            models_to_cleanup = []

            for model_id, model_info in self._models.items():
                # 检查是否可以清理
                if (model_info.status == ModelStatus.LOADED and
                    model_info.reference_count == 0 and
                    model_info.last_used is not None and
                    current_time - model_info.last_used > self.idle_timeout_minutes * 60):
                    models_to_cleanup.append(model_id)

            # 执行清理
            for model_id in models_to_cleanup:
                if self._unload_model_instance(model_id):
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"自动清理了 {cleaned_count} 个未使用的模型")

        return cleaned_count

    def _on_memory_alert(self, alert: MemoryAlert) -> None:
        """内存警告回调"""
        if alert.alert_type in ["warning", "critical"] and self.auto_cleanup_enabled:
            logger.warning(f"收到内存警告，开始自动清理: {alert.message}")
            cleaned = self._cleanup_unused_models()
            if cleaned == 0:
                logger.warning("自动清理未释放任何模型，考虑手动卸载模型")

    def enable_auto_cleanup(self, enabled: bool = True) -> None:
        """启用/禁用自动清理"""
        self.auto_cleanup_enabled = enabled
        logger.info(f"自动清理已{'启用' if enabled else '禁用'}")

    def set_cleanup_threshold(self, threshold_percent: float) -> None:
        """设置自动清理阈值"""
        self.cleanup_threshold_percent = max(50.0, min(95.0, threshold_percent))
        logger.info(f"自动清理阈值设置为: {self.cleanup_threshold_percent}%")

    def set_idle_timeout(self, timeout_minutes: int) -> None:
        """设置空闲超时时间"""
        self.idle_timeout_minutes = max(1, timeout_minutes)
        logger.info(f"空闲超时时间设置为: {self.idle_timeout_minutes} 分钟")

    def optimize_memory(self) -> Dict[str, Any]:
        """优化内存使用

        Returns:
            优化结果
        """
        start_time = time.time()

        # 清理未使用的模型
        cleaned_models = self._cleanup_unused_models()

        # 清理GPU缓存
        if GPUUtils.is_cuda_available():
            GPUUtils.clear_gpu_cache()

        # 强制垃圾回收
        gc.collect()

        optimization_time = time.time() - start_time

        result = {
            "cleaned_models": cleaned_models,
            "optimization_time": optimization_time,
            "memory_status": self.get_memory_usage()
        }

        logger.info(f"内存优化完成，清理模型: {cleaned_models}, 耗时: {optimization_time:.2f}s")
        return result


# 全局模型管理器实例
_global_manager: Optional[GPUModelManager] = None


def get_model_manager() -> GPUModelManager:
    """获取全局模型管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = GPUModelManager()
    return _global_manager


def load_ddsp_model(model_path: Union[str, Path], device: str = "auto", model_id: Optional[str] = None) -> str:
    """便捷函数：加载DDSP-SVC模型"""
    manager = get_model_manager()
    return manager.load_ddsp_model(model_path, device, model_id)


def load_index_tts_model(model_dir: Union[str, Path], device: str = "auto", model_id: Optional[str] = None, **kwargs) -> str:
    """便捷函数：加载IndexTTS模型"""
    manager = get_model_manager()
    return manager.load_index_tts_model(model_dir, device, model_id, **kwargs)


def unload_model(model_id: str, force: bool = False) -> bool:
    """便捷函数：卸载模型"""
    manager = get_model_manager()
    return manager.unload_model(model_id, force)


def get_model(model_id: str) -> Optional[Any]:
    """便捷函数：获取模型实例"""
    manager = get_model_manager()
    return manager.get_model(model_id)


def optimize_memory() -> Dict[str, Any]:
    """便捷函数：优化内存"""
    manager = get_model_manager()
    return manager.optimize_memory()
