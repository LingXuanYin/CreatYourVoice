"""模型生命周期管理模块

这个模块提供模型的完整生命周期管理，包括预加载、热切换、自动清理等功能。
设计原则：
1. 生命周期清晰 - 明确的状态转换
2. 自动化管理 - 减少手动干预
3. 性能优化 - 预加载和缓存策略
"""

import time
import threading
from typing import Dict, List, Optional, Set, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from .gpu_model_manager import get_model_manager, ModelType, ModelStatus
from ..utils.gpu_utils import GPUUtils
from ..utils.memory_monitor import get_memory_monitor

logger = logging.getLogger(__name__)


class LifecycleEvent(Enum):
    """生命周期事件"""
    PRELOAD_REQUESTED = "preload_requested"
    LOAD_STARTED = "load_started"
    LOAD_COMPLETED = "load_completed"
    LOAD_FAILED = "load_failed"
    UNLOAD_STARTED = "unload_started"
    UNLOAD_COMPLETED = "unload_completed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    AUTO_CLEANUP = "auto_cleanup"


@dataclass
class LifecycleEventData:
    """生命周期事件数据"""
    event: LifecycleEvent
    model_id: str
    model_type: ModelType
    timestamp: float
    duration: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreloadConfig:
    """预加载配置"""
    model_type: ModelType
    model_path: str
    device: str = "auto"
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)  # 预加载条件
    kwargs: Dict[str, Any] = field(default_factory=dict)  # 模型参数


@dataclass
class TaskProfile:
    """任务配置文件"""
    task_name: str
    required_models: List[PreloadConfig]
    optional_models: List[PreloadConfig] = field(default_factory=list)
    memory_limit_mb: Optional[int] = None
    auto_cleanup: bool = True


class ModelLifecycleManager:
    """模型生命周期管理器

    提供模型的完整生命周期管理功能。
    设计原则：基于任务的智能模型管理。
    """

    def __init__(self):
        """初始化生命周期管理器"""
        self.model_manager = get_model_manager()
        self.memory_monitor = get_memory_monitor()

        # 预加载配置
        self._preload_configs: Dict[str, PreloadConfig] = {}
        self._task_profiles: Dict[str, TaskProfile] = {}

        # 事件系统
        self._event_listeners: List[Callable[[LifecycleEventData], None]] = []
        self._event_history: List[LifecycleEventData] = []
        self._max_event_history = 1000

        # 预加载状态
        self._preload_queue: List[str] = []  # 预加载队列
        self._preloading: Set[str] = set()   # 正在预加载的模型
        self._preload_lock = threading.Lock()

        # 自动管理
        self._auto_management_enabled = True
        self._management_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info("模型生命周期管理器初始化完成")

    def add_event_listener(self, listener: Callable[[LifecycleEventData], None]) -> None:
        """添加事件监听器"""
        self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[LifecycleEventData], None]) -> None:
        """移除事件监听器"""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def _emit_event(self, event_data: LifecycleEventData) -> None:
        """发送事件"""
        # 记录事件历史
        self._event_history.append(event_data)
        if len(self._event_history) > self._max_event_history:
            self._event_history.pop(0)

        # 通知监听器
        for listener in self._event_listeners:
            try:
                listener(event_data)
            except Exception as e:
                logger.error(f"事件监听器执行失败: {e}")

    def register_preload_config(self, config_id: str, config: PreloadConfig) -> None:
        """注册预加载配置

        Args:
            config_id: 配置ID
            config: 预加载配置
        """
        self._preload_configs[config_id] = config
        logger.info(f"注册预加载配置: {config_id} -> {config.model_path}")

    def register_task_profile(self, profile: TaskProfile) -> None:
        """注册任务配置文件

        Args:
            profile: 任务配置文件
        """
        self._task_profiles[profile.task_name] = profile
        logger.info(f"注册任务配置: {profile.task_name}, 必需模型: {len(profile.required_models)}")

    def preload_for_task(self, task_name: str) -> Dict[str, bool]:
        """为任务预加载模型

        Args:
            task_name: 任务名称

        Returns:
            预加载结果 {model_id: success}
        """
        if task_name not in self._task_profiles:
            logger.warning(f"未知任务: {task_name}")
            return {}

        profile = self._task_profiles[task_name]
        results = {}

        # 检查内存限制
        if profile.memory_limit_mb:
            memory_status = self.memory_monitor.get_current_status()
            available_memory = self._calculate_available_memory(memory_status)
            if available_memory < profile.memory_limit_mb:
                logger.warning(f"内存不足，无法为任务 {task_name} 预加载模型")
                return {}

        # 预加载必需模型
        for config in profile.required_models:
            model_id = self._preload_model(config, required=True)
            results[model_id] = model_id is not None

        # 预加载可选模型（如果内存允许）
        for config in profile.optional_models:
            if self._check_memory_availability():
                model_id = self._preload_model(config, required=False)
                if model_id:
                    results[model_id] = True
            else:
                logger.info(f"内存不足，跳过可选模型预加载: {config.model_path}")

        logger.info(f"任务 {task_name} 预加载完成，成功: {sum(results.values())}/{len(results)}")
        return results

    def _preload_model(self, config: PreloadConfig, required: bool = True) -> Optional[str]:
        """预加载单个模型"""
        start_time = time.time()

        try:
            # 生成模型ID
            model_id = f"{config.model_type.value}_{Path(config.model_path).stem}_{int(time.time())}"

            # 发送预加载请求事件
            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.PRELOAD_REQUESTED,
                model_id=model_id,
                model_type=config.model_type,
                timestamp=start_time,
                metadata={"required": required, "config": config.__dict__}
            ))

            # 检查是否已加载相同模型
            existing_model = self._find_existing_model(config)
            if existing_model:
                self._emit_event(LifecycleEventData(
                    event=LifecycleEvent.CACHE_HIT,
                    model_id=existing_model,
                    model_type=config.model_type,
                    timestamp=time.time(),
                    duration=time.time() - start_time
                ))
                return existing_model

            # 发送加载开始事件
            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.LOAD_STARTED,
                model_id=model_id,
                model_type=config.model_type,
                timestamp=time.time()
            ))

            # 执行加载
            if config.model_type == ModelType.DDSP_SVC:
                actual_model_id = self.model_manager.load_ddsp_model(
                    model_path=config.model_path,
                    device=config.device,
                    model_id=model_id
                )
            elif config.model_type == ModelType.INDEX_TTS:
                actual_model_id = self.model_manager.load_index_tts_model(
                    model_dir=config.model_path,
                    device=config.device,
                    model_id=model_id,
                    **config.kwargs
                )
            else:
                raise ValueError(f"不支持的模型类型: {config.model_type}")

            # 发送加载完成事件
            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.LOAD_COMPLETED,
                model_id=actual_model_id,
                model_type=config.model_type,
                timestamp=time.time(),
                duration=time.time() - start_time
            ))

            return actual_model_id

        except Exception as e:
            # 发送加载失败事件
            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.LOAD_FAILED,
                model_id=model_id,
                model_type=config.model_type,
                timestamp=time.time(),
                duration=time.time() - start_time,
                error_message=str(e)
            ))

            if required:
                logger.error(f"必需模型加载失败: {config.model_path}, 错误: {e}")
                raise
            else:
                logger.warning(f"可选模型加载失败: {config.model_path}, 错误: {e}")
                return None

    def _find_existing_model(self, config: PreloadConfig) -> Optional[str]:
        """查找已存在的相同模型"""
        models = self.model_manager.list_models()

        for model_info in models:
            if (model_info["model_type"] == config.model_type.value and
                model_info["model_path"] == config.model_path and
                model_info["device"] == config.device and
                model_info["status"] == ModelStatus.LOADED.value):
                return model_info["model_id"]

        return None

    def _calculate_available_memory(self, memory_status: Dict[str, Any]) -> int:
        """计算可用内存(MB)"""
        system_memory = memory_status.get("system_memory", {})
        gpu_memory = memory_status.get("gpu_memory", [])

        # 系统内存
        system_available = system_memory.get("available_mb", 0)

        # GPU内存
        gpu_available = 0
        for gpu in gpu_memory:
            gpu_available += gpu.get("free_mb", 0)

        return min(system_available, gpu_available) if gpu_available > 0 else system_available

    def _check_memory_availability(self, required_mb: int = 1024) -> bool:
        """检查内存可用性"""
        memory_status = self.memory_monitor.get_current_status()
        available = self._calculate_available_memory(memory_status)
        return available >= required_mb

    def cleanup_for_task(self, task_name: str) -> Dict[str, bool]:
        """为任务清理模型

        Args:
            task_name: 任务名称

        Returns:
            清理结果 {model_id: success}
        """
        if task_name not in self._task_profiles:
            return {}

        profile = self._task_profiles[task_name]
        if not profile.auto_cleanup:
            return {}

        # 获取任务相关的模型ID
        task_model_paths = set()
        for config in profile.required_models + profile.optional_models:
            task_model_paths.add(config.model_path)

        # 查找需要清理的模型
        models_to_cleanup = []
        for model_info in self.model_manager.list_models():
            if (model_info["model_path"] in task_model_paths and
                model_info["reference_count"] == 0):
                models_to_cleanup.append(model_info["model_id"])

        # 执行清理
        results = {}
        for model_id in models_to_cleanup:
            start_time = time.time()

            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.UNLOAD_STARTED,
                model_id=model_id,
                model_type=ModelType.DDSP_SVC,  # 这里需要从模型信息获取
                timestamp=start_time
            ))

            success = self.model_manager.unload_model(model_id)
            results[model_id] = success

            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.UNLOAD_COMPLETED,
                model_id=model_id,
                model_type=ModelType.DDSP_SVC,
                timestamp=time.time(),
                duration=time.time() - start_time
            ))

        logger.info(f"任务 {task_name} 清理完成，清理模型: {len(results)}")
        return results

    def start_auto_management(self) -> bool:
        """启动自动管理

        Returns:
            是否成功启动
        """
        if self._management_thread and self._management_thread.is_alive():
            logger.warning("自动管理已在运行")
            return True

        try:
            self._stop_event.clear()
            self._auto_management_enabled = True

            self._management_thread = threading.Thread(
                target=self._auto_management_loop,
                name="ModelLifecycleManager",
                daemon=True
            )
            self._management_thread.start()

            logger.info("模型自动管理已启动")
            return True

        except Exception as e:
            logger.error(f"启动自动管理失败: {e}")
            return False

    def stop_auto_management(self) -> None:
        """停止自动管理"""
        self._auto_management_enabled = False
        self._stop_event.set()

        if self._management_thread and self._management_thread.is_alive():
            self._management_thread.join(timeout=5.0)

        logger.info("模型自动管理已停止")

    def _auto_management_loop(self) -> None:
        """自动管理循环"""
        while self._auto_management_enabled and not self._stop_event.is_set():
            try:
                # 检查内存状态
                memory_status = self.memory_monitor.get_current_status()

                # 自动清理逻辑
                self._auto_cleanup_check(memory_status)

                # 等待下次检查
                if self._stop_event.wait(30.0):  # 30秒检查间隔
                    break

            except Exception as e:
                logger.error(f"自动管理循环异常: {e}")
                time.sleep(5.0)

    def _auto_cleanup_check(self, memory_status: Dict[str, Any]) -> None:
        """自动清理检查"""
        # 检查系统内存使用率
        system_memory = memory_status.get("system_memory", {})
        system_usage = system_memory.get("usage_percent", 0)

        # 检查GPU内存使用率
        gpu_memory = memory_status.get("gpu_memory", [])
        max_gpu_usage = max((gpu.get("usage_percent", 0) for gpu in gpu_memory), default=0)

        # 如果内存使用率过高，触发自动清理
        if system_usage > 85.0 or max_gpu_usage > 85.0:
            logger.info(f"内存使用率过高，触发自动清理 (系统: {system_usage:.1f}%, GPU: {max_gpu_usage:.1f}%)")

            self._emit_event(LifecycleEventData(
                event=LifecycleEvent.AUTO_CLEANUP,
                model_id="system",
                model_type=ModelType.DDSP_SVC,
                timestamp=time.time(),
                metadata={
                    "system_usage": system_usage,
                    "gpu_usage": max_gpu_usage
                }
            ))

            # 执行清理
            self.model_manager.optimize_memory()

    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """获取生命周期统计信息"""
        # 统计事件
        event_counts = {}
        for event_data in self._event_history:
            event_type = event_data.event.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # 计算平均加载时间
        load_times = [
            event_data.duration for event_data in self._event_history
            if event_data.event == LifecycleEvent.LOAD_COMPLETED and event_data.duration
        ]
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0

        return {
            "total_events": len(self._event_history),
            "event_counts": event_counts,
            "average_load_time": avg_load_time,
            "registered_configs": len(self._preload_configs),
            "registered_tasks": len(self._task_profiles),
            "auto_management_enabled": self._auto_management_enabled
        }

    def get_recent_events(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近的事件"""
        cutoff_time = time.time() - (minutes * 60)
        recent_events = []

        for event_data in self._event_history:
            if event_data.timestamp >= cutoff_time:
                recent_events.append({
                    "event": event_data.event.value,
                    "model_id": event_data.model_id,
                    "model_type": event_data.model_type.value,
                    "timestamp": event_data.timestamp,
                    "duration": event_data.duration,
                    "error_message": event_data.error_message,
                    "metadata": event_data.metadata
                })

        return recent_events


# 全局生命周期管理器实例
_global_lifecycle_manager: Optional[ModelLifecycleManager] = None


def get_lifecycle_manager() -> ModelLifecycleManager:
    """获取全局生命周期管理器实例"""
    global _global_lifecycle_manager
    if _global_lifecycle_manager is None:
        _global_lifecycle_manager = ModelLifecycleManager()
    return _global_lifecycle_manager


def preload_for_task(task_name: str) -> Dict[str, bool]:
    """便捷函数：为任务预加载模型"""
    manager = get_lifecycle_manager()
    return manager.preload_for_task(task_name)


def cleanup_for_task(task_name: str) -> Dict[str, bool]:
    """便捷函数：为任务清理模型"""
    manager = get_lifecycle_manager()
    return manager.cleanup_for_task(task_name)
