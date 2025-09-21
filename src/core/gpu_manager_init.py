"""GPU模型管理器初始化模块

这个模块负责初始化和配置GPU模型管理系统。
设计原则：
1. 自动初始化 - 系统启动时自动配置
2. 配置驱动 - 基于配置文件进行初始化
3. 错误恢复 - 初始化失败时提供降级方案
"""

import logging
from typing import Optional, Dict, Any

from .gpu_model_manager import get_model_manager
from .model_lifecycle import get_lifecycle_manager, PreloadConfig, TaskProfile, ModelType
from ..utils.memory_monitor import get_memory_monitor, start_global_monitoring
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class GPUManagerInitializer:
    """GPU模型管理器初始化器"""

    def __init__(self):
        """初始化"""
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.lifecycle_manager = get_lifecycle_manager()
        self.memory_monitor = get_memory_monitor()
        self._initialized = False

    def initialize(self) -> bool:
        """初始化GPU模型管理系统

        Returns:
            是否成功初始化
        """
        if self._initialized:
            logger.info("GPU模型管理系统已初始化")
            return True

        try:
            logger.info("开始初始化GPU模型管理系统...")

            # 1. 配置模型管理器
            self._configure_model_manager()

            # 2. 启动内存监控
            self._start_memory_monitoring()

            # 3. 注册默认任务配置
            self._register_default_tasks()

            # 4. 启动生命周期管理
            self._start_lifecycle_management()

            self._initialized = True
            logger.info("GPU模型管理系统初始化完成")
            return True

        except Exception as e:
            logger.error(f"GPU模型管理系统初始化失败: {e}")
            return False

    def _configure_model_manager(self) -> None:
        """配置模型管理器"""
        gpu_config = self.config.gpu

        # 配置自动清理
        self.model_manager.enable_auto_cleanup(gpu_config.auto_cleanup_enabled)
        self.model_manager.set_cleanup_threshold(gpu_config.cleanup_threshold_percent)
        self.model_manager.set_idle_timeout(gpu_config.idle_timeout_minutes)
        self.model_manager.max_memory_usage_percent = gpu_config.max_memory_usage_percent

        logger.info(f"模型管理器配置完成: 自动清理={gpu_config.auto_cleanup_enabled}, "
                   f"清理阈值={gpu_config.cleanup_threshold_percent}%")

    def _start_memory_monitoring(self) -> None:
        """启动内存监控"""
        gpu_config = self.config.gpu

        if gpu_config.memory_monitoring_enabled:
            # 配置监控参数
            self.memory_monitor.update_interval = gpu_config.memory_monitoring_interval

            # 启动监控
            success = start_global_monitoring()
            if success:
                logger.info(f"内存监控已启动，更新间隔: {gpu_config.memory_monitoring_interval}s")
            else:
                logger.warning("内存监控启动失败")
        else:
            logger.info("内存监控已禁用")

    def _register_default_tasks(self) -> None:
        """注册默认任务配置"""
        # 语音合成任务
        voice_synthesis_task = TaskProfile(
            task_name="voice_synthesis",
            required_models=[
                PreloadConfig(
                    model_type=ModelType.INDEX_TTS,
                    model_path=self.config.index_tts.model_dir,
                    device="auto",
                    priority=1
                )
            ],
            optional_models=[
                PreloadConfig(
                    model_type=ModelType.DDSP_SVC,
                    model_path=self.config.ddsp_svc.model_dir,
                    device="auto",
                    priority=0
                )
            ],
            memory_limit_mb=4096,  # 4GB限制
            auto_cleanup=True
        )

        # 音色转换任务
        voice_conversion_task = TaskProfile(
            task_name="voice_conversion",
            required_models=[
                PreloadConfig(
                    model_type=ModelType.DDSP_SVC,
                    model_path=self.config.ddsp_svc.model_dir,
                    device="auto",
                    priority=1
                )
            ],
            memory_limit_mb=2048,  # 2GB限制
            auto_cleanup=True
        )

        # 音色创建任务
        voice_creation_task = TaskProfile(
            task_name="voice_creation",
            required_models=[
                PreloadConfig(
                    model_type=ModelType.INDEX_TTS,
                    model_path=self.config.index_tts.model_dir,
                    device="auto",
                    priority=1
                ),
                PreloadConfig(
                    model_type=ModelType.DDSP_SVC,
                    model_path=self.config.ddsp_svc.model_dir,
                    device="auto",
                    priority=1
                )
            ],
            memory_limit_mb=6144,  # 6GB限制
            auto_cleanup=False  # 创建任务不自动清理
        )

        # 注册任务配置
        self.lifecycle_manager.register_task_profile(voice_synthesis_task)
        self.lifecycle_manager.register_task_profile(voice_conversion_task)
        self.lifecycle_manager.register_task_profile(voice_creation_task)

        logger.info("默认任务配置已注册: voice_synthesis, voice_conversion, voice_creation")

    def _start_lifecycle_management(self) -> None:
        """启动生命周期管理"""
        success = self.lifecycle_manager.start_auto_management()
        if success:
            logger.info("模型生命周期自动管理已启动")
        else:
            logger.warning("模型生命周期自动管理启动失败")

    def shutdown(self) -> None:
        """关闭GPU模型管理系统"""
        if not self._initialized:
            return

        try:
            logger.info("关闭GPU模型管理系统...")

            # 停止生命周期管理
            self.lifecycle_manager.stop_auto_management()

            # 停止内存监控
            self.memory_monitor.stop_monitoring()

            # 卸载所有模型
            self.model_manager.unload_all_models(force=True)

            self._initialized = False
            logger.info("GPU模型管理系统已关闭")

        except Exception as e:
            logger.error(f"关闭GPU模型管理系统失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self._initialized,
            "model_manager_status": {
                "loaded_models": len([m for m in self.model_manager.list_models() if m["status"] == "loaded"]),
                "total_models": len(self.model_manager.list_models()),
                "auto_cleanup_enabled": self.model_manager.auto_cleanup_enabled
            },
            "memory_monitoring": {
                "enabled": self.memory_monitor._monitoring,
                "current_status": self.memory_monitor.get_current_status()
            },
            "lifecycle_management": {
                "auto_management_enabled": self.lifecycle_manager._auto_management_enabled,
                "statistics": self.lifecycle_manager.get_lifecycle_statistics()
            }
        }


# 全局初始化器实例
_global_initializer: Optional[GPUManagerInitializer] = None


def get_gpu_manager_initializer() -> GPUManagerInitializer:
    """获取全局GPU管理器初始化器实例"""
    global _global_initializer
    if _global_initializer is None:
        _global_initializer = GPUManagerInitializer()
    return _global_initializer


def initialize_gpu_management() -> bool:
    """初始化GPU模型管理系统"""
    initializer = get_gpu_manager_initializer()
    return initializer.initialize()


def shutdown_gpu_management() -> None:
    """关闭GPU模型管理系统"""
    global _global_initializer
    if _global_initializer:
        _global_initializer.shutdown()


def get_gpu_management_status() -> Dict[str, Any]:
    """获取GPU管理系统状态"""
    initializer = get_gpu_manager_initializer()
    return initializer.get_status()


# 便捷函数：为特定任务预加载模型
def preload_models_for_task(task_name: str) -> Dict[str, bool]:
    """为特定任务预加载模型"""
    from .model_lifecycle import preload_for_task
    return preload_for_task(task_name)


def cleanup_models_for_task(task_name: str) -> Dict[str, bool]:
    """为特定任务清理模型"""
    from .model_lifecycle import cleanup_for_task
    return cleanup_for_task(task_name)


# 便捷函数：智能模型管理
def auto_load_for_synthesis(text: str, speaker_audio: str) -> str:
    """为语音合成自动加载模型"""
    try:
        # 预加载IndexTTS模型
        results = preload_models_for_task("voice_synthesis")

        # 获取IndexTTS模型
        from .gpu_model_manager import get_model
        models = get_model_manager().list_models()

        for model_info in models:
            if (model_info["model_type"] == "index_tts" and
                model_info["status"] == "loaded"):
                return model_info["model_id"]

        raise RuntimeError("IndexTTS模型加载失败")

    except Exception as e:
        logger.error(f"自动加载语音合成模型失败: {e}")
        raise


def auto_load_for_conversion(audio_path: str, target_speaker: str) -> str:
    """为音色转换自动加载模型"""
    try:
        # 预加载DDSP-SVC模型
        results = preload_models_for_task("voice_conversion")

        # 获取DDSP-SVC模型
        models = get_model_manager().list_models()

        for model_info in models:
            if (model_info["model_type"] == "ddsp_svc" and
                model_info["status"] == "loaded"):
                return model_info["model_id"]

        raise RuntimeError("DDSP-SVC模型加载失败")

    except Exception as e:
        logger.error(f"自动加载音色转换模型失败: {e}")
        raise
