"""错误处理和恢复机制

这个模块提供统一的错误处理、恢复和降级策略。
设计原则：
1. 快速失败 - 立即发现和报告错误
2. 优雅降级 - 提供备用方案
3. 自动恢复 - 尝试自动修复问题
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, Optional, Type, Union, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 轻微错误，不影响核心功能
    MEDIUM = "medium"     # 中等错误，影响部分功能
    HIGH = "high"         # 严重错误，影响核心功能
    CRITICAL = "critical" # 致命错误，系统无法继续运行


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"               # 重试操作
    FALLBACK = "fallback"         # 使用备用方案
    DEGRADE = "degrade"           # 功能降级
    ABORT = "abort"               # 中止操作
    RESTART = "restart"           # 重启组件


@dataclass
class ErrorContext:
    """错误上下文信息"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    timestamp: float
    traceback_info: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class RecoveryAction:
    """恢复动作"""
    strategy: RecoveryStrategy
    action: Callable[[], Any]
    description: str
    max_attempts: int = 3
    delay_seconds: float = 1.0


class ErrorHandler:
    """错误处理器

    提供统一的错误处理、记录和恢复功能。
    """

    def __init__(self):
        """初始化错误处理器"""
        self.error_history: List[ErrorContext] = []
        self.max_history_size = 1000
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}
        self.error_callbacks: List[Callable[[ErrorContext], None]] = []

        # 统计信息
        self.error_counts: Dict[str, int] = {}
        self.recovery_success_counts: Dict[str, int] = {}

        logger.info("错误处理器初始化完成")

    def register_recovery_action(
        self,
        error_pattern: str,
        action: RecoveryAction
    ) -> None:
        """注册恢复动作

        Args:
            error_pattern: 错误模式（错误类型或组件名）
            action: 恢复动作
        """
        if error_pattern not in self.recovery_actions:
            self.recovery_actions[error_pattern] = []

        self.recovery_actions[error_pattern].append(action)
        logger.info(f"注册恢复动作: {error_pattern} -> {action.description}")

    def add_error_callback(self, callback: Callable[[ErrorContext], None]) -> None:
        """添加错误回调函数"""
        self.error_callbacks.append(callback)

    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """处理错误

        Args:
            error: 异常对象
            component: 组件名称
            operation: 操作名称
            severity: 错误严重程度
            additional_info: 附加信息

        Returns:
            是否成功恢复
        """
        # 创建错误上下文
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            component=component,
            operation=operation,
            timestamp=time.time(),
            traceback_info=traceback.format_exc(),
            additional_info=additional_info or {}
        )

        # 记录错误
        self._log_error(error_context)

        # 更新统计
        self._update_error_statistics(error_context)

        # 添加到历史记录
        self._add_to_history(error_context)

        # 通知回调函数
        self._notify_callbacks(error_context)

        # 尝试恢复
        recovery_success = self._attempt_recovery(error_context)

        return recovery_success

    def _log_error(self, context: ErrorContext) -> None:
        """记录错误日志"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(context.severity, logging.ERROR)

        message = (
            f"[{context.component}] {context.operation} 失败: "
            f"{context.error_type}: {context.error_message}"
        )

        logger.log(log_level, message)

        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.debug(f"错误堆栈:\n{context.traceback_info}")

    def _update_error_statistics(self, context: ErrorContext) -> None:
        """更新错误统计"""
        error_key = f"{context.component}.{context.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    def _add_to_history(self, context: ErrorContext) -> None:
        """添加到历史记录"""
        self.error_history.append(context)

        # 限制历史记录大小
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)

    def _notify_callbacks(self, context: ErrorContext) -> None:
        """通知回调函数"""
        for callback in self.error_callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"错误回调函数执行失败: {e}")

    def _attempt_recovery(self, context: ErrorContext) -> bool:
        """尝试恢复"""
        # 查找匹配的恢复动作
        recovery_actions = self._find_recovery_actions(context)

        if not recovery_actions:
            logger.debug(f"未找到 {context.component}.{context.error_type} 的恢复动作")
            return False

        # 按优先级执行恢复动作
        for action in recovery_actions:
            success = self._execute_recovery_action(action, context)
            if success:
                recovery_key = f"{context.component}.{action.strategy.value}"
                self.recovery_success_counts[recovery_key] = (
                    self.recovery_success_counts.get(recovery_key, 0) + 1
                )
                logger.info(f"恢复成功: {action.description}")
                return True

        logger.warning(f"所有恢复尝试失败: {context.component}.{context.operation}")
        return False

    def _find_recovery_actions(self, context: ErrorContext) -> List[RecoveryAction]:
        """查找匹配的恢复动作"""
        actions = []

        # 精确匹配：组件.错误类型
        exact_key = f"{context.component}.{context.error_type}"
        if exact_key in self.recovery_actions:
            actions.extend(self.recovery_actions[exact_key])

        # 组件匹配
        if context.component in self.recovery_actions:
            actions.extend(self.recovery_actions[context.component])

        # 错误类型匹配
        if context.error_type in self.recovery_actions:
            actions.extend(self.recovery_actions[context.error_type])

        # 通用匹配
        if "default" in self.recovery_actions:
            actions.extend(self.recovery_actions["default"])

        return actions

    def _execute_recovery_action(
        self,
        action: RecoveryAction,
        context: ErrorContext
    ) -> bool:
        """执行恢复动作"""
        for attempt in range(action.max_attempts):
            try:
                logger.info(f"执行恢复动作 (尝试 {attempt + 1}/{action.max_attempts}): {action.description}")

                result = action.action()

                # 如果动作返回布尔值，使用它作为成功标志
                if isinstance(result, bool):
                    if result:
                        return True
                else:
                    # 如果没有异常，认为成功
                    return True

            except Exception as e:
                logger.warning(f"恢复动作失败 (尝试 {attempt + 1}): {e}")

                if attempt < action.max_attempts - 1:
                    time.sleep(action.delay_seconds)

        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        total_errors = sum(self.error_counts.values())
        total_recoveries = sum(self.recovery_success_counts.values())

        return {
            "total_errors": total_errors,
            "total_recoveries": total_recoveries,
            "recovery_rate": total_recoveries / total_errors if total_errors > 0 else 0,
            "error_counts": self.error_counts.copy(),
            "recovery_success_counts": self.recovery_success_counts.copy(),
            "recent_errors": len([
                e for e in self.error_history
                if time.time() - e.timestamp < 3600  # 最近1小时
            ])
        }

    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取最近的错误"""
        cutoff_time = time.time() - (hours * 3600)

        recent_errors = []
        for error in self.error_history:
            if error.timestamp >= cutoff_time:
                recent_errors.append({
                    "timestamp": error.timestamp,
                    "component": error.component,
                    "operation": error.operation,
                    "error_type": error.error_type,
                    "error_message": error.error_message,
                    "severity": error.severity.value
                })

        return recent_errors

    def clear_history(self) -> None:
        """清空错误历史"""
        self.error_history.clear()
        self.error_counts.clear()
        self.recovery_success_counts.clear()
        logger.info("错误历史已清空")


# 装饰器：自动错误处理
def handle_errors(
    component: str,
    operation: Optional[str] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    fallback_return: Any = None
):
    """错误处理装饰器

    Args:
        component: 组件名称
        operation: 操作名称（默认使用函数名）
        severity: 错误严重程度
        fallback_return: 发生错误时的返回值
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__

            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                recovery_success = handler.handle_error(
                    error=e,
                    component=component,
                    operation=op_name,
                    severity=severity
                )

                if not recovery_success:
                    if severity == ErrorSeverity.CRITICAL:
                        raise
                    return fallback_return

                # 如果恢复成功，重试一次
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if severity == ErrorSeverity.CRITICAL:
                        raise
                    return fallback_return

        return wrapper
    return decorator


# 全局错误处理器实例
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
        _setup_default_recovery_actions(_global_error_handler)
    return _global_error_handler


def _setup_default_recovery_actions(handler: ErrorHandler) -> None:
    """设置默认恢复动作"""
    # GPU内存不足恢复
    def clear_gpu_cache():
        from .gpu_utils import GPUUtils
        return GPUUtils.clear_gpu_cache()

    handler.register_recovery_action(
        "RuntimeError",
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            action=clear_gpu_cache,
            description="清理GPU缓存",
            max_attempts=1
        )
    )

    # 模型加载失败恢复
    def fallback_to_cpu():
        logger.info("GPU模型加载失败，回退到CPU模式")
        return True

    handler.register_recovery_action(
        "gpu_model_manager.ModelLoadError",
        RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK,
            action=fallback_to_cpu,
            description="回退到CPU模式",
            max_attempts=1
        )
    )

    # 内存监控恢复
    def restart_memory_monitor():
        from .memory_monitor import get_memory_monitor
        monitor = get_memory_monitor()
        monitor.stop_monitoring()
        time.sleep(1.0)
        return monitor.start_monitoring()

    handler.register_recovery_action(
        "memory_monitor",
        RecoveryAction(
            strategy=RecoveryStrategy.RESTART,
            action=restart_memory_monitor,
            description="重启内存监控",
            max_attempts=2,
            delay_seconds=2.0
        )
    )


# 便捷函数
def handle_error(
    error: Exception,
    component: str,
    operation: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> bool:
    """便捷的错误处理函数"""
    handler = get_error_handler()
    return handler.handle_error(error, component, operation, severity)


def register_recovery_action(error_pattern: str, action: RecoveryAction) -> None:
    """便捷的恢复动作注册函数"""
    handler = get_error_handler()
    handler.register_recovery_action(error_pattern, action)
