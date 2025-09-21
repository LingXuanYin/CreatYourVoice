"""内存监控模块

这个模块提供实时内存监控和警告功能。
设计原则：
1. 轻量级监控 - 最小化监控开销
2. 阈值警告 - 内存不足时及时警告
3. 历史记录 - 保留内存使用历史用于分析
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging

from .gpu_utils import GPUUtils, GPUInfo, SystemMemoryInfo

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照"""
    timestamp: float
    system_memory: SystemMemoryInfo
    gpu_memory: List[GPUInfo]
    total_gpu_used: int = 0
    total_gpu_free: int = 0

    def __post_init__(self):
        """计算GPU总内存"""
        self.total_gpu_used = sum(gpu.used_memory for gpu in self.gpu_memory)
        self.total_gpu_free = sum(gpu.free_memory for gpu in self.gpu_memory)


@dataclass
class MemoryAlert:
    """内存警告"""
    timestamp: float
    alert_type: str  # "warning", "critical", "info"
    device: str      # "system", "gpu:0", "gpu:1", etc.
    message: str
    memory_usage: float  # 使用率百分比
    threshold: float     # 触发阈值


class MemoryMonitor:
    """内存监控器

    提供实时内存监控、警告和历史记录功能。
    设计原则：单线程监控，避免复杂的并发问题。
    """

    def __init__(
        self,
        update_interval: float = 2.0,
        history_size: int = 100,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0
    ):
        """初始化内存监控器

        Args:
            update_interval: 监控更新间隔(秒)
            history_size: 历史记录数量
            warning_threshold: 警告阈值(%)
            critical_threshold: 严重警告阈值(%)
        """
        self.update_interval = update_interval
        self.history_size = history_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # 监控状态
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 数据存储
        self._memory_history: deque = deque(maxlen=history_size)
        self._alerts: deque = deque(maxlen=50)  # 保留最近50个警告

        # 回调函数
        self._alert_callbacks: List[Callable[[MemoryAlert], None]] = []

        # 上次警告时间（避免重复警告）
        self._last_alert_time: Dict[str, float] = {}
        self._alert_cooldown = 30.0  # 30秒冷却时间

        logger.info(f"内存监控器初始化完成，更新间隔: {update_interval}s")

    def add_alert_callback(self, callback: Callable[[MemoryAlert], None]) -> None:
        """添加警告回调函数"""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[MemoryAlert], None]) -> None:
        """移除警告回调函数"""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def start_monitoring(self) -> bool:
        """开始监控

        Returns:
            是否成功启动
        """
        if self._monitoring:
            logger.warning("内存监控已在运行")
            return True

        try:
            self._stop_event.clear()
            self._monitoring = True

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="MemoryMonitor",
                daemon=True
            )
            self._monitor_thread.start()

            logger.info("内存监控已启动")
            return True

        except Exception as e:
            logger.error(f"启动内存监控失败: {e}")
            self._monitoring = False
            return False

    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        logger.info("内存监控已停止")

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._monitoring and not self._stop_event.is_set():
            try:
                # 获取内存快照
                snapshot = self._take_snapshot()
                self._memory_history.append(snapshot)

                # 检查警告条件
                self._check_alerts(snapshot)

                # 等待下次更新
                if self._stop_event.wait(self.update_interval):
                    break

            except Exception as e:
                logger.error(f"内存监控循环异常: {e}")
                time.sleep(1.0)  # 异常时短暂等待

    def _take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        timestamp = time.time()
        system_memory = GPUUtils.get_system_memory_info()
        gpu_memory = GPUUtils.get_all_gpu_info()

        return MemorySnapshot(
            timestamp=timestamp,
            system_memory=system_memory,
            gpu_memory=gpu_memory
        )

    def _check_alerts(self, snapshot: MemorySnapshot) -> None:
        """检查警告条件"""
        current_time = snapshot.timestamp

        # 检查系统内存
        self._check_system_memory_alert(snapshot, current_time)

        # 检查GPU内存
        for gpu in snapshot.gpu_memory:
            self._check_gpu_memory_alert(gpu, current_time)

    def _check_system_memory_alert(self, snapshot: MemorySnapshot, current_time: float) -> None:
        """检查系统内存警告"""
        usage_percent = snapshot.system_memory.percent
        device_key = "system"

        # 检查冷却时间
        if self._is_in_cooldown(device_key, current_time):
            return

        alert_type = None
        if usage_percent >= self.critical_threshold:
            alert_type = "critical"
        elif usage_percent >= self.warning_threshold:
            alert_type = "warning"

        if alert_type:
            alert = MemoryAlert(
                timestamp=current_time,
                alert_type=alert_type,
                device=device_key,
                message=f"系统内存使用率过高: {usage_percent:.1f}% (阈值: {self.warning_threshold:.1f}%)",
                memory_usage=usage_percent,
                threshold=self.warning_threshold if alert_type == "warning" else self.critical_threshold
            )

            self._trigger_alert(alert)
            self._last_alert_time[device_key] = current_time

    def _check_gpu_memory_alert(self, gpu: GPUInfo, current_time: float) -> None:
        """检查GPU内存警告"""
        if gpu.total_memory == 0:
            return

        usage_percent = (gpu.used_memory / gpu.total_memory) * 100
        device_key = f"gpu:{gpu.device_id}"

        # 检查冷却时间
        if self._is_in_cooldown(device_key, current_time):
            return

        alert_type = None
        if usage_percent >= self.critical_threshold:
            alert_type = "critical"
        elif usage_percent >= self.warning_threshold:
            alert_type = "warning"

        if alert_type:
            alert = MemoryAlert(
                timestamp=current_time,
                alert_type=alert_type,
                device=device_key,
                message=f"GPU {gpu.device_id} ({gpu.name}) 内存使用率过高: {usage_percent:.1f}% (阈值: {self.warning_threshold:.1f}%)",
                memory_usage=usage_percent,
                threshold=self.warning_threshold if alert_type == "warning" else self.critical_threshold
            )

            self._trigger_alert(alert)
            self._last_alert_time[device_key] = current_time

    def _is_in_cooldown(self, device_key: str, current_time: float) -> bool:
        """检查是否在冷却时间内"""
        last_time = self._last_alert_time.get(device_key, 0)
        return current_time - last_time < self._alert_cooldown

    def _trigger_alert(self, alert: MemoryAlert) -> None:
        """触发警告"""
        self._alerts.append(alert)

        # 记录日志
        log_level = logging.CRITICAL if alert.alert_type == "critical" else logging.WARNING
        logger.log(log_level, alert.message)

        # 调用回调函数
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"警告回调函数执行失败: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        if not self._memory_history:
            # 如果没有历史记录，立即获取一次
            snapshot = self._take_snapshot()
            return self._snapshot_to_dict(snapshot)

        # 返回最新的快照
        latest_snapshot = self._memory_history[-1]
        return self._snapshot_to_dict(latest_snapshot)

    def _snapshot_to_dict(self, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """将快照转换为字典"""
        return {
            "timestamp": snapshot.timestamp,
            "system_memory": {
                "total_mb": snapshot.system_memory.total,
                "used_mb": snapshot.system_memory.used,
                "available_mb": snapshot.system_memory.available,
                "usage_percent": snapshot.system_memory.percent
            },
            "gpu_memory": [
                {
                    "device_id": gpu.device_id,
                    "name": gpu.name,
                    "total_mb": gpu.total_memory,
                    "used_mb": gpu.used_memory,
                    "free_mb": gpu.free_memory,
                    "usage_percent": (gpu.used_memory / gpu.total_memory * 100) if gpu.total_memory > 0 else 0,
                    "utilization": gpu.utilization
                }
                for gpu in snapshot.gpu_memory
            ],
            "summary": {
                "total_gpu_used_mb": snapshot.total_gpu_used,
                "total_gpu_free_mb": snapshot.total_gpu_free,
                "gpu_count": len(snapshot.gpu_memory)
            }
        }

    def get_memory_history(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """获取内存使用历史

        Args:
            minutes: 获取最近多少分钟的历史

        Returns:
            历史记录列表
        """
        if not self._memory_history:
            return []

        cutoff_time = time.time() - (minutes * 60)
        history = []

        for snapshot in self._memory_history:
            if snapshot.timestamp >= cutoff_time:
                history.append(self._snapshot_to_dict(snapshot))

        return history

    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近的警告

        Args:
            minutes: 获取最近多少分钟的警告

        Returns:
            警告列表
        """
        if not self._alerts:
            return []

        cutoff_time = time.time() - (minutes * 60)
        recent_alerts = []

        for alert in self._alerts:
            if alert.timestamp >= cutoff_time:
                recent_alerts.append({
                    "timestamp": alert.timestamp,
                    "alert_type": alert.alert_type,
                    "device": alert.device,
                    "message": alert.message,
                    "memory_usage": alert.memory_usage,
                    "threshold": alert.threshold
                })

        return recent_alerts

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._memory_history:
            return {}

        # 计算统计信息
        system_usage = [s.system_memory.percent for s in self._memory_history]
        gpu_usage = []

        for snapshot in self._memory_history:
            for gpu in snapshot.gpu_memory:
                if gpu.total_memory > 0:
                    usage = (gpu.used_memory / gpu.total_memory) * 100
                    gpu_usage.append(usage)

        stats = {
            "monitoring_duration_minutes": (time.time() - self._memory_history[0].timestamp) / 60,
            "snapshot_count": len(self._memory_history),
            "alert_count": len(self._alerts),
            "system_memory": {
                "avg_usage": sum(system_usage) / len(system_usage) if system_usage else 0,
                "max_usage": max(system_usage) if system_usage else 0,
                "min_usage": min(system_usage) if system_usage else 0
            },
            "gpu_memory": {
                "avg_usage": sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0,
                "max_usage": max(gpu_usage) if gpu_usage else 0,
                "min_usage": min(gpu_usage) if gpu_usage else 0
            }
        }

        return stats

    def clear_history(self) -> None:
        """清空历史记录"""
        self._memory_history.clear()
        self._alerts.clear()
        self._last_alert_time.clear()
        logger.info("内存监控历史记录已清空")

    def __del__(self):
        """析构函数"""
        self.stop_monitoring()


# 全局内存监控器实例
_global_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """获取全局内存监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def start_global_monitoring() -> bool:
    """启动全局内存监控"""
    monitor = get_memory_monitor()
    return monitor.start_monitoring()


def stop_global_monitoring() -> None:
    """停止全局内存监控"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
