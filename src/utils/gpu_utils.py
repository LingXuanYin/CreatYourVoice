"""GPU工具函数模块

这个模块提供GPU设备检测、内存监控和管理功能。
设计原则：
1. 简单直接 - 避免复杂的抽象层
2. 统一接口 - 无论CUDA还是CPU都用同一套接口
3. 实时监控 - 提供准确的GPU状态信息
"""

import os
import gc
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU设备信息"""
    device_id: int
    name: str
    total_memory: int  # 总内存(MB)
    free_memory: int   # 可用内存(MB)
    used_memory: int   # 已用内存(MB)
    utilization: float # GPU利用率(%)
    temperature: Optional[float] = None  # 温度(°C)
    power_usage: Optional[float] = None  # 功耗(W)


@dataclass
class SystemMemoryInfo:
    """系统内存信息"""
    total: int      # 总内存(MB)
    available: int  # 可用内存(MB)
    used: int       # 已用内存(MB)
    percent: float  # 使用率(%)


class GPUUtils:
    """GPU工具类

    提供GPU设备检测、内存监控等功能。
    设计原则：简单的静态方法，避免状态管理的复杂性。
    """

    _torch_available = None
    _cuda_available = None

    @classmethod
    def is_torch_available(cls) -> bool:
        """检查PyTorch是否可用"""
        if cls._torch_available is None:
            try:
                import torch
                cls._torch_available = True
            except ImportError:
                cls._torch_available = False
        return cls._torch_available

    @classmethod
    def is_cuda_available(cls) -> bool:
        """检查CUDA是否可用"""
        if cls._cuda_available is None:
            if cls.is_torch_available():
                import torch
                cls._cuda_available = torch.cuda.is_available()
            else:
                cls._cuda_available = False
        return cls._cuda_available

    @classmethod
    def get_device_count(cls) -> int:
        """获取GPU设备数量"""
        if not cls.is_cuda_available():
            return 0

        import torch
        return torch.cuda.device_count()

    @classmethod
    def get_current_device(cls) -> int:
        """获取当前GPU设备ID"""
        if not cls.is_cuda_available():
            return -1

        import torch
        return torch.cuda.current_device()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """获取GPU设备名称"""
        if not cls.is_cuda_available():
            return "CPU"

        try:
            import torch
            return torch.cuda.get_device_name(device_id)
        except Exception:
            return f"GPU-{device_id}"

    @classmethod
    def get_gpu_memory_info(cls, device_id: int = 0) -> Dict[str, int]:
        """获取GPU内存信息

        Returns:
            包含total, free, used的字典，单位为MB
        """
        if not cls.is_cuda_available():
            return {"total": 0, "free": 0, "used": 0}

        try:
            import torch
            torch.cuda.set_device(device_id)

            # 获取内存信息（字节）
            total_bytes = torch.cuda.get_device_properties(device_id).total_memory
            reserved_bytes = torch.cuda.memory_reserved(device_id)
            allocated_bytes = torch.cuda.memory_allocated(device_id)

            # 转换为MB
            total_mb = total_bytes // (1024 * 1024)
            reserved_mb = reserved_bytes // (1024 * 1024)
            allocated_mb = allocated_bytes // (1024 * 1024)
            free_mb = total_mb - reserved_mb

            return {
                "total": total_mb,
                "free": free_mb,
                "used": allocated_mb,
                "reserved": reserved_mb
            }

        except Exception as e:
            logger.warning(f"获取GPU内存信息失败: {e}")
            return {"total": 0, "free": 0, "used": 0}

    @classmethod
    def get_gpu_utilization(cls, device_id: int = 0) -> float:
        """获取GPU利用率

        Returns:
            GPU利用率百分比 (0-100)
        """
        if not cls.is_cuda_available():
            return 0.0

        try:
            # 尝试使用nvidia-ml-py获取利用率
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except ImportError:
            # 如果没有pynvml，返回估算值
            memory_info = cls.get_gpu_memory_info(device_id)
            if memory_info["total"] > 0:
                return (memory_info["used"] / memory_info["total"]) * 100
            return 0.0
        except Exception as e:
            logger.debug(f"获取GPU利用率失败: {e}")
            return 0.0

    @classmethod
    def get_all_gpu_info(cls) -> List[GPUInfo]:
        """获取所有GPU设备信息"""
        gpu_list = []

        if not cls.is_cuda_available():
            return gpu_list

        device_count = cls.get_device_count()
        for device_id in range(device_count):
            try:
                name = cls.get_device_name(device_id)
                memory_info = cls.get_gpu_memory_info(device_id)
                utilization = cls.get_gpu_utilization(device_id)

                gpu_info = GPUInfo(
                    device_id=device_id,
                    name=name,
                    total_memory=memory_info["total"],
                    free_memory=memory_info["free"],
                    used_memory=memory_info["used"],
                    utilization=utilization
                )
                gpu_list.append(gpu_info)

            except Exception as e:
                logger.warning(f"获取GPU {device_id} 信息失败: {e}")

        return gpu_list

    @classmethod
    def get_system_memory_info(cls) -> SystemMemoryInfo:
        """获取系统内存信息"""
        try:
            memory = psutil.virtual_memory()
            return SystemMemoryInfo(
                total=memory.total // (1024 * 1024),
                available=memory.available // (1024 * 1024),
                used=memory.used // (1024 * 1024),
                percent=memory.percent
            )
        except Exception as e:
            logger.warning(f"获取系统内存信息失败: {e}")
            return SystemMemoryInfo(0, 0, 0, 0.0)

    @classmethod
    def clear_gpu_cache(cls, device_id: Optional[int] = None) -> bool:
        """清理GPU缓存

        Args:
            device_id: GPU设备ID，None表示清理所有设备

        Returns:
            是否成功清理
        """
        if not cls.is_cuda_available():
            return True

        try:
            import torch

            if device_id is not None:
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                # 清理所有设备
                for i in range(cls.get_device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            # 强制垃圾回收
            gc.collect()

            logger.info(f"GPU缓存清理完成: device_id={device_id}")
            return True

        except Exception as e:
            logger.error(f"清理GPU缓存失败: {e}")
            return False

    @classmethod
    def get_optimal_device(cls) -> str:
        """获取最优计算设备

        Returns:
            设备字符串: "cuda:0", "cuda:1", "cpu"
        """
        if not cls.is_cuda_available():
            return "cpu"

        # 选择内存最多的GPU
        gpu_list = cls.get_all_gpu_info()
        if not gpu_list:
            return "cpu"

        # 按可用内存排序
        gpu_list.sort(key=lambda x: x.free_memory, reverse=True)
        best_gpu = gpu_list[0]

        # 如果最好的GPU可用内存少于1GB，使用CPU
        if best_gpu.free_memory < 1024:
            logger.warning("GPU内存不足，使用CPU")
            return "cpu"

        return f"cuda:{best_gpu.device_id}"

    @classmethod
    def check_memory_requirement(cls, required_mb: int, device: str = "auto") -> bool:
        """检查内存需求是否满足

        Args:
            required_mb: 需要的内存大小(MB)
            device: 设备类型

        Returns:
            是否满足内存需求
        """
        if device == "auto":
            device = cls.get_optimal_device()

        if device == "cpu":
            system_memory = cls.get_system_memory_info()
            return system_memory.available >= required_mb

        if device.startswith("cuda:"):
            try:
                device_id = int(device.split(":")[1])
                memory_info = cls.get_gpu_memory_info(device_id)
                return memory_info["free"] >= required_mb
            except (ValueError, IndexError):
                return False

        return False

    @classmethod
    def estimate_model_memory(cls, model_path: str) -> int:
        """估算模型内存需求

        Args:
            model_path: 模型文件路径

        Returns:
            估算的内存需求(MB)
        """
        try:
            path_obj = Path(model_path)
            if not path_obj.exists():
                return 0

            # 基于文件大小估算
            file_size_mb = path_obj.stat().st_size // (1024 * 1024)

            # 经验公式：模型运行时内存约为文件大小的2-3倍
            estimated_mb = int(file_size_mb * 2.5)

            return estimated_mb

        except Exception as e:
            logger.warning(f"估算模型内存失败: {e}")
            return 1024  # 默认1GB


class GPUMonitor:
    """GPU监控器

    提供实时GPU状态监控功能。
    """

    def __init__(self, update_interval: float = 1.0):
        """初始化监控器

        Args:
            update_interval: 更新间隔(秒)
        """
        self.update_interval = update_interval
        self._monitoring = False
        self._last_update = 0.0
        self._cached_info = {}

    def start_monitoring(self) -> None:
        """开始监控"""
        self._monitoring = True
        logger.info("GPU监控已启动")

    def stop_monitoring(self) -> None:
        """停止监控"""
        self._monitoring = False
        logger.info("GPU监控已停止")

    def get_current_status(self, force_update: bool = False) -> Dict[str, Any]:
        """获取当前状态

        Args:
            force_update: 是否强制更新

        Returns:
            包含GPU和系统状态的字典
        """
        current_time = time.time()

        # 检查是否需要更新
        if (not force_update and
            current_time - self._last_update < self.update_interval and
            self._cached_info):
            return self._cached_info

        # 更新状态信息
        status = {
            "timestamp": current_time,
            "cuda_available": GPUUtils.is_cuda_available(),
            "device_count": GPUUtils.get_device_count(),
            "current_device": GPUUtils.get_current_device(),
            "optimal_device": GPUUtils.get_optimal_device(),
            "gpu_list": [gpu.__dict__ for gpu in GPUUtils.get_all_gpu_info()],
            "system_memory": GPUUtils.get_system_memory_info().__dict__
        }

        self._cached_info = status
        self._last_update = current_time

        return status

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        status = self.get_current_status()

        summary = {
            "system_memory_used_percent": status["system_memory"]["percent"],
            "gpu_count": status["device_count"],
            "gpu_memory_info": []
        }

        for gpu in status["gpu_list"]:
            gpu_summary = {
                "device_id": gpu["device_id"],
                "name": gpu["name"],
                "memory_used_percent": (gpu["used_memory"] / gpu["total_memory"] * 100) if gpu["total_memory"] > 0 else 0,
                "memory_used_mb": gpu["used_memory"],
                "memory_total_mb": gpu["total_memory"],
                "utilization": gpu["utilization"]
            }
            summary["gpu_memory_info"].append(gpu_summary)

        return summary
