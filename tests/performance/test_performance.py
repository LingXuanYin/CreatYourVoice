"""性能基准测试

这个模块测试系统的性能指标和资源使用情况。
设计原则：
1. 基准测试 - 建立性能基线，监控性能回归
2. 资源监控 - 测量内存、CPU使用情况
3. 扩展性测试 - 测试系统在不同负载下的表现
"""

import unittest
import tempfile
import shutil
import time
import threading
import gc
import sys
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.voice_manager import VoiceManager
from src.core.weight_calculator import WeightCalculator
from src.core.voice_fusion import VoiceFuser, FusionSource, FusionConfig
from src.core.voice_inheritance import VoiceInheritor, InheritanceConfig
from src.core.voice_base_creator import VoiceBaseCreator, VoiceBaseCreationParams
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    throughput: Optional[float] = None  # 操作/秒
    items_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "operation": self.operation_name,
            "execution_time_s": round(self.execution_time, 3),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "cpu_usage_percent": round(self.cpu_usage_percent, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "throughput_ops_per_sec": round(self.throughput, 2) if self.throughput else None,
            "items_processed": self.items_processed
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    baseline_metrics: Optional[Dict[str, float]] = None
    performance_regression: bool = False
    warnings: List[str] = field(default_factory=list)

    def add_metric(self, metric: PerformanceMetrics) -> None:
        """添加性能指标"""
        self.metrics.append(metric)

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        if not self.metrics:
            return {"test_name": self.test_name, "status": "no_metrics"}

        total_time = sum(m.execution_time for m in self.metrics)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics) / len(self.metrics)
        peak_memory = max(m.peak_memory_mb for m in self.metrics)

        return {
            "test_name": self.test_name,
            "total_execution_time_s": round(total_time, 3),
            "average_memory_mb": round(avg_memory, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "operations_count": len(self.metrics),
            "performance_regression": self.performance_regression,
            "warnings_count": len(self.warnings)
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.process = psutil.Process()
        self._monitoring = False
        self._peak_memory = 0.0
        self._cpu_samples = []

    def start_monitoring(self):
        """开始监控"""
        self._monitoring = True
        self._peak_memory = 0.0
        self._cpu_samples = []

        def monitor():
            while self._monitoring:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()

                    self._peak_memory = max(self._peak_memory, memory_mb)
                    self._cpu_samples.append(cpu_percent)

                    time.sleep(0.1)  # 100ms采样间隔
                except:
                    break

        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> Tuple[float, float, float]:
        """停止监控并返回指标"""
        self._monitoring = False

        current_memory = self.process.memory_info().rss / 1024 / 1024
        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples) if self._cpu_samples else 0.0

        return current_memory, avg_cpu, self._peak_memory


class PerformanceTestCase(unittest.TestCase):
    """性能测试基类"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="WARNING", console_output=False)  # 减少日志输出

        self.temp_dir = Path(tempfile.mkdtemp())
        self.voices_dir = self.temp_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.voice_manager = VoiceManager(self.voices_dir)
        self.monitor = PerformanceMonitor()

        # 性能基线（可以根据实际情况调整）
        self.performance_baselines = {
            "voice_creation_time": 2.0,  # 秒
            "voice_loading_time": 0.1,   # 秒
            "weight_calculation_time": 0.01,  # 秒
            "fusion_time": 1.0,          # 秒
            "memory_usage_mb": 100.0,    # MB
        }

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        # 强制垃圾回收
        gc.collect()

    def measure_performance(self, operation_name: str, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """测量操作性能"""
        # 垃圾回收
        gc.collect()

        # 开始监控
        self.monitor.start_monitoring()

        # 执行操作
        start_time = time.time()
        result = operation_func(*args, **kwargs)
        execution_time = time.time() - start_time

        # 停止监控
        memory_usage, cpu_usage, peak_memory = self.monitor.stop_monitoring()

        return PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            peak_memory_mb=peak_memory
        )

    def create_test_voice(self, name: str) -> VoiceConfig:
        """创建测试音色"""
        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1
        )

        index_config = IndexTTSConfig(
            model_path="test_index_model",
            config_path="test_index_config.yaml",
            speaker_name="test_speaker"
        )

        return VoiceConfig(
            name=name,
            description=f"性能测试音色: {name}",
            tags=["性能测试"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )


class VoiceManagerPerformanceTest(PerformanceTestCase):
    """音色管理器性能测试"""

    def test_voice_creation_performance(self):
        """测试音色创建性能"""
        logger.info("测试音色创建性能")

        result = BenchmarkResult("voice_creation_performance")

        # 测试单个音色创建
        def create_single_voice():
            voice = self.create_test_voice("性能测试音色")
            self.voice_manager.save_voice(voice)
            return voice

        metric = self.measure_performance("single_voice_creation", create_single_voice)
        result.add_metric(metric)

        # 验证性能基线
        if metric.execution_time > self.performance_baselines["voice_creation_time"]:
            result.performance_regression = True
            result.warnings.append(f"音色创建时间超过基线: {metric.execution_time:.3f}s > {self.performance_baselines['voice_creation_time']}s")

        logger.info(f"单个音色创建: {metric.execution_time:.3f}s, 内存: {metric.memory_usage_mb:.2f}MB")

        # 测试批量音色创建
        def create_batch_voices():
            voices = []
            for i in range(100):
                voice = self.create_test_voice(f"批量音色_{i}")
                self.voice_manager.save_voice(voice)
                voices.append(voice)
            return voices

        batch_metric = self.measure_performance("batch_voice_creation", create_batch_voices)
        batch_metric.items_processed = 100
        batch_metric.throughput = 100 / batch_metric.execution_time
        result.add_metric(batch_metric)

        logger.info(f"批量音色创建(100个): {batch_metric.execution_time:.3f}s, 吞吐量: {batch_metric.throughput:.2f} ops/s")

        return result

    def test_voice_loading_performance(self):
        """测试音色加载性能"""
        logger.info("测试音色加载性能")

        result = BenchmarkResult("voice_loading_performance")

        # 预创建音色
        voices = []
        for i in range(50):
            voice = self.create_test_voice(f"加载测试音色_{i}")
            self.voice_manager.save_voice(voice)
            voices.append(voice)

        # 测试单个音色加载
        def load_single_voice():
            return self.voice_manager.load_voice(voices[0].voice_id)

        metric = self.measure_performance("single_voice_loading", load_single_voice)
        result.add_metric(metric)

        # 验证性能基线
        if metric.execution_time > self.performance_baselines["voice_loading_time"]:
            result.performance_regression = True
            result.warnings.append(f"音色加载时间超过基线: {metric.execution_time:.3f}s > {self.performance_baselines['voice_loading_time']}s")

        # 测试批量音色加载
        def load_batch_voices():
            loaded_voices = []
            for voice in voices:
                loaded_voice = self.voice_manager.load_voice(voice.voice_id)
                loaded_voices.append(loaded_voice)
            return loaded_voices

        batch_metric = self.measure_performance("batch_voice_loading", load_batch_voices)
        batch_metric.items_processed = len(voices)
        batch_metric.throughput = len(voices) / batch_metric.execution_time
        result.add_metric(batch_metric)

        logger.info(f"批量音色加载({len(voices)}个): {batch_metric.execution_time:.3f}s, 吞吐量: {batch_metric.throughput:.2f} ops/s")

        return result

    def test_voice_search_performance(self):
        """测试音色搜索性能"""
        logger.info("测试音色搜索性能")

        result = BenchmarkResult("voice_search_performance")

        # 创建大量音色用于搜索测试
        for i in range(1000):
            voice = self.create_test_voice(f"搜索测试音色_{i}")
            voice.tags = [f"tag_{i % 10}", f"category_{i % 5}"]
            self.voice_manager.save_voice(voice)

        # 测试名称搜索
        def search_by_name():
            return self.voice_manager.search_voices(name_pattern="搜索测试")

        name_metric = self.measure_performance("search_by_name", search_by_name)
        result.add_metric(name_metric)

        # 测试标签搜索
        def search_by_tags():
            return self.voice_manager.search_voices(tags=["tag_1"])

        tag_metric = self.measure_performance("search_by_tags", search_by_tags)
        result.add_metric(tag_metric)

        logger.info(f"名称搜索: {name_metric.execution_time:.3f}s")
        logger.info(f"标签搜索: {tag_metric.execution_time:.3f}s")

        return result


class WeightCalculationPerformanceTest(PerformanceTestCase):
    """权重计算性能测试"""

    def test_weight_normalization_performance(self):
        """测试权重归一化性能"""
        logger.info("测试权重归一化性能")

        result = BenchmarkResult("weight_normalization_performance")

        # 测试小规模权重归一化
        def normalize_small_weights():
            weights = {f"speaker_{i:03d}": float(i + 1) for i in range(10)}
            return WeightCalculator.normalize_weights(weights)

        small_metric = self.measure_performance("small_weight_normalization", normalize_small_weights)
        result.add_metric(small_metric)

        # 测试大规模权重归一化
        def normalize_large_weights():
            weights = {f"speaker_{i:03d}": float(i + 1) for i in range(1000)}
            return WeightCalculator.normalize_weights(weights)

        large_metric = self.measure_performance("large_weight_normalization", normalize_large_weights)
        result.add_metric(large_metric)

        # 验证性能基线
        if large_metric.execution_time > self.performance_baselines["weight_calculation_time"]:
            result.performance_regression = True
            result.warnings.append(f"大规模权重计算时间超过基线: {large_metric.execution_time:.3f}s > {self.performance_baselines['weight_calculation_time']}s")

        logger.info(f"小规模权重归一化(10个): {small_metric.execution_time:.3f}s")
        logger.info(f"大规模权重归一化(1000个): {large_metric.execution_time:.3f}s")

        return result

    def test_weight_merging_performance(self):
        """测试权重合并性能"""
        logger.info("测试权重合并性能")

        result = BenchmarkResult("weight_merging_performance")

        # 准备测试数据
        old_weights = {f"speaker_{i:03d}": float(i + 1) for i in range(100)}
        new_weights = {f"speaker_{i:03d}": float(100 - i) for i in range(100)}

        # 测试权重合并
        def merge_weights():
            return WeightCalculator.merge_weights(old_weights, new_weights, 0.5)

        metric = self.measure_performance("weight_merging", merge_weights)
        result.add_metric(metric)

        logger.info(f"权重合并(100个说话人): {metric.execution_time:.3f}s")

        return result


class VoiceFusionPerformanceTest(PerformanceTestCase):
    """音色融合性能测试"""

    def test_fusion_performance(self):
        """测试音色融合性能"""
        logger.info("测试音色融合性能")

        result = BenchmarkResult("voice_fusion_performance")

        # 创建测试音色
        voices = []
        for i in range(10):
            voice = self.create_test_voice(f"融合性能测试音色_{i}")
            self.voice_manager.save_voice(voice)
            voices.append(voice)

        fuser = VoiceFuser(self.voice_manager)

        # 测试小规模融合（2个音色）
        def small_fusion():
            fusion_sources = [
                FusionSource(voice_config=voices[0], weight=0.6),
                FusionSource(voice_config=voices[1], weight=0.4)
            ]
            return fuser.fuse_voices(fusion_sources, "小规模融合测试")

        small_metric = self.measure_performance("small_fusion", small_fusion)
        result.add_metric(small_metric)

        # 测试大规模融合（10个音色）
        def large_fusion():
            fusion_sources = [
                FusionSource(voice_config=voice, weight=1.0/(i+1))
                for i, voice in enumerate(voices)
            ]
            return fuser.fuse_voices(fusion_sources, "大规模融合测试")

        large_metric = self.measure_performance("large_fusion", large_fusion)
        result.add_metric(large_metric)

        # 验证性能基线
        if large_metric.execution_time > self.performance_baselines["fusion_time"]:
            result.performance_regression = True
            result.warnings.append(f"大规模融合时间超过基线: {large_metric.execution_time:.3f}s > {self.performance_baselines['fusion_time']}s")

        logger.info(f"小规模融合(2个音色): {small_metric.execution_time:.3f}s")
        logger.info(f"大规模融合(10个音色): {large_metric.execution_time:.3f}s")

        return result


class ConcurrencyPerformanceTest(PerformanceTestCase):
    """并发性能测试"""

    def test_concurrent_voice_operations(self):
        """测试并发音色操作"""
        logger.info("测试并发音色操作")

        result = BenchmarkResult("concurrent_voice_operations")

        # 测试并发音色创建
        def concurrent_voice_creation():
            def create_voice(index):
                voice = self.create_test_voice(f"并发测试音色_{index}")
                self.voice_manager.save_voice(voice)
                return voice

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(create_voice, i) for i in range(20)]
                voices = [future.result() for future in as_completed(futures)]

            return voices

        concurrent_metric = self.measure_performance("concurrent_creation", concurrent_voice_creation)
        concurrent_metric.items_processed = 20
        concurrent_metric.throughput = 20 / concurrent_metric.execution_time
        result.add_metric(concurrent_metric)

        logger.info(f"并发音色创建(20个, 4线程): {concurrent_metric.execution_time:.3f}s, 吞吐量: {concurrent_metric.throughput:.2f} ops/s")

        return result


class MemoryUsageTest(PerformanceTestCase):
    """内存使用测试"""

    def test_memory_usage_scaling(self):
        """测试内存使用扩展性"""
        logger.info("测试内存使用扩展性")

        result = BenchmarkResult("memory_usage_scaling")

        # 测试不同规模下的内存使用
        scales = [10, 50, 100, 500]

        for scale in scales:
            def create_voices_at_scale():
                voices = []
                for i in range(scale):
                    voice = self.create_test_voice(f"内存测试音色_{i}")
                    self.voice_manager.save_voice(voice)
                    voices.append(voice)
                return voices

            metric = self.measure_performance(f"memory_test_scale_{scale}", create_voices_at_scale)
            metric.items_processed = scale
            result.add_metric(metric)

            logger.info(f"规模 {scale}: 内存使用 {metric.memory_usage_mb:.2f}MB, 峰值 {metric.peak_memory_mb:.2f}MB")

            # 清理内存
            gc.collect()

        # 检查内存使用是否合理
        max_memory = max(m.peak_memory_mb for m in result.metrics)
        if max_memory > self.performance_baselines["memory_usage_mb"]:
            result.performance_regression = True
            result.warnings.append(f"内存使用超过基线: {max_memory:.2f}MB > {self.performance_baselines['memory_usage_mb']}MB")

        return result


def run_performance_tests():
    """运行性能测试"""
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        VoiceManagerPerformanceTest,
        WeightCalculationPerformanceTest,
        VoiceFusionPerformanceTest,
        ConcurrencyPerformanceTest,
        MemoryUsageTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def generate_performance_report(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """生成性能报告"""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "python_version": sys.version,
            "platform": sys.platform
        },
        "test_results": [],
        "summary": {
            "total_tests": len(results),
            "performance_regressions": 0,
            "total_warnings": 0
        }
    }

    for result in results:
        test_result = {
            "test_name": result.test_name,
            "summary": result.get_summary(),
            "metrics": [m.to_dict() for m in result.metrics],
            "warnings": result.warnings
        }
        report["test_results"].append(test_result)

        if result.performance_regression:
            report["summary"]["performance_regressions"] += 1
        report["summary"]["total_warnings"] += len(result.warnings)

    return report


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)
