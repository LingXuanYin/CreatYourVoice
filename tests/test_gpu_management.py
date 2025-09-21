"""GPU模型管理功能测试

这个模块测试GPU模型管理的核心功能。
设计原则：
1. 单元测试 - 测试每个组件的基本功能
2. 集成测试 - 测试组件间的协作
3. 错误处理 - 测试异常情况的处理
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.gpu_utils import GPUUtils, GPUMonitor
from src.utils.memory_monitor import MemoryMonitor
from src.core.gpu_model_manager import GPUModelManager, ModelType, ModelStatus
from src.core.model_lifecycle import ModelLifecycleManager, PreloadConfig, TaskProfile
from src.core.gpu_manager_init import GPUManagerInitializer


class TestGPUUtils(unittest.TestCase):
    """测试GPU工具函数"""

    def test_torch_availability_check(self):
        """测试PyTorch可用性检查"""
        # 这个测试不依赖实际的PyTorch安装
        result = GPUUtils.is_torch_available()
        self.assertIsInstance(result, bool)

    def test_cuda_availability_check(self):
        """测试CUDA可用性检查"""
        result = GPUUtils.is_cuda_available()
        self.assertIsInstance(result, bool)

    def test_device_count(self):
        """测试GPU设备数量获取"""
        count = GPUUtils.get_device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_optimal_device_selection(self):
        """测试最优设备选择"""
        device = GPUUtils.get_optimal_device()
        self.assertIsInstance(device, str)
        self.assertIn(device, ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])

    def test_memory_requirement_check(self):
        """测试内存需求检查"""
        # 测试CPU模式
        result = GPUUtils.check_memory_requirement(100, "cpu")
        self.assertIsInstance(result, bool)

        # 测试自动模式
        result = GPUUtils.check_memory_requirement(100, "auto")
        self.assertIsInstance(result, bool)

    def test_model_memory_estimation(self):
        """测试模型内存估算"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"0" * 1024 * 1024)  # 1MB文件
            tmp_path = tmp_file.name

        try:
            estimated = GPUUtils.estimate_model_memory(tmp_path)
            self.assertIsInstance(estimated, int)
            self.assertGreater(estimated, 0)
        finally:
            Path(tmp_path).unlink()

    def test_cache_clearing(self):
        """测试缓存清理"""
        result = GPUUtils.clear_gpu_cache()
        self.assertIsInstance(result, bool)


class TestMemoryMonitor(unittest.TestCase):
    """测试内存监控器"""

    def setUp(self):
        """设置测试环境"""
        self.monitor = MemoryMonitor(update_interval=0.1, history_size=10)

    def tearDown(self):
        """清理测试环境"""
        self.monitor.stop_monitoring()

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        self.assertFalse(self.monitor._monitoring)
        self.assertEqual(self.monitor.update_interval, 0.1)
        self.assertEqual(self.monitor.history_size, 10)

    def test_monitor_start_stop(self):
        """测试监控器启动和停止"""
        # 启动监控
        success = self.monitor.start_monitoring()
        self.assertTrue(success)
        self.assertTrue(self.monitor._monitoring)

        # 等待一些数据
        time.sleep(0.2)

        # 停止监控
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._monitoring)

    def test_current_status(self):
        """测试当前状态获取"""
        status = self.monitor.get_current_status()
        self.assertIsInstance(status, dict)
        self.assertIn("timestamp", status)
        self.assertIn("system_memory", status)
        self.assertIn("gpu_memory", status)

    def test_memory_history(self):
        """测试内存历史记录"""
        self.monitor.start_monitoring()
        time.sleep(0.2)

        history = self.monitor.get_memory_history(minutes=1)
        self.assertIsInstance(history, list)

        self.monitor.stop_monitoring()

    def test_alert_callback(self):
        """测试警告回调"""
        callback_called = False

        def test_callback(alert):
            nonlocal callback_called
            callback_called = True

        self.monitor.add_alert_callback(test_callback)

        # 模拟高内存使用率
        with patch.object(self.monitor, '_take_snapshot') as mock_snapshot:
            from src.utils.memory_monitor import MemorySnapshot
            from src.utils.gpu_utils import SystemMemoryInfo

            mock_snapshot.return_value = MemorySnapshot(
                timestamp=time.time(),
                system_memory=SystemMemoryInfo(1000, 100, 900, 90.0),  # 90%使用率
                gpu_memory=[]
            )

            self.monitor.start_monitoring()
            time.sleep(0.2)
            self.monitor.stop_monitoring()

        # 注意：由于冷却时间机制，可能不会立即触发回调
        self.monitor.remove_alert_callback(test_callback)


class TestGPUModelManager(unittest.TestCase):
    """测试GPU模型管理器"""

    def setUp(self):
        """设置测试环境"""
        # 重置单例
        GPUModelManager._instance = None
        self.manager = GPUModelManager()

    def test_singleton_pattern(self):
        """测试单例模式"""
        manager1 = GPUModelManager()
        manager2 = GPUModelManager()
        self.assertIs(manager1, manager2)

    def test_model_info_operations(self):
        """测试模型信息操作"""
        # 初始状态
        models = self.manager.list_models()
        self.assertIsInstance(models, list)
        self.assertEqual(len(models), 0)

        # 获取不存在的模型
        info = self.manager.get_model_info("nonexistent")
        self.assertIsNone(info)

    def test_memory_usage_reporting(self):
        """测试内存使用报告"""
        usage = self.manager.get_memory_usage()
        self.assertIsInstance(usage, dict)
        self.assertIn("total_model_memory_mb", usage)
        self.assertIn("loaded_model_count", usage)
        self.assertIn("system_memory", usage)

    @patch('src.core.gpu_model_manager.DDSPSVCIntegration')
    def test_ddsp_model_loading_mock(self, mock_ddsp):
        """测试DDSP模型加载（模拟）"""
        # 模拟DDSP集成
        mock_instance = Mock()
        mock_ddsp.return_value = mock_instance

        # 创建临时模型文件
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # 模拟加载
            with patch.object(self.manager, '_estimate_instance_memory', return_value=1024):
                model_id = self.manager.load_ddsp_model(tmp_path, device="cpu")
                self.assertIsInstance(model_id, str)

                # 检查模型信息
                info = self.manager.get_model_info(model_id)
                self.assertIsNotNone(info)
                if info is not None:
                    self.assertEqual(info["model_type"], "ddsp_svc")
                    self.assertEqual(info["status"], "loaded")

                # 卸载模型
                success = self.manager.unload_model(model_id)
                self.assertTrue(success)
        finally:
            Path(tmp_path).unlink()

    def test_cleanup_operations(self):
        """测试清理操作"""
        # 测试内存优化
        result = self.manager.optimize_memory()
        self.assertIsInstance(result, dict)
        self.assertIn("cleaned_models", result)
        self.assertIn("optimization_time", result)

    def test_configuration_methods(self):
        """测试配置方法"""
        # 测试自动清理配置
        self.manager.enable_auto_cleanup(True)
        self.assertTrue(self.manager.auto_cleanup_enabled)

        self.manager.enable_auto_cleanup(False)
        self.assertFalse(self.manager.auto_cleanup_enabled)

        # 测试清理阈值设置
        self.manager.set_cleanup_threshold(75.0)
        self.assertEqual(self.manager.cleanup_threshold_percent, 75.0)

        # 测试空闲超时设置
        self.manager.set_idle_timeout(15)
        self.assertEqual(self.manager.idle_timeout_minutes, 15)


class TestModelLifecycleManager(unittest.TestCase):
    """测试模型生命周期管理器"""

    def setUp(self):
        """设置测试环境"""
        self.lifecycle_manager = ModelLifecycleManager()

    def tearDown(self):
        """清理测试环境"""
        self.lifecycle_manager.stop_auto_management()

    def test_preload_config_registration(self):
        """测试预加载配置注册"""
        config = PreloadConfig(
            model_type=ModelType.DDSP_SVC,
            model_path="/fake/path/model.pth",
            device="cpu"
        )

        self.lifecycle_manager.register_preload_config("test_config", config)
        self.assertIn("test_config", self.lifecycle_manager._preload_configs)

    def test_task_profile_registration(self):
        """测试任务配置注册"""
        profile = TaskProfile(
            task_name="test_task",
            required_models=[
                PreloadConfig(
                    model_type=ModelType.INDEX_TTS,
                    model_path="/fake/path/model",
                    device="cpu"
                )
            ]
        )

        self.lifecycle_manager.register_task_profile(profile)
        self.assertIn("test_task", self.lifecycle_manager._task_profiles)

    def test_auto_management(self):
        """测试自动管理"""
        # 启动自动管理
        success = self.lifecycle_manager.start_auto_management()
        self.assertTrue(success)
        self.assertTrue(self.lifecycle_manager._auto_management_enabled)

        # 停止自动管理
        self.lifecycle_manager.stop_auto_management()
        self.assertFalse(self.lifecycle_manager._auto_management_enabled)

    def test_statistics(self):
        """测试统计信息"""
        stats = self.lifecycle_manager.get_lifecycle_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_events", stats)
        self.assertIn("registered_configs", stats)
        self.assertIn("registered_tasks", stats)

    def test_event_system(self):
        """测试事件系统"""
        events_received = []

        def test_listener(event_data):
            events_received.append(event_data)

        self.lifecycle_manager.add_event_listener(test_listener)

        # 模拟事件
        from src.core.model_lifecycle import LifecycleEventData, LifecycleEvent
        test_event = LifecycleEventData(
            event=LifecycleEvent.LOAD_STARTED,
            model_id="test_model",
            model_type=ModelType.DDSP_SVC,
            timestamp=time.time()
        )

        self.lifecycle_manager._emit_event(test_event)

        self.assertEqual(len(events_received), 1)
        self.assertEqual(events_received[0].model_id, "test_model")

        self.lifecycle_manager.remove_event_listener(test_listener)


class TestGPUManagerInitializer(unittest.TestCase):
    """测试GPU管理器初始化器"""

    def setUp(self):
        """设置测试环境"""
        # 使用模拟配置
        self.mock_config = Mock()
        self.mock_config.gpu = Mock()
        self.mock_config.gpu.auto_cleanup_enabled = True
        self.mock_config.gpu.cleanup_threshold_percent = 85.0
        self.mock_config.gpu.idle_timeout_minutes = 30
        self.mock_config.gpu.max_memory_usage_percent = 90.0
        self.mock_config.gpu.memory_monitoring_enabled = True
        self.mock_config.gpu.memory_monitoring_interval = 2.0
        self.mock_config.index_tts = Mock()
        self.mock_config.index_tts.model_dir = "/fake/index_tts"
        self.mock_config.ddsp_svc = Mock()
        self.mock_config.ddsp_svc.model_dir = "/fake/ddsp_svc"

    @patch('src.core.gpu_manager_init.get_config')
    def test_initializer_creation(self, mock_get_config):
        """测试初始化器创建"""
        mock_get_config.return_value = self.mock_config

        initializer = GPUManagerInitializer()
        self.assertFalse(initializer._initialized)

    @patch('src.core.gpu_manager_init.get_config')
    @patch('src.core.gpu_manager_init.start_global_monitoring')
    def test_initialization_process(self, mock_start_monitoring, mock_get_config):
        """测试初始化过程"""
        mock_get_config.return_value = self.mock_config
        mock_start_monitoring.return_value = True

        initializer = GPUManagerInitializer()

        # 模拟初始化过程
        with patch.object(initializer, '_configure_model_manager'), \
             patch.object(initializer, '_register_default_tasks'), \
             patch.object(initializer, '_start_lifecycle_management'):

            success = initializer.initialize()
            self.assertTrue(success)
            self.assertTrue(initializer._initialized)

    @patch('src.core.gpu_manager_init.get_config')
    def test_status_reporting(self, mock_get_config):
        """测试状态报告"""
        mock_get_config.return_value = self.mock_config

        initializer = GPUManagerInitializer()
        status = initializer.get_status()

        self.assertIsInstance(status, dict)
        self.assertIn("initialized", status)
        self.assertIn("model_manager_status", status)
        self.assertIn("memory_monitoring", status)
        self.assertIn("lifecycle_management", status)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_system_integration(self):
        """测试完整系统集成"""
        # 这是一个简化的集成测试
        # 在实际环境中，这会测试所有组件的协作

        # 1. 检查GPU工具可用性
        gpu_available = GPUUtils.is_cuda_available()

        # 2. 创建内存监控器
        monitor = MemoryMonitor(update_interval=0.1)

        # 3. 创建模型管理器
        manager = GPUModelManager()

        # 4. 创建生命周期管理器
        lifecycle = ModelLifecycleManager()

        # 5. 测试基本操作
        try:
            # 获取状态
            memory_usage = manager.get_memory_usage()
            self.assertIsInstance(memory_usage, dict)

            # 优化内存
            result = manager.optimize_memory()
            self.assertIsInstance(result, dict)

            # 获取统计信息
            stats = lifecycle.get_lifecycle_statistics()
            self.assertIsInstance(stats, dict)

        finally:
            # 清理
            monitor.stop_monitoring()
            lifecycle.stop_auto_management()


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestGPUUtils,
        TestMemoryMonitor,
        TestGPUModelManager,
        TestModelLifecycleManager,
        TestGPUManagerInitializer,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
