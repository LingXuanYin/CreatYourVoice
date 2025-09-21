"""DDSP-SVC版本支持测试

这个模块测试DDSP-SVC 6.1和6.3版本的支持功能。
测试内容：
1. 版本检测功能
2. 版本切换功能
3. 统一接口功能
4. 适配器功能
5. 兼容性测试
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.version_detector import (
    DDSPSVCVersionDetector,
    DDSPSVCVersion,
    VersionInfo,
    get_ddsp_svc_version
)
from src.integrations.version_manager import (
    DDSPSVCVersionManager,
    VersionManagerConfig,
    get_version_manager
)


class TestDDSPSVCVersionDetector(unittest.TestCase):
    """测试版本检测器"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.detector = DDSPSVCVersionDetector(self.temp_dir)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_version_enum(self):
        """测试版本枚举"""
        self.assertEqual(DDSPSVCVersion.V6_1.value, "6.1")
        self.assertEqual(DDSPSVCVersion.V6_3.value, "6.3")
        self.assertEqual(DDSPSVCVersion.UNKNOWN.value, "unknown")

    def test_detector_initialization(self):
        """测试检测器初始化"""
        self.assertEqual(self.detector.ddsp_svc_path, self.temp_dir)
        self.assertIsNone(self.detector._version_cache)

    def test_detect_version_no_path(self):
        """测试检测不存在的路径"""
        non_existent_path = self.temp_dir / "non_existent"
        detector = DDSPSVCVersionDetector(non_existent_path)

        version_info = detector.detect_version()
        self.assertEqual(version_info.version, DDSPSVCVersion.UNKNOWN)

    def test_detect_version_by_file_features_61(self):
        """测试通过文件特征检测6.1版本"""
        # 创建6.1版本特征的main_reflow.py
        main_reflow_content = '''
import os
import torch

def main():
    # 6.1版本特征
    volume_extractor = Volume_Extractor(hop_size)
    result = model(
        units, f0, volume,
        return_wav=True,
        infer_step=infer_step
    )
'''
        main_reflow_path = self.temp_dir / "main_reflow.py"
        main_reflow_path.write_text(main_reflow_content)

        version_info = self.detector.detect_version()
        self.assertEqual(version_info.version, DDSPSVCVersion.V6_1)
        self.assertFalse(version_info.features.get("vocal_register_shift", True))

    def test_detect_version_by_file_features_63(self):
        """测试通过文件特征检测6.3版本"""
        # 创建6.3版本特征的main_reflow.py
        main_reflow_content = '''
import os
import torch

def main():
    # 6.3版本特征
    volume_extractor = Volume_Extractor(hop_size, win_size)
    vocal_register_shift_key = args.vocal_register_shift_key
    result = model(
        units, f0, volume,
        infer_step=infer_step
    )
    output = vocoder.infer(result, f0)
'''
        main_reflow_path = self.temp_dir / "main_reflow.py"
        main_reflow_path.write_text(main_reflow_content)

        version_info = self.detector.detect_version()
        self.assertEqual(version_info.version, DDSPSVCVersion.V6_3)
        self.assertTrue(version_info.features.get("vocal_register_shift", False))

    def test_version_config(self):
        """测试版本配置"""
        config_61 = self.detector.get_version_config(DDSPSVCVersion.V6_1)
        self.assertEqual(config_61["volume_extractor_args"], ["hop_size"])
        self.assertFalse(config_61["supports_vocal_register"])
        self.assertEqual(config_61["default_t_start"], 0.7)

        config_63 = self.detector.get_version_config(DDSPSVCVersion.V6_3)
        self.assertEqual(config_63["volume_extractor_args"], ["hop_size", "win_size"])
        self.assertTrue(config_63["supports_vocal_register"])
        self.assertEqual(config_63["default_t_start"], 0.0)

    @patch('subprocess.run')
    def test_git_detection(self, mock_run):
        """测试Git版本检测"""
        # 模拟Git命令成功
        mock_run.side_effect = [
            Mock(returncode=0, stdout="6.1\n"),  # git branch --show-current
            Mock(returncode=0, stdout="abc12345\n")  # git rev-parse HEAD
        ]

        version_info = self.detector._detect_by_git()
        self.assertEqual(version_info.version, DDSPSVCVersion.V6_1)
        self.assertEqual(version_info.branch, "6.1")
        self.assertEqual(version_info.commit_hash, "abc12345")


class TestDDSPSVCVersionManager(unittest.TestCase):
    """测试版本管理器"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        config = VersionManagerConfig(ddsp_svc_path=self.temp_dir)
        self.manager = DDSPSVCVersionManager(config)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manager_initialization(self):
        """测试管理器初始化"""
        self.assertIsNotNone(self.manager.detector)
        self.assertEqual(self.manager.detector.ddsp_svc_path, self.temp_dir)

    def test_supported_versions(self):
        """测试支持的版本列表"""
        supported = self.manager.get_supported_versions()
        self.assertIn(DDSPSVCVersion.V6_1, supported)
        self.assertIn(DDSPSVCVersion.V6_3, supported)

    def test_version_config(self):
        """测试版本配置获取"""
        config_61 = self.manager.get_version_config(DDSPSVCVersion.V6_1)
        self.assertIsInstance(config_61, dict)
        self.assertIn("volume_extractor_args", config_61)

        config_63 = self.manager.get_version_config(DDSPSVCVersion.V6_3)
        self.assertIsInstance(config_63, dict)
        self.assertIn("supports_vocal_register", config_63)

    @patch('src.integrations.version_manager.DDSPSVCv61Adapter')
    def test_get_adapter_61(self, mock_adapter_class):
        """测试获取6.1版本适配器"""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        # 设置当前版本为6.1
        self.manager._current_version_info = VersionInfo(
            version=DDSPSVCVersion.V6_1,
            branch="6.1",
            commit_hash="test",
            path=self.temp_dir,
            features={}
        )

        adapter = self.manager.get_adapter(DDSPSVCVersion.V6_1)
        self.assertEqual(adapter, mock_adapter)
        mock_adapter_class.assert_called_once()

    @patch('src.integrations.version_manager.DDSPSVCv63Adapter')
    def test_get_adapter_63(self, mock_adapter_class):
        """测试获取6.3版本适配器"""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        # 设置当前版本为6.3
        self.manager._current_version_info = VersionInfo(
            version=DDSPSVCVersion.V6_3,
            branch="6.3",
            commit_hash="test",
            path=self.temp_dir,
            features={}
        )

        adapter = self.manager.get_adapter(DDSPSVCVersion.V6_3)
        self.assertEqual(adapter, mock_adapter)
        mock_adapter_class.assert_called_once()

    def test_clear_cache(self):
        """测试清理缓存"""
        # 添加一些缓存数据
        self.manager._adapter_cache[DDSPSVCVersion.V6_1] = Mock()

        self.manager.clear_cache()
        self.assertEqual(len(self.manager._adapter_cache), 0)


class TestDDSPSVCUnified(unittest.TestCase):
    """测试统一接口"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.integrations.ddsp_svc_unified.get_version_manager')
    def test_unified_initialization_auto(self, mock_get_manager):
        """测试统一接口自动版本初始化"""
        from src.integrations.ddsp_svc_unified import DDSPSVCUnified

        # 模拟版本管理器
        mock_manager = Mock()
        mock_version_info = VersionInfo(
            version=DDSPSVCVersion.V6_3,
            branch="6.3",
            commit_hash="test",
            path=self.temp_dir,
            features={}
        )
        mock_manager.detect_and_set_version.return_value = mock_version_info
        mock_manager.get_adapter.return_value = Mock()
        mock_get_manager.return_value = mock_manager

        unified = DDSPSVCUnified(version="auto")
        self.assertEqual(unified._current_version, DDSPSVCVersion.V6_3)

    @patch('src.integrations.ddsp_svc_unified.get_version_manager')
    def test_unified_initialization_specific(self, mock_get_manager):
        """测试统一接口指定版本初始化"""
        from src.integrations.ddsp_svc_unified import DDSPSVCUnified

        # 模拟版本管理器
        mock_manager = Mock()
        mock_manager.switch_version.return_value = True
        mock_manager.get_adapter.return_value = Mock()
        mock_get_manager.return_value = mock_manager

        unified = DDSPSVCUnified(version="6.1")
        self.assertEqual(unified._current_version, DDSPSVCVersion.V6_1)

    @patch('src.integrations.ddsp_svc_unified.get_version_manager')
    def test_unified_load_model(self, mock_get_manager):
        """测试统一接口模型加载"""
        from src.integrations.ddsp_svc_unified import DDSPSVCUnified

        # 模拟版本管理器和适配器
        mock_adapter = Mock()
        mock_manager = Mock()
        mock_manager.detect_and_set_version.return_value = VersionInfo(
            version=DDSPSVCVersion.V6_3,
            branch="6.3",
            commit_hash="test",
            path=self.temp_dir,
            features={}
        )
        mock_manager.get_adapter.return_value = mock_adapter
        mock_get_manager.return_value = mock_manager

        unified = DDSPSVCUnified()
        unified.load_model("test_model.pt")

        mock_adapter.load_model.assert_called_once_with("test_model.pt")


class TestVersionCompatibility(unittest.TestCase):
    """测试版本兼容性"""

    def test_version_features_61(self):
        """测试6.1版本特性"""
        from src.utils.version_detector import get_version_config

        config = get_version_config(DDSPSVCVersion.V6_1)

        # 6.1版本特性
        self.assertEqual(config["volume_extractor_args"], ["hop_size"])
        self.assertEqual(config["mask_processing"], "padding")
        self.assertEqual(config["model_return_type"], "wav")
        self.assertFalse(config["supports_vocal_register"])
        self.assertEqual(config["default_t_start"], 0.7)

    def test_version_features_63(self):
        """测试6.3版本特性"""
        from src.utils.version_detector import get_version_config

        config = get_version_config(DDSPSVCVersion.V6_3)

        # 6.3版本特性
        self.assertEqual(config["volume_extractor_args"], ["hop_size", "win_size"])
        self.assertEqual(config["mask_processing"], "upsample")
        self.assertEqual(config["model_return_type"], "mel")
        self.assertTrue(config["supports_vocal_register"])
        self.assertEqual(config["default_t_start"], 0.0)

    def test_version_support_check(self):
        """测试版本支持检查"""
        from src.utils.version_detector import is_version_supported

        self.assertTrue(is_version_supported(DDSPSVCVersion.V6_1))
        self.assertTrue(is_version_supported(DDSPSVCVersion.V6_3))
        self.assertFalse(is_version_supported(DDSPSVCVersion.UNKNOWN))


class TestIntegrationCompatibility(unittest.TestCase):
    """测试集成兼容性"""

    @patch('src.integrations.ddsp_svc.DDSPSVCUnified')
    def test_ddsp_svc_integration_unified(self, mock_unified_class):
        """测试DDSP-SVC集成使用统一接口"""
        from src.integrations.ddsp_svc import DDSPSVCIntegration

        # 模拟统一接口
        mock_unified = Mock()
        mock_unified_class.return_value = mock_unified

        integration = DDSPSVCIntegration()
        self.assertTrue(integration._use_unified)
        self.assertEqual(integration._unified, mock_unified)

    @patch('src.integrations.ddsp_svc.DDSPSVCUnified', None)
    def test_ddsp_svc_integration_fallback(self):
        """测试DDSP-SVC集成回退到原有实现"""
        from src.integrations.ddsp_svc import DDSPSVCIntegration

        with patch('src.integrations.ddsp_svc.logger') as mock_logger:
            integration = DDSPSVCIntegration()
            self.assertFalse(integration._use_unified)


def run_version_tests():
    """运行版本测试"""
    print("开始DDSP-SVC版本支持测试...")

    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestDDSPSVCVersionDetector,
        TestDDSPSVCVersionManager,
        TestDDSPSVCUnified,
        TestVersionCompatibility,
        TestIntegrationCompatibility
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 输出结果
    print(f"\n测试完成:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_version_tests()
    sys.exit(0 if success else 1)
