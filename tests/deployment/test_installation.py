"""安装测试

这个模块测试系统的安装和依赖配置。
设计原则：
1. 依赖检查 - 验证所有必需的依赖包是否正确安装
2. 环境验证 - 检查运行环境是否满足要求
3. 模型验证 - 验证模型文件是否存在和可用
"""

import unittest
import sys
import subprocess
import importlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import platform

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class InstallationTestCase(unittest.TestCase):
    """安装测试基类"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="WARNING", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.required_packages = self._get_required_packages()
        self.optional_packages = self._get_optional_packages()

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _get_required_packages(self) -> List[str]:
        """获取必需的包列表"""
        return [
            "torch",
            "torchaudio",
            "numpy",
            "scipy",
            "librosa",
            "soundfile",
            "pydub",
            "gradio",
            "pyyaml",
            "psutil"
        ]

    def _get_optional_packages(self) -> List[str]:
        """获取可选的包列表"""
        return [
            "matplotlib",
            "pandas",
            "requests",
            "pillow"
        ]


class DependencyTest(InstallationTestCase):
    """依赖测试"""

    def test_python_version(self):
        """测试Python版本"""
        logger.info("测试Python版本")

        version = sys.version_info

        # 要求Python 3.8+
        self.assertGreaterEqual(version.major, 3)
        self.assertGreaterEqual(version.minor, 8)

        logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    def test_required_packages(self):
        """测试必需包"""
        logger.info("测试必需包安装")

        missing_packages = []
        package_versions = {}

        for package in self.required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                package_versions[package] = version
                logger.info(f"✓ {package}: {version}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"✗ {package}: 未安装")

        if missing_packages:
            self.fail(f"缺少必需包: {', '.join(missing_packages)}")

        # 验证特定包的版本要求
        self._validate_package_versions(package_versions)

    def test_optional_packages(self):
        """测试可选包"""
        logger.info("测试可选包安装")

        available_packages = []
        missing_packages = []

        for package in self.optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                available_packages.append((package, version))
                logger.info(f"✓ {package}: {version}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"○ {package}: 未安装（可选）")

        # 可选包不影响测试结果，只记录
        logger.info(f"可用可选包: {len(available_packages)}/{len(self.optional_packages)}")

    def _validate_package_versions(self, package_versions: Dict[str, str]):
        """验证包版本要求"""
        version_requirements = {
            "torch": "1.9.0",
            "numpy": "1.19.0",
            "gradio": "3.0.0"
        }

        for package, min_version in version_requirements.items():
            if package in package_versions:
                current_version = package_versions[package]
                if current_version != 'unknown':
                    # 简单的版本比较（实际应该使用packaging.version）
                    try:
                        current_parts = [int(x) for x in current_version.split('.')]
                        min_parts = [int(x) for x in min_version.split('.')]

                        for i in range(min(len(current_parts), len(min_parts))):
                            if current_parts[i] > min_parts[i]:
                                break
                            elif current_parts[i] < min_parts[i]:
                                logger.warning(f"{package} 版本可能过低: {current_version} < {min_version}")
                                break
                    except ValueError:
                        logger.warning(f"无法解析 {package} 版本: {current_version}")

    def test_torch_cuda_availability(self):
        """测试CUDA可用性"""
        logger.info("测试CUDA可用性")

        try:
            import torch

            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0

            logger.info(f"CUDA可用: {cuda_available}")
            if cuda_available:
                logger.info(f"CUDA设备数量: {device_count}")
                for i in range(device_count):
                    device_name = torch.cuda.get_device_name(i)
                    logger.info(f"设备 {i}: {device_name}")

            # CUDA不是必需的，但记录状态
            self.assertIsInstance(cuda_available, bool)

        except ImportError:
            self.fail("PyTorch未安装")


class EnvironmentTest(InstallationTestCase):
    """环境测试"""

    def test_system_info(self):
        """测试系统信息"""
        logger.info("测试系统信息")

        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation()
        }

        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
            self.assertIsNotNone(value)

    def test_memory_availability(self):
        """测试内存可用性"""
        logger.info("测试内存可用性")

        try:
            import psutil

            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            logger.info(f"总内存: {total_gb:.2f} GB")
            logger.info(f"可用内存: {available_gb:.2f} GB")
            logger.info(f"内存使用率: {memory.percent:.1f}%")

            # 建议至少4GB可用内存
            self.assertGreater(available_gb, 2.0, "可用内存不足，建议至少4GB")

        except ImportError:
            logger.warning("psutil未安装，跳过内存检查")

    def test_disk_space(self):
        """测试磁盘空间"""
        logger.info("测试磁盘空间")

        try:
            import psutil

            disk_usage = psutil.disk_usage('.')
            total_gb = disk_usage.total / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            logger.info(f"总磁盘空间: {total_gb:.2f} GB")
            logger.info(f"可用磁盘空间: {free_gb:.2f} GB")
            logger.info(f"磁盘使用率: {used_percent:.1f}%")

            # 建议至少10GB可用空间
            self.assertGreater(free_gb, 5.0, "可用磁盘空间不足，建议至少10GB")

        except ImportError:
            logger.warning("psutil未安装，跳过磁盘检查")


class ProjectStructureTest(InstallationTestCase):
    """项目结构测试"""

    def test_project_directories(self):
        """测试项目目录结构"""
        logger.info("测试项目目录结构")

        required_dirs = [
            "src",
            "src/core",
            "src/integrations",
            "src/utils",
            "src/webui",
            "src/data",
            "tests"
        ]

        missing_dirs = []
        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
            else:
                logger.info(f"✓ {dir_path}")

        if missing_dirs:
            self.fail(f"缺少必需目录: {', '.join(missing_dirs)}")

    def test_core_modules(self):
        """测试核心模块"""
        logger.info("测试核心模块")

        core_modules = [
            "src.core.models",
            "src.core.voice_manager",
            "src.core.weight_calculator",
            "src.core.voice_fusion",
            "src.core.voice_inheritance",
            "src.utils.config",
            "src.utils.logging_config"
        ]

        missing_modules = []
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                logger.info(f"✓ {module_name}")
            except ImportError as e:
                missing_modules.append(module_name)
                logger.error(f"✗ {module_name}: {e}")

        if missing_modules:
            self.fail(f"无法导入核心模块: {', '.join(missing_modules)}")

    def test_config_files(self):
        """测试配置文件"""
        logger.info("测试配置文件")

        config_files = [
            "pyproject.toml",
            "src/data/presets/voice_tags.yaml",
            "src/data/presets/speakers.yaml"
        ]

        missing_files = []
        for file_path in config_files:
            full_path = PROJECT_ROOT / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                logger.info(f"✓ {file_path}")

        if missing_files:
            self.fail(f"缺少配置文件: {', '.join(missing_files)}")


class ModelTest(InstallationTestCase):
    """模型测试"""

    def test_ddsp_svc_models(self):
        """测试DDSP-SVC模型"""
        logger.info("测试DDSP-SVC模型")

        ddsp_dir = PROJECT_ROOT / "DDSP-SVC"

        if not ddsp_dir.exists():
            logger.warning("DDSP-SVC目录不存在，跳过模型检查")
            return

        # 检查关键文件
        key_files = [
            "requirements.txt",
            "reflow/reflow.py",
            "reflow/vocoder.py"
        ]

        missing_files = []
        for file_path in key_files:
            full_path = ddsp_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                logger.info(f"✓ DDSP-SVC/{file_path}")

        if missing_files:
            logger.warning(f"DDSP-SVC缺少文件: {', '.join(missing_files)}")

    def test_index_tts_models(self):
        """测试IndexTTS模型"""
        logger.info("测试IndexTTS模型")

        index_dir = PROJECT_ROOT / "index-tts"

        if not index_dir.exists():
            logger.warning("index-tts目录不存在，跳过模型检查")
            return

        # 检查关键文件
        key_files = [
            "pyproject.toml",
            "indextts/__init__.py",
            "indextts/infer.py"
        ]

        missing_files = []
        for file_path in key_files:
            full_path = index_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                logger.info(f"✓ index-tts/{file_path}")

        if missing_files:
            logger.warning(f"index-tts缺少文件: {', '.join(missing_files)}")


class IntegrationTest(InstallationTestCase):
    """集成测试"""

    def test_basic_imports(self):
        """测试基本导入"""
        logger.info("测试基本导入")

        try:
            from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
            from src.core.voice_manager import VoiceManager
            from src.utils.config import get_config

            logger.info("✓ 基本导入成功")

        except ImportError as e:
            self.fail(f"基本导入失败: {e}")

    def test_config_loading(self):
        """测试配置加载"""
        logger.info("测试配置加载")

        try:
            from src.utils.config import get_config

            config = get_config()
            self.assertIsNotNone(config)

            # 检查基本配置项
            self.assertTrue(hasattr(config, 'system'))
            self.assertTrue(hasattr(config, 'ddsp_svc'))
            self.assertTrue(hasattr(config, 'index_tts'))

            logger.info("✓ 配置加载成功")

        except Exception as e:
            self.fail(f"配置加载失败: {e}")

    def test_voice_manager_creation(self):
        """测试音色管理器创建"""
        logger.info("测试音色管理器创建")

        try:
            from src.core.voice_manager import VoiceManager

            # 使用临时目录创建音色管理器
            voice_manager = VoiceManager(self.temp_dir / "voices")
            self.assertIsNotNone(voice_manager)

            logger.info("✓ 音色管理器创建成功")

        except Exception as e:
            self.fail(f"音色管理器创建失败: {e}")


def run_installation_tests():
    """运行安装测试"""
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        DependencyTest,
        EnvironmentTest,
        ProjectStructureTest,
        ModelTest,
        IntegrationTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def generate_installation_report() -> Dict[str, Any]:
    """生成安装报告"""
    report = {
        "timestamp": "2024-01-01 00:00:00",  # 实际应该使用当前时间
        "system_info": {},
        "dependencies": {},
        "project_structure": {},
        "recommendations": []
    }

    try:
        # 系统信息
        report["system_info"] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "architecture": platform.machine()
        }

        # 依赖检查
        test_case = DependencyTest()
        test_case.setUp()

        dependencies = {}
        for package in test_case.required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                dependencies[package] = {"installed": True, "version": version}
            except ImportError:
                dependencies[package] = {"installed": False, "version": None}

        report["dependencies"] = dependencies

        # 项目结构检查
        required_dirs = ["src", "src/core", "src/integrations", "src/utils", "tests"]
        structure_status = {}
        for dir_path in required_dirs:
            full_path = PROJECT_ROOT / dir_path
            structure_status[dir_path] = full_path.exists()

        report["project_structure"] = structure_status

        # 生成建议
        recommendations = []

        # 检查缺失的依赖
        missing_deps = [pkg for pkg, info in dependencies.items() if not info["installed"]]
        if missing_deps:
            recommendations.append({
                "type": "dependency",
                "priority": "high",
                "message": f"安装缺失的依赖: {', '.join(missing_deps)}",
                "command": f"pip install {' '.join(missing_deps)}"
            })

        # 检查缺失的目录
        missing_dirs = [dir_path for dir_path, exists in structure_status.items() if not exists]
        if missing_dirs:
            recommendations.append({
                "type": "structure",
                "priority": "high",
                "message": f"创建缺失的目录: {', '.join(missing_dirs)}"
            })

        report["recommendations"] = recommendations

    except Exception as e:
        report["error"] = str(e)

    return report


if __name__ == "__main__":
    success = run_installation_tests()

    # 生成报告
    report = generate_installation_report()
    report_file = PROJECT_ROOT / "installation_report.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"安装报告已保存到: {report_file}")

    sys.exit(0 if success else 1)
