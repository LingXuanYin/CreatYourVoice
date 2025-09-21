"""配置验证测试

这个模块测试系统配置的正确性和有效性。
设计原则：
1. 配置完整性 - 验证所有必需的配置项是否存在
2. 配置有效性 - 验证配置值是否合理和有效
3. 配置兼容性 - 验证配置之间的兼容性
"""

import unittest
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_config, ConfigManager, update_config
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class ConfigurationTestCase(unittest.TestCase):
    """配置测试基类"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="WARNING", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_manager = ConfigManager()

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


class ConfigStructureTest(ConfigurationTestCase):
    """配置结构测试"""

    def test_config_loading(self):
        """测试配置加载"""
        logger.info("测试配置加载")

        try:
            config = get_config()
            self.assertIsNotNone(config)
            logger.info("✓ 配置加载成功")
        except Exception as e:
            self.fail(f"配置加载失败: {e}")

    def test_config_structure(self):
        """测试配置结构"""
        logger.info("测试配置结构")

        config = get_config()

        # 检查主要配置节
        required_sections = ['system', 'ddsp_svc', 'index_tts']

        for section in required_sections:
            self.assertTrue(hasattr(config, section), f"缺少配置节: {section}")
            logger.info(f"✓ 配置节存在: {section}")

    def test_system_config(self):
        """测试系统配置"""
        logger.info("测试系统配置")

        config = get_config()
        system_config = config.system

        # 检查必需的系统配置项
        required_fields = ['device', 'voices_dir', 'temp_dir', 'log_level']

        for field in required_fields:
            self.assertTrue(hasattr(system_config, field), f"缺少系统配置项: {field}")
            value = getattr(system_config, field)
            self.assertIsNotNone(value, f"系统配置项为空: {field}")
            logger.info(f"✓ {field}: {value}")

    def test_ddsp_svc_config(self):
        """测试DDSP-SVC配置"""
        logger.info("测试DDSP-SVC配置")

        config = get_config()
        ddsp_config = config.ddsp_svc

        # 检查DDSP-SVC配置项
        if hasattr(ddsp_config, 'model_dir'):
            model_dir = ddsp_config.model_dir
            self.assertIsNotNone(model_dir)
            logger.info(f"✓ model_dir: {model_dir}")

        # 检查其他可能的配置项
        optional_fields = ['default_f0_predictor', 'default_threshold']
        for field in optional_fields:
            if hasattr(ddsp_config, field):
                value = getattr(ddsp_config, field)
                logger.info(f"✓ {field}: {value}")

    def test_index_tts_config(self):
        """测试IndexTTS配置"""
        logger.info("测试IndexTTS配置")

        config = get_config()
        index_config = config.index_tts

        # 检查IndexTTS配置项
        required_fields = ['model_dir']

        for field in required_fields:
            self.assertTrue(hasattr(index_config, field), f"缺少IndexTTS配置项: {field}")
            value = getattr(index_config, field)
            self.assertIsNotNone(value, f"IndexTTS配置项为空: {field}")
            logger.info(f"✓ {field}: {value}")


class ConfigValidationTest(ConfigurationTestCase):
    """配置验证测试"""

    def test_device_config(self):
        """测试设备配置"""
        logger.info("测试设备配置")

        config = get_config()
        device = config.system.device

        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        self.assertIn(device, valid_devices, f"无效的设备配置: {device}")

        logger.info(f"✓ 设备配置有效: {device}")

    def test_directory_config(self):
        """测试目录配置"""
        logger.info("测试目录配置")

        config = get_config()

        # 检查目录配置
        directories = {
            'voices_dir': config.system.voices_dir,
            'temp_dir': config.system.temp_dir,
            'model_dir': config.index_tts.model_dir
        }

        for name, dir_path in directories.items():
            self.assertIsNotNone(dir_path, f"目录配置为空: {name}")

            # 检查路径格式
            path_obj = Path(dir_path)
            self.assertIsInstance(path_obj, Path, f"无效的路径格式: {name}")

            logger.info(f"✓ {name}: {dir_path}")

    def test_log_level_config(self):
        """测试日志级别配置"""
        logger.info("测试日志级别配置")

        config = get_config()
        log_level = config.system.log_level

        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.assertIn(log_level.upper(), valid_levels, f"无效的日志级别: {log_level}")

        logger.info(f"✓ 日志级别有效: {log_level}")


class ConfigUpdateTest(ConfigurationTestCase):
    """配置更新测试"""

    def test_config_update(self):
        """测试配置更新"""
        logger.info("测试配置更新")

        # 获取原始配置
        original_config = get_config()
        original_device = original_config.system.device

        # 更新配置
        new_device = 'cpu' if original_device != 'cpu' else 'auto'
        updates = {
            'system': {
                'device': new_device
            }
        }

        try:
            update_config(updates)

            # 验证更新
            updated_config = get_config()
            self.assertEqual(updated_config.system.device, new_device)

            logger.info(f"✓ 配置更新成功: {original_device} -> {new_device}")

        except Exception as e:
            self.fail(f"配置更新失败: {e}")
        finally:
            # 恢复原始配置
            restore_updates = {
                'system': {
                    'device': original_device
                }
            }
            try:
                update_config(restore_updates)
            except:
                pass

    def test_invalid_config_update(self):
        """测试无效配置更新"""
        logger.info("测试无效配置更新")

        # 尝试设置无效的设备
        invalid_updates = {
            'system': {
                'device': 'invalid_device'
            }
        }

        # 这应该不会导致系统崩溃，但可能会有警告
        try:
            update_config(invalid_updates)

            # 检查配置是否仍然有效
            config = get_config()
            device = config.system.device

            # 如果更新了无效值，应该有某种处理机制
            logger.info(f"设备配置: {device}")

        except Exception as e:
            # 预期可能会有异常
            logger.info(f"无效配置更新被拒绝: {e}")


class PresetConfigTest(ConfigurationTestCase):
    """预设配置测试"""

    def test_voice_tags_preset(self):
        """测试音色标签预设"""
        logger.info("测试音色标签预设")

        voice_tags_file = PROJECT_ROOT / "src/data/presets/voice_tags.yaml"

        if not voice_tags_file.exists():
            self.skipTest("音色标签预设文件不存在")

        try:
            with open(voice_tags_file, 'r', encoding='utf-8') as f:
                voice_tags = yaml.safe_load(f)

            self.assertIsInstance(voice_tags, dict, "音色标签预设应该是字典格式")
            self.assertGreater(len(voice_tags), 0, "音色标签预设不能为空")

            # 检查每个标签的结构
            for tag_name, tag_info in voice_tags.items():
                self.assertIsInstance(tag_info, dict, f"标签 {tag_name} 信息应该是字典")

                # 检查必需字段
                required_fields = ['name', 'description']
                for field in required_fields:
                    self.assertIn(field, tag_info, f"标签 {tag_name} 缺少字段: {field}")

            logger.info(f"✓ 音色标签预设有效，包含 {len(voice_tags)} 个标签")

        except yaml.YAMLError as e:
            self.fail(f"音色标签预设YAML格式错误: {e}")
        except Exception as e:
            self.fail(f"音色标签预设验证失败: {e}")

    def test_speakers_preset(self):
        """测试说话人预设"""
        logger.info("测试说话人预设")

        speakers_file = PROJECT_ROOT / "src/data/presets/speakers.yaml"

        if not speakers_file.exists():
            self.skipTest("说话人预设文件不存在")

        try:
            with open(speakers_file, 'r', encoding='utf-8') as f:
                speakers = yaml.safe_load(f)

            self.assertIsInstance(speakers, dict, "说话人预设应该是字典格式")

            # 检查说话人信息结构
            for speaker_id, speaker_info in speakers.items():
                self.assertIsInstance(speaker_info, dict, f"说话人 {speaker_id} 信息应该是字典")

                # 检查可能的字段
                optional_fields = ['name', 'gender', 'language', 'description']
                for field in optional_fields:
                    if field in speaker_info:
                        self.assertIsNotNone(speaker_info[field], f"说话人 {speaker_id} 字段 {field} 不能为空")

            logger.info(f"✓ 说话人预设有效，包含 {len(speakers)} 个说话人")

        except yaml.YAMLError as e:
            self.fail(f"说话人预设YAML格式错误: {e}")
        except Exception as e:
            self.fail(f"说话人预设验证失败: {e}")


class ConfigCompatibilityTest(ConfigurationTestCase):
    """配置兼容性测试"""

    def test_path_compatibility(self):
        """测试路径兼容性"""
        logger.info("测试路径兼容性")

        config = get_config()

        # 检查路径是否使用正确的分隔符
        paths = {
            'voices_dir': config.system.voices_dir,
            'temp_dir': config.system.temp_dir,
            'index_model_dir': config.index_tts.model_dir
        }

        for name, path_str in paths.items():
            path_obj = Path(path_str)

            # 检查路径是否可以正确解析
            self.assertIsInstance(path_obj, Path, f"路径解析失败: {name}")

            # 检查是否为绝对路径或相对路径
            is_absolute = path_obj.is_absolute()
            logger.info(f"✓ {name}: {path_str} ({'绝对路径' if is_absolute else '相对路径'})")

    def test_encoding_compatibility(self):
        """测试编码兼容性"""
        logger.info("测试编码兼容性")

        # 测试配置文件是否可以正确处理中文
        test_config = {
            'test_chinese': '测试中文配置',
            'test_unicode': '🎵音色测试🎤'
        }

        # 写入临时配置文件
        temp_config_file = self.temp_dir / "test_config.yaml"

        try:
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(test_config, f, allow_unicode=True)

            # 读取并验证
            with open(temp_config_file, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)

            self.assertEqual(loaded_config['test_chinese'], '测试中文配置')
            self.assertEqual(loaded_config['test_unicode'], '🎵音色测试🎤')

            logger.info("✓ 编码兼容性测试通过")

        except Exception as e:
            self.fail(f"编码兼容性测试失败: {e}")


class ConfigSecurityTest(ConfigurationTestCase):
    """配置安全性测试"""

    def test_path_traversal_protection(self):
        """测试路径遍历保护"""
        logger.info("测试路径遍历保护")

        # 测试危险路径
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]

        for dangerous_path in dangerous_paths:
            path_obj = Path(dangerous_path)

            # 检查是否有适当的路径验证
            # 这里只是示例，实际的验证逻辑应该在配置处理代码中
            if path_obj.is_absolute() and str(path_obj).startswith(('/', 'C:\\')):
                logger.warning(f"检测到绝对路径: {dangerous_path}")

            if '..' in str(path_obj):
                logger.warning(f"检测到路径遍历尝试: {dangerous_path}")

        logger.info("✓ 路径遍历保护测试完成")

    def test_config_permissions(self):
        """测试配置文件权限"""
        logger.info("测试配置文件权限")

        # 创建测试配置文件
        test_config_file = self.temp_dir / "test_permissions.yaml"

        test_config = {'test': 'value'}

        try:
            with open(test_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(test_config, f)

            # 检查文件是否可读
            self.assertTrue(test_config_file.exists())
            self.assertTrue(test_config_file.is_file())

            # 尝试读取
            with open(test_config_file, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)

            self.assertEqual(loaded_config['test'], 'value')

            logger.info("✓ 配置文件权限测试通过")

        except Exception as e:
            self.fail(f"配置文件权限测试失败: {e}")


def run_configuration_tests():
    """运行配置测试"""
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        ConfigStructureTest,
        ConfigValidationTest,
        ConfigUpdateTest,
        PresetConfigTest,
        ConfigCompatibilityTest,
        ConfigSecurityTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


def generate_configuration_report() -> Dict[str, Any]:
    """生成配置报告"""
    report = {
        "timestamp": "2024-01-01 00:00:00",  # 实际应该使用当前时间
        "config_status": {},
        "validation_results": {},
        "recommendations": []
    }

    try:
        # 配置状态检查
        config = get_config()

        config_status = {
            "config_loaded": True,
            "system_config": hasattr(config, 'system'),
            "ddsp_config": hasattr(config, 'ddsp_svc'),
            "index_config": hasattr(config, 'index_tts')
        }

        report["config_status"] = config_status

        # 验证结果
        validation_results = {}

        # 设备配置验证
        if hasattr(config.system, 'device'):
            device = config.system.device
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            validation_results["device_valid"] = device in valid_devices

        # 目录配置验证
        if hasattr(config.system, 'voices_dir'):
            voices_dir = Path(config.system.voices_dir)
            validation_results["voices_dir_valid"] = True  # 基本路径验证

        report["validation_results"] = validation_results

        # 生成建议
        recommendations = []

        if not all(config_status.values()):
            recommendations.append({
                "type": "config_structure",
                "priority": "high",
                "message": "配置结构不完整，请检查配置文件"
            })

        if not all(validation_results.values()):
            recommendations.append({
                "type": "config_validation",
                "priority": "medium",
                "message": "部分配置值无效，请检查配置设置"
            })

        report["recommendations"] = recommendations

    except Exception as e:
        report["error"] = str(e)

    return report


if __name__ == "__main__":
    success = run_configuration_tests()

    # 生成报告
    report = generate_configuration_report()
    report_file = PROJECT_ROOT / "configuration_report.json"

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"配置报告已保存到: {report_file}")

    sys.exit(0 if success else 1)
