"""核心功能测试

这个模块包含核心组件的基本测试用例。
设计原则：
1. 简单直接 - 测试核心功能，不过度复杂化
2. 独立运行 - 每个测试都能独立运行
3. 快速反馈 - 快速发现基本问题
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.weight_calculator import WeightCalculator
from src.core.voice_manager import VoiceManager
from src.utils.config import Config, ConfigManager


class TestModels(unittest.TestCase):
    """测试数据模型"""

    def test_ddsp_svc_config(self):
        """测试DDSP-SVC配置"""
        config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1
        )

        self.assertEqual(config.model_path, "test_model.pth")
        self.assertEqual(config.speaker_id, 1)
        self.assertEqual(config.f0_predictor, "rmvpe")  # 默认值

    def test_index_tts_config(self):
        """测试IndexTTS配置"""
        config = IndexTTSConfig(
            model_path="test_model",
            config_path="test_config.yaml"
        )

        self.assertEqual(config.model_path, "test_model")
        self.assertEqual(config.speaker_name, "default")  # 默认值
        self.assertGreaterEqual(config.emotion_strength, 0.0)
        self.assertLessEqual(config.emotion_strength, 2.0)

    def test_voice_config(self):
        """测试音色配置"""
        ddsp_config = DDSPSVCConfig(
            model_path="ddsp_model.pth",
            config_path="ddsp_config.yaml"
        )

        index_tts_config = IndexTTSConfig(
            model_path="index_model",
            config_path="index_config.yaml"
        )

        voice_config = VoiceConfig(
            name="测试音色",
            ddsp_config=ddsp_config,
            index_tts_config=index_tts_config
        )

        self.assertEqual(voice_config.name, "测试音色")
        self.assertIsNotNone(voice_config.voice_id)
        self.assertIsNotNone(voice_config.created_at)

        # 测试序列化
        data = voice_config.to_dict()
        self.assertIn("voice_id", data)
        self.assertIn("name", data)
        self.assertIn("ddsp_config", data)
        self.assertIn("index_tts_config", data)

        # 测试反序列化
        restored_config = VoiceConfig.from_dict(data)
        self.assertEqual(restored_config.name, voice_config.name)
        self.assertEqual(restored_config.voice_id, voice_config.voice_id)


class TestWeightCalculator(unittest.TestCase):
    """测试权重计算器"""

    def test_normalize_weights(self):
        """测试权重归一化"""
        weights = {"speaker1": 2.0, "speaker2": 3.0, "speaker3": 1.0}
        result = WeightCalculator.normalize_weights(weights)

        # 检查权重总和为1
        total = sum(result.normalized_weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

        # 检查比例正确
        self.assertAlmostEqual(result.normalized_weights["speaker2"], 0.5, places=6)
        self.assertAlmostEqual(result.normalized_weights["speaker1"], 1/3, places=6)

    def test_zero_weights(self):
        """测试零权重处理"""
        weights = {"speaker1": 0.0, "speaker2": 0.0}
        result = WeightCalculator.normalize_weights(weights)

        # 零权重应该平均分配
        for weight in result.normalized_weights.values():
            self.assertAlmostEqual(weight, 0.5, places=6)

    def test_negative_weights(self):
        """测试负权重处理"""
        weights = {"speaker1": -1.0, "speaker2": 2.0}
        result = WeightCalculator.normalize_weights(weights)

        # 负权重应该被设为0
        self.assertEqual(result.normalized_weights["speaker1"], 0.0)
        self.assertEqual(result.normalized_weights["speaker2"], 1.0)

    def test_merge_weights(self):
        """测试权重合并"""
        old_weights = {"speaker1": 1.0, "speaker2": 0.0}
        new_weights = {"speaker1": 0.0, "speaker2": 1.0}

        result = WeightCalculator.merge_weights(old_weights, new_weights, 0.5)

        # 50%混合应该得到相等权重
        self.assertAlmostEqual(result.normalized_weights["speaker1"], 0.5, places=6)
        self.assertAlmostEqual(result.normalized_weights["speaker2"], 0.5, places=6)


class TestVoiceManager(unittest.TestCase):
    """测试音色管理器"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(self.temp_dir)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_voice(self):
        """测试保存和加载音色"""
        # 创建测试音色
        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml"
        )

        index_tts_config = IndexTTSConfig(
            model_path="test_model",
            config_path="test_config.yaml"
        )

        voice_config = VoiceConfig(
            name="测试音色",
            description="这是一个测试音色",
            tags=["测试", "demo"],
            ddsp_config=ddsp_config,
            index_tts_config=index_tts_config
        )

        # 保存音色
        self.voice_manager.save_voice(voice_config)

        # 加载音色
        loaded_voice = self.voice_manager.load_voice(voice_config.voice_id)

        # 验证
        self.assertEqual(loaded_voice.name, voice_config.name)
        self.assertEqual(loaded_voice.description, voice_config.description)
        self.assertEqual(loaded_voice.tags, voice_config.tags)
        self.assertEqual(loaded_voice.voice_id, voice_config.voice_id)

    def test_list_voices(self):
        """测试列出音色"""
        # 初始应该为空
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 0)

        # 添加音色
        voice_config = VoiceConfig(
            name="测试音色",
            ddsp_config=DDSPSVCConfig("model.pth", "config.yaml"),
            index_tts_config=IndexTTSConfig("model", "config.yaml")
        )

        self.voice_manager.save_voice(voice_config)

        # 检查列表
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].name, "测试音色")

    def test_search_voices(self):
        """测试搜索音色"""
        # 创建多个音色
        voice1 = VoiceConfig(
            name="女声音色",
            tags=["女声", "温柔"],
            ddsp_config=DDSPSVCConfig("model1.pth", "config1.yaml"),
            index_tts_config=IndexTTSConfig("model1", "config1.yaml")
        )

        voice2 = VoiceConfig(
            name="男声音色",
            tags=["男声", "磁性"],
            ddsp_config=DDSPSVCConfig("model2.pth", "config2.yaml"),
            index_tts_config=IndexTTSConfig("model2", "config2.yaml")
        )

        self.voice_manager.save_voice(voice1)
        self.voice_manager.save_voice(voice2)

        # 按名称搜索
        results = self.voice_manager.search_voices(name_pattern="女声")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "女声音色")

        # 按标签搜索
        results = self.voice_manager.search_voices(tags=["温柔"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "女声音色")


class TestConfig(unittest.TestCase):
    """测试配置管理"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)

    def test_default_config(self):
        """测试默认配置"""
        config = Config()

        # 检查默认值
        self.assertEqual(config.ddsp_svc.default_f0_predictor, "rmvpe")
        self.assertEqual(config.index_tts.use_fp16, False)
        self.assertEqual(config.ui.port, 7860)
        self.assertEqual(config.system.device, "auto")

    def test_config_validation(self):
        """测试配置验证"""
        config = Config()

        # 正常配置应该通过验证
        errors = config.validate()
        self.assertEqual(len(errors), 0)

        # 无效端口应该失败
        config.ui.port = 99999
        errors = config.validate()
        self.assertGreater(len(errors), 0)

    def test_config_serialization(self):
        """测试配置序列化"""
        config = Config()
        config.ddsp_svc.model_dir = "custom_ddsp_dir"
        config.ui.port = 8080

        # 转换为字典
        data = config.to_dict()
        self.assertEqual(data["ddsp_svc"]["model_dir"], "custom_ddsp_dir")
        self.assertEqual(data["ui"]["port"], 8080)

        # 从字典恢复
        restored_config = Config.from_dict(data)
        self.assertEqual(restored_config.ddsp_svc.model_dir, "custom_ddsp_dir")
        self.assertEqual(restored_config.ui.port, 8080)

    def test_config_manager(self):
        """测试配置管理器"""
        config_manager = ConfigManager(self.config_path)

        # 加载配置（应该创建默认配置）
        config = config_manager.load_config()
        self.assertIsInstance(config, Config)

        # 配置文件应该被创建
        self.assertTrue(self.config_path.exists())

        # 修改配置
        config.ui.port = 9999
        config_manager.save_config(config)

        # 重新加载配置
        new_manager = ConfigManager(self.config_path)
        loaded_config = new_manager.load_config()
        self.assertEqual(loaded_config.ui.port, 9999)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestModels,
        TestWeightCalculator,
        TestVoiceManager,
        TestConfig
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
    sys.exit(0 if success else 1)
