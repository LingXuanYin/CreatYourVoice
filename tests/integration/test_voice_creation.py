"""音色创建集成测试

测试音色创建的各个组件集成工作。
设计原则：
1. 组件协作 - 测试各模块间的接口和数据传递
2. 边界条件 - 测试极端情况和错误处理
3. 数据一致性 - 验证数据在组件间传递的完整性
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from src.core.voice_manager import VoiceManager, VoiceNotFoundError, VoiceAlreadyExistsError
from src.core.weight_calculator import WeightCalculator, WeightCalculationResult
from src.core.voice_base_creator import VoiceBaseCreator, VoiceBaseCreationParams
from src.core.voice_preset_manager import VoicePresetManager
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class VoiceCreationIntegrationTest(unittest.TestCase):
    """音色创建集成测试"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="DEBUG", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.voices_dir = self.temp_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.voice_manager = VoiceManager(self.voices_dir)
        self.preset_manager = VoicePresetManager()

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_voice_config_serialization(self):
        """测试音色配置序列化和反序列化"""
        logger.info("测试音色配置序列化")

        # 创建完整的音色配置
        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1,
            f0_predictor="rmvpe",
            f0_min=50.0,
            f0_max=1100.0,
            threhold=-60.0,
            spk_mix_dict={"speaker_001": 0.6, "speaker_002": 0.4},
            use_spk_mix=True
        )

        index_config = IndexTTSConfig(
            model_path="index_model",
            config_path="index_config.yaml",
            speaker_name="test_speaker",
            emotion_strength=0.75,
            speed=1.2,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        weight_info = WeightInfo(
            speaker_weights={"speaker_001": 0.6, "speaker_002": 0.4}
        )

        original_voice = VoiceConfig(
            name="序列化测试音色",
            description="用于测试序列化功能的音色",
            tags=["测试", "序列化", "集成"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config,
            weight_info=weight_info,
            parent_voice_ids=["parent_001", "parent_002"],
            fusion_weights={"parent_001": 0.7, "parent_002": 0.3}
        )

        # 序列化到字典
        voice_dict = original_voice.to_dict()

        # 验证字典结构
        self.assertIn("voice_id", voice_dict)
        self.assertIn("name", voice_dict)
        self.assertIn("ddsp_config", voice_dict)
        self.assertIn("index_tts_config", voice_dict)
        self.assertIn("weight_info", voice_dict)

        # 反序列化
        restored_voice = VoiceConfig.from_dict(voice_dict)

        # 验证数据完整性
        self.assertEqual(restored_voice.name, original_voice.name)
        self.assertEqual(restored_voice.description, original_voice.description)
        self.assertEqual(restored_voice.tags, original_voice.tags)
        self.assertEqual(restored_voice.voice_id, original_voice.voice_id)

        # 验证DDSP配置
        self.assertEqual(restored_voice.ddsp_config.model_path, ddsp_config.model_path)
        self.assertEqual(restored_voice.ddsp_config.speaker_id, ddsp_config.speaker_id)
        self.assertEqual(restored_voice.ddsp_config.spk_mix_dict, ddsp_config.spk_mix_dict)

        # 验证IndexTTS配置
        self.assertEqual(restored_voice.index_tts_config.speaker_name, index_config.speaker_name)
        self.assertEqual(restored_voice.index_tts_config.emotion_strength, index_config.emotion_strength)

        # 验证权重信息
        self.assertEqual(restored_voice.weight_info.speaker_weights, weight_info.speaker_weights)

    def test_voice_manager_operations(self):
        """测试音色管理器操作"""
        logger.info("测试音色管理器操作")

        # 创建测试音色
        voice = self._create_test_voice("管理器测试音色")

        # 测试保存
        self.voice_manager.save_voice(voice)

        # 测试加载
        loaded_voice = self.voice_manager.load_voice(voice.voice_id)
        self.assertEqual(loaded_voice.name, voice.name)
        self.assertEqual(loaded_voice.voice_id, voice.voice_id)

        # 测试列表
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].voice_id, voice.voice_id)

        # 测试搜索
        search_results = self.voice_manager.search_voices(name_pattern="管理器")
        self.assertEqual(len(search_results), 1)

        search_results = self.voice_manager.search_voices(tags=["测试"])
        self.assertEqual(len(search_results), 1)

        # 测试重复保存（应该失败）
        with self.assertRaises(VoiceAlreadyExistsError):
            self.voice_manager.save_voice(voice, overwrite=False)

        # 测试覆盖保存
        voice.description = "更新后的描述"
        self.voice_manager.save_voice(voice, overwrite=True)

        updated_voice = self.voice_manager.load_voice(voice.voice_id)
        self.assertEqual(updated_voice.description, "更新后的描述")

        # 测试删除
        self.voice_manager.delete_voice(voice.voice_id)

        with self.assertRaises(VoiceNotFoundError):
            self.voice_manager.load_voice(voice.voice_id)

        # 验证列表为空
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 0)

    def test_weight_calculator_integration(self):
        """测试权重计算器集成"""
        logger.info("测试权重计算器集成")

        # 测试基本权重归一化
        weights = {"speaker_001": 2.0, "speaker_002": 3.0, "speaker_003": 1.0}
        result = WeightCalculator.normalize_weights(weights)

        self.assertIsInstance(result, WeightCalculationResult)
        self.assertAlmostEqual(sum(result.normalized_weights.values()), 1.0, places=6)
        self.assertEqual(result.total_original, 6.0)
        self.assertAlmostEqual(result.scaling_factor, 1.0/6.0, places=6)

        # 测试零权重处理
        zero_weights = {"speaker_001": 0.0, "speaker_002": 0.0}
        zero_result = WeightCalculator.normalize_weights(zero_weights)

        for weight in zero_result.normalized_weights.values():
            self.assertAlmostEqual(weight, 0.5, places=6)

        # 测试负权重处理
        negative_weights = {"speaker_001": -1.0, "speaker_002": 2.0}
        negative_result = WeightCalculator.normalize_weights(negative_weights)

        self.assertEqual(negative_result.normalized_weights["speaker_001"], 0.0)
        self.assertEqual(negative_result.normalized_weights["speaker_002"], 1.0)

        # 测试权重合并
        old_weights = {"speaker_001": 1.0, "speaker_002": 0.0}
        new_weights = {"speaker_001": 0.0, "speaker_002": 1.0}

        merge_result = WeightCalculator.merge_weights(old_weights, new_weights, 0.5)

        # 50%混合应该得到相等权重
        self.assertAlmostEqual(merge_result.normalized_weights["speaker_001"], 0.5, places=6)
        self.assertAlmostEqual(merge_result.normalized_weights["speaker_002"], 0.5, places=6)

    def test_voice_config_validation(self):
        """测试音色配置验证"""
        logger.info("测试音色配置验证")

        # 测试有效配置
        valid_voice = self._create_test_voice("有效音色")

        # 验证DDSP配置后处理
        self.assertIsNotNone(valid_voice.ddsp_config.spk_mix_dict)
        if valid_voice.ddsp_config.spk_mix_dict:
            self.assertTrue(valid_voice.ddsp_config.use_spk_mix)

        # 验证IndexTTS配置参数范围
        index_config = valid_voice.index_tts_config
        self.assertGreaterEqual(index_config.emotion_strength, 0.0)
        self.assertLessEqual(index_config.emotion_strength, 2.0)
        self.assertGreaterEqual(index_config.speed, 0.1)
        self.assertLessEqual(index_config.speed, 3.0)
        self.assertGreaterEqual(index_config.temperature, 0.1)
        self.assertLessEqual(index_config.temperature, 2.0)

        # 验证权重信息归一化
        weight_info = valid_voice.weight_info
        if weight_info.speaker_weights:
            total_weight = sum(weight_info.speaker_weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=6)

    def test_file_operations(self):
        """测试文件操作"""
        logger.info("测试文件操作")

        voice = self._create_test_voice("文件操作测试音色")

        # 测试保存到文件
        file_path = self.temp_dir / "test_voice.json"
        voice.save_to_file(file_path)

        self.assertTrue(file_path.exists())

        # 测试从文件加载
        loaded_voice = VoiceConfig.load_from_file(file_path)

        self.assertEqual(loaded_voice.name, voice.name)
        self.assertEqual(loaded_voice.voice_id, voice.voice_id)

        # 测试导出和导入
        export_path = self.temp_dir / "exported_voice.json"
        self.voice_manager.save_voice(voice)
        self.voice_manager.export_voice(voice.voice_id, export_path)

        self.assertTrue(export_path.exists())

        # 删除原音色
        self.voice_manager.delete_voice(voice.voice_id)

        # 导入音色
        imported_voice = self.voice_manager.import_voice(export_path)

        self.assertEqual(imported_voice.name, voice.name)
        self.assertEqual(imported_voice.voice_id, voice.voice_id)

    def test_voice_statistics(self):
        """测试音色统计"""
        logger.info("测试音色统计")

        # 创建多个测试音色
        voices = []
        for i in range(5):
            voice = self._create_test_voice(f"统计测试音色_{i}")
            if i % 2 == 0:
                voice.tags.append("偶数")
            else:
                voice.tags.append("奇数")

            # 设置父音色（模拟融合音色）
            if i > 0:
                voice.parent_voice_ids = [voices[0].voice_id]

            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 获取统计信息
        stats = self.voice_manager.get_statistics()

        self.assertEqual(stats["total_voices"], 5)
        self.assertEqual(stats["original_voices"], 1)  # 只有第一个没有父音色
        self.assertEqual(stats["fusion_voices"], 4)   # 其他都有父音色

        # 验证标签统计
        most_used_tags = dict(stats["most_used_tags"])
        self.assertIn("测试", most_used_tags)
        self.assertEqual(most_used_tags["测试"], 5)  # 所有音色都有"测试"标签

    def test_voice_duplication(self):
        """测试音色复制"""
        logger.info("测试音色复制")

        # 创建原始音色
        original_voice = self._create_test_voice("原始音色")
        self.voice_manager.save_voice(original_voice)

        # 复制音色
        modifications = {
            "description": "这是复制的音色",
            "ddsp_config.speaker_id": 2,
            "index_tts_config.emotion_strength": 0.8
        }

        duplicated_voice = self.voice_manager.duplicate_voice(
            original_voice.voice_id,
            "复制音色",
            modifications
        )

        # 验证复制结果
        self.assertNotEqual(duplicated_voice.voice_id, original_voice.voice_id)
        self.assertEqual(duplicated_voice.name, "复制音色")
        self.assertEqual(duplicated_voice.description, "这是复制的音色")
        self.assertIn(original_voice.voice_id, duplicated_voice.parent_voice_ids)

        # 验证修改应用
        self.assertEqual(duplicated_voice.ddsp_config.speaker_id, 2)
        self.assertEqual(duplicated_voice.index_tts_config.emotion_strength, 0.8)

        # 验证其他属性保持不变
        self.assertEqual(duplicated_voice.ddsp_config.model_path, original_voice.ddsp_config.model_path)

    def test_error_handling(self):
        """测试错误处理"""
        logger.info("测试错误处理")

        # 测试加载不存在的音色
        with self.assertRaises(VoiceNotFoundError):
            self.voice_manager.load_voice("non_existent_id")

        # 测试删除不存在的音色
        with self.assertRaises(VoiceNotFoundError):
            self.voice_manager.delete_voice("non_existent_id")

        # 测试无效的权重
        invalid_weights = {}
        result = WeightCalculator.normalize_weights(invalid_weights)
        self.assertEqual(result.normalized_weights, {})

        # 测试权重验证
        is_valid, errors = WeightCalculator.validate_weights({"speaker_001": -1.0})
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def _create_test_voice(self, name: str) -> VoiceConfig:
        """创建测试音色"""
        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1,
            spk_mix_dict={"speaker_001": 0.6, "speaker_002": 0.4},
            use_spk_mix=True
        )

        index_config = IndexTTSConfig(
            model_path="test_index_model",
            config_path="test_index_config.yaml",
            speaker_name="test_speaker",
            emotion_strength=0.75
        )

        weight_info = WeightInfo(
            speaker_weights={"speaker_001": 0.6, "speaker_002": 0.4}
        )

        return VoiceConfig(
            name=name,
            description=f"测试音色: {name}",
            tags=["测试", "集成"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config,
            weight_info=weight_info
        )


def run_voice_creation_tests():
    """运行音色创建集成测试"""
    test_suite = unittest.TestSuite()

    # 添加测试方法
    test_methods = [
        'test_voice_config_serialization',
        'test_voice_manager_operations',
        'test_weight_calculator_integration',
        'test_voice_config_validation',
        'test_file_operations',
        'test_voice_statistics',
        'test_voice_duplication',
        'test_error_handling'
    ]

    for method in test_methods:
        test_suite.addTest(VoiceCreationIntegrationTest(method))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_voice_creation_tests()
    sys.exit(0 if success else 1)
