#!/usr/bin/env python3
"""音色继承和融合功能测试

这个模块测试音色继承和融合的核心功能。
测试内容：
1. 高级权重计算器
2. 音色继承器
3. 音色融合器
4. 界面组件
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from src.core.voice_manager import VoiceManager
from src.core.advanced_weight_calc import AdvancedWeightCalculator
from src.core.voice_inheritance import VoiceInheritor, InheritanceConfig
from src.core.voice_fusion import VoiceFuser, FusionConfig, FusionSource


class TestAdvancedWeightCalculator(unittest.TestCase):
    """测试高级权重计算器"""

    def setUp(self):
        """设置测试环境"""
        self.calculator = AdvancedWeightCalculator()

    def test_calculate_inheritance_weights(self):
        """测试继承权重计算"""
        # 创建父音色配置
        parent_ddsp = DDSPSVCConfig(
            model_path="parent_model.pth",
            config_path="parent_config.yaml",
            speaker_id=0,
            spk_mix_dict={"speaker1": 0.6, "speaker2": 0.4}
        )

        parent_index = IndexTTSConfig(
            model_path="parent_index_model",
            config_path="parent_index_config.yaml",
            speaker_name="parent_speaker"
        )

        parent_voice = VoiceConfig(
            voice_id="parent_id",
            name="Parent Voice",
            ddsp_config=parent_ddsp,
            index_tts_config=parent_index
        )

        # 创建新配置
        new_ddsp = DDSPSVCConfig(
            model_path="new_model.pth",
            config_path="new_config.yaml",
            speaker_id=1,
            spk_mix_dict={"speaker3": 0.7, "speaker4": 0.3}
        )

        new_index = IndexTTSConfig(
            model_path="new_index_model",
            config_path="new_index_config.yaml",
            speaker_name="new_speaker"
        )

        # 计算继承权重
        result = self.calculator.calculate_inheritance_weights(
            parent_voice, new_ddsp, new_index, 0.6
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("speaker1", result.ddsp_weights)
        self.assertIn("speaker3", result.ddsp_weights)

        # 验证权重归一化
        total_weight = sum(result.ddsp_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)

    def test_calculate_fusion_weights(self):
        """测试融合权重计算"""
        # 创建多个音色配置
        voice1 = VoiceConfig(
            voice_id="voice1",
            name="Voice 1",
            ddsp_config=DDSPSVCConfig(
                model_path="model1.pth",
                config_path="config1.yaml",
                speaker_id=0,
                spk_mix_dict={"speaker1": 1.0}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="index1",
                config_path="index1.yaml",
                speaker_name="speaker1"
            )
        )

        voice2 = VoiceConfig(
            voice_id="voice2",
            name="Voice 2",
            ddsp_config=DDSPSVCConfig(
                model_path="model2.pth",
                config_path="config2.yaml",
                speaker_id=1,
                spk_mix_dict={"speaker2": 1.0}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="index2",
                config_path="index2.yaml",
                speaker_name="speaker2"
            )
        )

        voice_configs = {"voice1": voice1, "voice2": voice2}
        fusion_weights = {"voice1": 0.6, "voice2": 0.4}

        # 计算融合权重
        result = self.calculator.calculate_fusion_weights(voice_configs, fusion_weights)

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIn("speaker1", result.ddsp_weights)
        self.assertIn("speaker2", result.ddsp_weights)

        # 验证权重归一化
        total_weight = sum(result.ddsp_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)


class TestVoiceInheritance(unittest.TestCase):
    """测试音色继承功能"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(self.temp_dir)
        self.inheritor = VoiceInheritor(self.voice_manager)

        # 创建测试用的父音色
        self.parent_voice = VoiceConfig(
            voice_id="parent_voice_id",
            name="Parent Voice",
            description="Test parent voice",
            ddsp_config=DDSPSVCConfig(
                model_path="parent_model.pth",
                config_path="parent_config.yaml",
                speaker_id=0,
                f0_predictor="rmvpe",
                f0_min=50.0,
                f0_max=1100.0,
                threhold=-60.0,
                spk_mix_dict={"speaker1": 0.6, "speaker2": 0.4}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="parent_index_model",
                config_path="parent_index_config.yaml",
                speaker_name="parent_speaker",
                emotion_strength=1.0,
                speed=1.0,
                temperature=0.7
            ),
            tags=["测试", "父音色"]
        )

        # 保存父音色
        self.voice_manager.save_voice(self.parent_voice)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_inherit_from_voice(self):
        """测试从现有音色继承"""
        # 创建新配置
        new_ddsp = DDSPSVCConfig(
            model_path="new_model.pth",
            config_path="new_config.yaml",
            speaker_id=1,
            f0_predictor="fcpe",
            f0_min=60.0,
            f0_max=1200.0,
            threhold=-50.0,
            spk_mix_dict={"speaker3": 0.8, "speaker4": 0.2}
        )

        new_index = IndexTTSConfig(
            model_path="new_index_model",
            config_path="new_index_config.yaml",
            speaker_name="new_speaker",
            emotion_strength=1.2,
            speed=1.1,
            temperature=0.8
        )

        inheritance_config = InheritanceConfig(
            inheritance_ratio=0.7,
            preserve_metadata=True,
            copy_tags=True
        )

        # 执行继承
        result = self.inheritor.inherit_from_voice(
            self.parent_voice.voice_id,
            "Inherited Voice",
            new_ddsp,
            new_index,
            inheritance_config
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.new_voice_config.name, "Inherited Voice")
        self.assertEqual(result.parent_voice_id, self.parent_voice.voice_id)
        self.assertIn("继承音色", result.new_voice_config.tags)

        # 验证参数继承
        inherited_ddsp = result.new_voice_config.ddsp_config
        self.assertIsNotNone(inherited_ddsp.spk_mix_dict)

        # 验证权重归一化
        total_weight = sum(inherited_ddsp.spk_mix_dict.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)

    def test_preview_inheritance(self):
        """测试继承预览"""
        new_ddsp = DDSPSVCConfig(
            model_path="new_model.pth",
            config_path="new_config.yaml",
            speaker_id=1,
            spk_mix_dict={"speaker3": 1.0}
        )

        new_index = IndexTTSConfig(
            model_path="new_index_model",
            config_path="new_index_config.yaml",
            speaker_name="new_speaker"
        )

        # 生成预览
        preview = self.inheritor.preview_inheritance(
            self.parent_voice.voice_id,
            new_ddsp,
            new_index,
            0.5
        )

        # 验证预览结果
        self.assertIsNotNone(preview)
        self.assertIn("parent_voice", preview)
        self.assertIn("resulting_weights", preview)
        self.assertIn("inheritance_ratio", preview)
        self.assertEqual(preview["inheritance_ratio"], 0.5)


class TestVoiceFusion(unittest.TestCase):
    """测试音色融合功能"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(self.temp_dir)
        self.fuser = VoiceFuser(self.voice_manager)

        # 创建测试音色
        self.voice1 = VoiceConfig(
            voice_id="voice1_id",
            name="Voice 1",
            ddsp_config=DDSPSVCConfig(
                model_path="model1.pth",
                config_path="config1.yaml",
                speaker_id=0,
                spk_mix_dict={"speaker1": 0.7, "speaker2": 0.3}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="index1",
                config_path="index1.yaml",
                speaker_name="speaker1"
            ),
            tags=["测试", "音色1"]
        )

        self.voice2 = VoiceConfig(
            voice_id="voice2_id",
            name="Voice 2",
            ddsp_config=DDSPSVCConfig(
                model_path="model2.pth",
                config_path="config2.yaml",
                speaker_id=1,
                spk_mix_dict={"speaker3": 0.6, "speaker4": 0.4}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="index2",
                config_path="index2.yaml",
                speaker_name="speaker2"
            ),
            tags=["测试", "音色2"]
        )

        # 保存测试音色
        self.voice_manager.save_voice(self.voice1)
        self.voice_manager.save_voice(self.voice2)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fuse_voices(self):
        """测试音色融合"""
        # 创建融合源
        fusion_sources = [
            FusionSource(voice_config=self.voice1, weight=0.6),
            FusionSource(voice_config=self.voice2, weight=0.4)
        ]

        fusion_config = FusionConfig(
            auto_normalize_weights=True,
            resolve_conflicts=True
        )

        # 执行融合
        result = self.fuser.fuse_voices(
            fusion_sources,
            "Fused Voice",
            fusion_config
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.fused_voice_config.name, "Fused Voice")
        self.assertEqual(len(result.source_voices), 2)
        self.assertIn("融合音色", result.fused_voice_config.tags)

        # 验证权重归一化
        ddsp_weights = result.fusion_weights.ddsp_weights
        total_weight = sum(ddsp_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)

    def test_fuse_by_voice_ids(self):
        """测试通过音色ID融合"""
        voice_ids_and_weights = {
            self.voice1.voice_id: 0.7,
            self.voice2.voice_id: 0.3
        }

        # 执行融合
        result = self.fuser.fuse_by_voice_ids(
            voice_ids_and_weights,
            "ID Fused Voice"
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result.fused_voice_config.name, "ID Fused Voice")
        self.assertEqual(len(result.source_voices), 2)

    def test_preview_fusion(self):
        """测试融合预览"""
        fusion_sources = [
            FusionSource(voice_config=self.voice1, weight=0.5),
            FusionSource(voice_config=self.voice2, weight=0.5)
        ]

        # 生成预览
        preview = self.fuser.preview_fusion(fusion_sources)

        # 验证预览结果
        self.assertIsNotNone(preview)
        self.assertIn("source_voices", preview)
        self.assertIn("fusion_weights", preview)
        self.assertIn("compatibility_analysis", preview)
        self.assertEqual(len(preview["source_voices"]), 2)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.voice_manager = VoiceManager(self.temp_dir)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_inheritance_then_fusion(self):
        """测试继承后融合的完整流程"""
        # 1. 创建基础音色
        base_voice = VoiceConfig(
            voice_id="base_voice",
            name="Base Voice",
            ddsp_config=DDSPSVCConfig(
                model_path="base_model.pth",
                config_path="base_config.yaml",
                speaker_id=0,
                spk_mix_dict={"base_speaker": 1.0}
            ),
            index_tts_config=IndexTTSConfig(
                model_path="base_index",
                config_path="base_index.yaml",
                speaker_name="base_speaker"
            )
        )
        self.voice_manager.save_voice(base_voice)

        # 2. 创建继承音色
        inheritor = VoiceInheritor(self.voice_manager)

        new_ddsp = DDSPSVCConfig(
            model_path="inherited_model.pth",
            config_path="inherited_config.yaml",
            speaker_id=1,
            spk_mix_dict={"inherited_speaker": 1.0}
        )

        new_index = IndexTTSConfig(
            model_path="inherited_index",
            config_path="inherited_index.yaml",
            speaker_name="inherited_speaker"
        )

        inheritance_result = inheritor.inherit_from_voice(
            base_voice.voice_id,
            "Inherited Voice",
            new_ddsp,
            new_index
        )

        # 保存继承结果
        self.voice_manager.save_voice(inheritance_result.new_voice_config)

        # 3. 融合基础音色和继承音色
        fuser = VoiceFuser(self.voice_manager)

        fusion_sources = [
            FusionSource(voice_config=base_voice, weight=0.4),
            FusionSource(voice_config=inheritance_result.new_voice_config, weight=0.6)
        ]

        fusion_result = fuser.fuse_voices(
            fusion_sources,
            "Final Fused Voice"
        )

        # 验证最终结果
        self.assertIsNotNone(fusion_result)
        self.assertEqual(fusion_result.fused_voice_config.name, "Final Fused Voice")

        # 验证融合音色包含了继承关系
        self.assertIn(base_voice.voice_id, fusion_result.fused_voice_config.parent_voice_ids)
        self.assertIn(inheritance_result.new_voice_config.voice_id,
                     fusion_result.fused_voice_config.parent_voice_ids)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestAdvancedWeightCalculator,
        TestVoiceInheritance,
        TestVoiceFusion,
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
    print("开始运行音色继承和融合功能测试...")
    success = run_tests()

    if success:
        print("\n✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败！")
        sys.exit(1)
