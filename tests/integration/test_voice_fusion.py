"""音色融合集成测试

测试音色融合的完整流程和组件集成。
设计原则：
1. 复杂场景 - 测试多音色融合的各种组合
2. 权重验证 - 确保权重计算和归一化的正确性
3. 冲突处理 - 测试参数冲突的自动解决
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List
from unittest.mock import Mock

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from src.core.voice_manager import VoiceManager
from src.core.voice_fusion import VoiceFuser, FusionSource, FusionConfig, FusionResult
from src.core.voice_inheritance import VoiceInheritor, InheritanceConfig
from src.core.advanced_weight_calc import AdvancedWeightCalculator
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class VoiceFusionIntegrationTest(unittest.TestCase):
    """音色融合集成测试"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="DEBUG", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.voices_dir = self.temp_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        self.voice_manager = VoiceManager(self.voices_dir)
        self.voice_fuser = VoiceFuser(self.voice_manager)
        self.voice_inheritor = VoiceInheritor(self.voice_manager)
        self.weight_calculator = AdvancedWeightCalculator()

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_basic_voice_fusion(self):
        """测试基本音色融合"""
        logger.info("测试基本音色融合")

        # 创建源音色
        voice1 = self._create_test_voice("音色1", {"speaker_001": 1.0})
        voice2 = self._create_test_voice("音色2", {"speaker_002": 1.0})

        # 保存音色
        self.voice_manager.save_voice(voice1)
        self.voice_manager.save_voice(voice2)

        # 创建融合源
        fusion_sources = [
            FusionSource(voice_config=voice1, weight=0.6, priority=1),
            FusionSource(voice_config=voice2, weight=0.4, priority=2)
        ]

        # 执行融合
        result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "基本融合音色",
            FusionConfig()
        )

        # 验证结果
        self.assertIsInstance(result, FusionResult)
        self.assertIsNotNone(result.fused_voice_config)
        self.assertEqual(result.fused_voice_config.name, "基本融合音色")
        self.assertEqual(len(result.source_voices), 2)

        # 验证父音色关系
        fused_voice = result.fused_voice_config
        self.assertIn(voice1.voice_id, fused_voice.parent_voice_ids)
        self.assertIn(voice2.voice_id, fused_voice.parent_voice_ids)
        self.assertIn("融合音色", fused_voice.tags)

        # 验证权重归一化
        fusion_weights = result.fusion_weights
        total_weight = sum(fusion_weights.combined_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)

    def test_multi_voice_fusion(self):
        """测试多音色融合"""
        logger.info("测试多音色融合")

        # 创建多个源音色
        voices = []
        for i in range(5):
            voice = self._create_test_voice(
                f"音色{i+1}",
                {f"speaker_{i:03d}": 1.0}
            )
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 创建融合源（不同权重）
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        fusion_sources = [
            FusionSource(voice_config=voice, weight=weight, priority=i+1)
            for i, (voice, weight) in enumerate(zip(voices, weights))
        ]

        # 执行融合
        fusion_config = FusionConfig(
            max_speakers=8,
            min_weight_threshold=0.05
        )

        result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "多音色融合",
            fusion_config
        )

        # 验证结果
        self.assertTrue(result.fused_voice_config is not None)
        self.assertEqual(len(result.source_voices), 5)

        # 验证权重分布
        fusion_weights = result.fusion_weights
        self.assertGreater(len(fusion_weights.combined_weights), 0)

        # 验证权重阈值过滤
        for weight in fusion_weights.combined_weights.values():
            self.assertGreaterEqual(weight, fusion_config.min_weight_threshold)

    def test_fusion_by_voice_ids(self):
        """测试通过音色ID融合"""
        logger.info("测试通过音色ID融合")

        # 创建和保存音色
        voices = []
        for i in range(3):
            voice = self._create_test_voice(f"ID融合音色{i+1}")
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 通过ID和权重融合
        voice_weights = {
            voices[0].voice_id: 0.5,
            voices[1].voice_id: 0.3,
            voices[2].voice_id: 0.2
        }

        result = self.voice_fuser.fuse_by_voice_ids(
            voice_weights,
            "ID融合音色"
        )

        # 验证结果
        self.assertIsNotNone(result.fused_voice_config)
        self.assertEqual(result.fused_voice_config.name, "ID融合音色")
        self.assertEqual(len(result.source_voices), 3)

    def test_fusion_with_conflicts(self):
        """测试有冲突的融合"""
        logger.info("测试有冲突的融合")

        # 创建有冲突参数的音色
        voice1 = self._create_test_voice("冲突音色1")
        voice1.ddsp_config.f0_predictor = "rmvpe"
        voice1.ddsp_config.f0_min = 50.0
        voice1.index_tts_config.speaker_name = "speaker_a"

        voice2 = self._create_test_voice("冲突音色2")
        voice2.ddsp_config.f0_predictor = "crepe"  # 不同的F0预测器
        voice2.ddsp_config.f0_min = 80.0  # 不同的F0范围
        voice2.index_tts_config.speaker_name = "speaker_b"  # 不同的说话人

        # 保存音色
        self.voice_manager.save_voice(voice1)
        self.voice_manager.save_voice(voice2)

        # 创建融合源
        fusion_sources = [
            FusionSource(voice_config=voice1, weight=0.7, priority=1),
            FusionSource(voice_config=voice2, weight=0.3, priority=2)
        ]

        # 执行融合（启用冲突解决）
        fusion_config = FusionConfig(resolve_conflicts=True)
        result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "冲突解决音色",
            fusion_config
        )

        # 验证结果
        self.assertIsNotNone(result.fused_voice_config)
        self.assertGreater(len(result.conflicts_resolved), 0)

        # 验证冲突解决（应该使用主导音色的参数）
        fused_voice = result.fused_voice_config
        self.assertEqual(fused_voice.ddsp_config.f0_predictor, "rmvpe")  # 主导音色的参数
        self.assertEqual(fused_voice.index_tts_config.speaker_name, "speaker_a")

    def test_fusion_preview(self):
        """测试融合预览"""
        logger.info("测试融合预览")

        # 创建音色
        voices = []
        for i in range(3):
            voice = self._create_test_voice(f"预览音色{i+1}")
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 创建融合源
        fusion_sources = [
            FusionSource(voice_config=voice, weight=1.0/(i+1), priority=i+1)
            for i, voice in enumerate(voices)
        ]

        # 预览融合
        preview = self.voice_fuser.preview_fusion(fusion_sources)

        # 验证预览结果
        self.assertIn("source_voices", preview)
        self.assertIn("fusion_weights", preview)
        self.assertIn("compatibility_analysis", preview)
        self.assertIn("speaker_distribution", preview)

        self.assertEqual(len(preview["source_voices"]), 3)

        # 验证权重信息
        fusion_weights = preview["fusion_weights"]
        self.assertIn("combined", fusion_weights)

        # 验证兼容性分析
        compatibility = preview["compatibility_analysis"]
        self.assertIn("model_compatibility", compatibility)

    def test_fusion_optimization(self):
        """测试融合优化"""
        logger.info("测试融合优化")

        # 创建大量音色
        voices = []
        for i in range(10):
            voice = self._create_test_voice(f"优化音色{i+1}")
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 创建融合源（权重递减）
        fusion_sources = [
            FusionSource(voice_config=voice, weight=1.0/(i+1), priority=i+1)
            for i, voice in enumerate(voices)
        ]

        # 执行融合（限制说话人数量）
        fusion_config = FusionConfig(
            max_speakers=5,
            min_weight_threshold=0.1
        )

        result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "优化融合音色",
            fusion_config
        )

        # 验证优化结果
        self.assertIsNotNone(result.fused_voice_config)

        # 验证说话人数量限制
        fusion_weights = result.fusion_weights
        self.assertLessEqual(len(fusion_weights.combined_weights), fusion_config.max_speakers)

        # 验证权重阈值
        for weight in fusion_weights.combined_weights.values():
            self.assertGreaterEqual(weight, fusion_config.min_weight_threshold)

    def test_inheritance_and_fusion_combination(self):
        """测试继承和融合的组合"""
        logger.info("测试继承和融合的组合")

        # 创建基础音色
        base_voice = self._create_test_voice("基础音色")
        self.voice_manager.save_voice(base_voice)

        # 创建继承音色
        new_ddsp_config = DDSPSVCConfig(
            model_path="inherited_model.pth",
            config_path="inherited_config.yaml",
            speaker_id=2,
            spk_mix_dict={"speaker_new": 1.0}
        )

        new_index_config = IndexTTSConfig(
            model_path="inherited_index_model",
            config_path="inherited_index_config.yaml",
            speaker_name="inherited_speaker"
        )

        inheritance_result = self.voice_inheritor.inherit_from_voice(
            base_voice.voice_id,
            "继承音色",
            new_ddsp_config,
            new_index_config,
            InheritanceConfig(inheritance_ratio=0.7)
        )

        inherited_voice = inheritance_result.new_voice_config
        self.voice_manager.save_voice(inherited_voice)

        # 创建另一个音色用于融合
        fusion_voice = self._create_test_voice("融合音色")
        self.voice_manager.save_voice(fusion_voice)

        # 融合继承音色和新音色
        fusion_sources = [
            FusionSource(voice_config=inherited_voice, weight=0.6, priority=1),
            FusionSource(voice_config=fusion_voice, weight=0.4, priority=2)
        ]

        fusion_result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "继承融合音色"
        )

        # 验证结果
        self.assertIsNotNone(fusion_result.fused_voice_config)
        final_voice = fusion_result.fused_voice_config

        # 验证继承链
        self.assertIn(inherited_voice.voice_id, final_voice.parent_voice_ids)
        self.assertIn(fusion_voice.voice_id, final_voice.parent_voice_ids)

    def test_fusion_template_creation(self):
        """测试融合模板创建"""
        logger.info("测试融合模板创建")

        # 创建音色
        voices = []
        for i in range(3):
            voice = self._create_test_voice(f"模板音色{i+1}")
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 创建融合源
        fusion_sources = [
            FusionSource(voice_config=voice, weight=0.5 if i == 0 else 0.25, priority=i+1)
            for i, voice in enumerate(voices)
        ]

        # 创建融合模板
        template = self.voice_fuser.create_fusion_template(
            "测试融合模板",
            fusion_sources,
            "这是一个测试融合模板"
        )

        # 验证模板
        self.assertEqual(template["name"], "测试融合模板")
        self.assertEqual(template["description"], "这是一个测试融合模板")
        self.assertEqual(template["total_sources"], 3)
        self.assertEqual(len(template["sources"]), 3)

        # 验证源信息
        for i, source_info in enumerate(template["sources"]):
            self.assertIn("voice_id", source_info)
            self.assertIn("voice_name", source_info)
            self.assertIn("weight", source_info)
            self.assertIn("priority", source_info)

    def test_weight_calculation_accuracy(self):
        """测试权重计算精度"""
        logger.info("测试权重计算精度")

        # 创建具有复杂权重的音色
        voices = []
        speaker_configs = [
            {"speaker_001": 0.7, "speaker_002": 0.3},
            {"speaker_002": 0.5, "speaker_003": 0.5},
            {"speaker_001": 0.4, "speaker_003": 0.6}
        ]

        for i, config in enumerate(speaker_configs):
            voice = self._create_test_voice(f"权重音色{i+1}", config)
            voices.append(voice)
            self.voice_manager.save_voice(voice)

        # 创建融合源
        fusion_weights = [0.5, 0.3, 0.2]
        fusion_sources = [
            FusionSource(voice_config=voice, weight=weight, priority=i+1)
            for i, (voice, weight) in enumerate(zip(voices, fusion_weights))
        ]

        # 执行融合
        result = self.voice_fuser.fuse_voices(
            fusion_sources,
            "权重精度测试音色"
        )

        # 验证权重计算精度
        final_weights = result.fusion_weights.combined_weights

        # 验证权重总和
        total_weight = sum(final_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)

        # 验证所有说话人都被包含
        all_speakers = set()
        for config in speaker_configs:
            all_speakers.update(config.keys())

        for speaker in all_speakers:
            self.assertIn(speaker, final_weights)
            self.assertGreater(final_weights[speaker], 0)

    def test_fusion_error_handling(self):
        """测试融合错误处理"""
        logger.info("测试融合错误处理")

        # 测试空融合源
        with self.assertRaises(Exception):
            self.voice_fuser.fuse_voices([], "空融合音色")

        # 测试单个融合源
        voice = self._create_test_voice("单一音色")
        self.voice_manager.save_voice(voice)

        with self.assertRaises(Exception):
            fusion_sources = [FusionSource(voice_config=voice, weight=1.0)]
            self.voice_fuser.fuse_voices(fusion_sources, "单一融合音色")

        # 测试零权重
        voice1 = self._create_test_voice("零权重音色1")
        voice2 = self._create_test_voice("零权重音色2")
        self.voice_manager.save_voice(voice1)
        self.voice_manager.save_voice(voice2)

        fusion_sources = [
            FusionSource(voice_config=voice1, weight=0.0),
            FusionSource(voice_config=voice2, weight=0.0)
        ]

        # 应该自动处理零权重（平均分配）
        result = self.voice_fuser.fuse_voices(fusion_sources, "零权重处理音色")
        self.assertIsNotNone(result.fused_voice_config)

    def _create_test_voice(self, name: str, speaker_weights: Optional[Dict[str, float]] = None) -> VoiceConfig:
        """创建测试音色"""
        if speaker_weights is None:
            speaker_weights = {"speaker_001": 0.6, "speaker_002": 0.4}

        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1,
            f0_predictor="rmvpe",
            f0_min=50.0,
            f0_max=1100.0,
            threhold=-60.0,
            spk_mix_dict=speaker_weights,
            use_spk_mix=len(speaker_weights) > 1
        )

        index_config = IndexTTSConfig(
            model_path="test_index_model",
            config_path="test_index_config.yaml",
            speaker_name="test_speaker",
            emotion_strength=0.65
        )

        weight_info = WeightInfo(speaker_weights=speaker_weights)

        return VoiceConfig(
            name=name,
            description=f"测试音色: {name}",
            tags=["测试", "融合"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config,
            weight_info=weight_info
        )


def run_voice_fusion_tests():
    """运行音色融合集成测试"""
    test_suite = unittest.TestSuite()

    # 添加测试方法
    test_methods = [
        'test_basic_voice_fusion',
        'test_multi_voice_fusion',
        'test_fusion_by_voice_ids',
        'test_fusion_with_conflicts',
        'test_fusion_preview',
        'test_fusion_optimization',
        'test_inheritance_and_fusion_combination',
        'test_fusion_template_creation',
        'test_weight_calculation_accuracy',
        'test_fusion_error_handling'
    ]

    for method in test_methods:
        test_suite.addTest(VoiceFusionIntegrationTest(method))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_voice_fusion_tests()
    sys.exit(0 if success else 1)
