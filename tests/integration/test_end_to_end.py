"""端到端测试套件

这个模块测试完整的工作流程，确保所有组件协同工作。
设计原则：
1. 真实场景 - 模拟用户实际使用流程
2. 完整覆盖 - 测试所有主要功能路径
3. 数据驱动 - 使用真实的测试数据
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
import sys
import logging
from typing import Optional, Dict, Any, Union
from unittest.mock import Mock, MagicMock

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.voice_manager import VoiceManager
from src.core.voice_base_creator import VoiceBaseCreator, VoiceBaseCreationParams
from src.core.voice_inheritance import VoiceInheritor, InheritanceConfig
from src.core.voice_fusion import VoiceFuser, FusionSource, FusionConfig
from src.core.voice_preset_manager import VoicePresetManager
from src.integrations.ddsp_svc import DDSPSVCIntegration, DDSPSVCResult
from src.integrations.index_tts import IndexTTSIntegration, IndexTTSResult
from src.utils.config import Config, ConfigManager
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class EndToEndTestCase(unittest.TestCase):
    """端到端测试基类"""

    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 设置日志
        setup_logging(log_level="DEBUG", console_output=True)

        # 创建临时目录
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.voices_dir = cls.temp_dir / "voices"
        cls.models_dir = cls.temp_dir / "models"
        cls.outputs_dir = cls.temp_dir / "outputs"

        # 创建目录
        cls.voices_dir.mkdir(parents=True, exist_ok=True)
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试配置
        cls.config = Config()
        cls.config.system.voices_dir = str(cls.voices_dir)
        cls.config.system.outputs_dir = str(cls.outputs_dir)
        cls.config.ddsp_svc.model_dir = str(cls.models_dir / "ddsp")
        cls.config.index_tts.model_dir = str(cls.models_dir / "index_tts")

        logger.info(f"测试环境设置完成: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
        logger.info("测试环境清理完成")

    def setUp(self):
        """每个测试前的设置"""
        self.voice_manager = VoiceManager(self.voices_dir)
        self.preset_manager = VoicePresetManager()

        # 创建模拟的集成组件
        self.ddsp_integration = self._create_mock_ddsp_integration()
        self.index_tts_integration = self._create_mock_index_tts_integration()

        self.voice_creator = VoiceBaseCreator(
            preset_manager=self.preset_manager,
            voice_manager=self.voice_manager,
            ddsp_integration=self.ddsp_integration,
            index_tts_integration=self.index_tts_integration,
            temp_dir=self.temp_dir / "voice_creation"
        )

    def _create_mock_ddsp_integration(self) -> DDSPSVCIntegration:
        """创建模拟的DDSP-SVC集成"""
        mock = Mock(spec=DDSPSVCIntegration)

        def mock_infer(**kwargs):
            # 创建模拟的音频数据
            try:
                import numpy as np
                audio = np.random.randn(22050).astype(np.float32)
            except ImportError:
                # 如果numpy不可用，使用简单的列表
                audio = [0.0] * 22050

            return DDSPSVCResult(
                audio=audio,
                sample_rate=22050,
                processing_time=0.5,
                segments_count=1
            )

        mock.load_model = Mock()
        mock.infer = Mock(side_effect=mock_infer)
        mock.save_audio = Mock()

        return mock

    def _create_mock_index_tts_integration(self) -> IndexTTSIntegration:
        """创建模拟的IndexTTS集成"""
        mock = Mock(spec=IndexTTSIntegration)

        def mock_infer(**kwargs):
            output_path = kwargs.get('output_path')
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()

            try:
                import numpy as np
                audio_data = (22050, np.random.randn(22050).astype(np.float32))
            except ImportError:
                audio_data = (22050, [0.0] * 22050)

            return IndexTTSResult(
                audio_path=output_path,
                audio_data=audio_data if not output_path else None,
                processing_time=1.0,
                segments_count=1
            )

        mock.load_model = Mock()
        mock.infer = Mock(side_effect=mock_infer)

        return mock


class TestCompleteVoiceCreationWorkflow(EndToEndTestCase):
    """测试完整的音色创建工作流"""

    def test_voice_base_creation_workflow(self):
        """测试角色声音基底创建完整流程"""
        logger.info("开始测试角色声音基底创建工作流")

        # 1. 创建测试参数
        params = VoiceBaseCreationParams(
            voice_name="测试角色",
            description="端到端测试创建的角色",
            tags=["测试", "端到端"],
            selected_tag="female_young",
            pitch_shift=2.0,
            formant_shift=1.0,
            vocal_register_shift=0.5,
            speaker_weights={"speaker_001": 0.7, "speaker_002": 0.3},
            preview_text="这是一个测试音色的预览文本。",
            emotion_control="speaker",
            emotion_weight=0.65
        )

        # 2. 执行创建
        start_time = time.time()
        result = self.voice_creator.create_voice_base(params)
        creation_time = time.time() - start_time

        # 3. 验证结果
        self.assertTrue(result.success, f"创建失败: {result.error_message}")
        self.assertIsNotNone(result.voice_config)
        self.assertIsNotNone(result.preview_audio_path)
        self.assertGreater(result.processing_time, 0)

        # 4. 验证音色配置
        voice_config = result.voice_config
        self.assertIsNotNone(voice_config)
        if voice_config:
            self.assertEqual(voice_config.name, "测试角色")
            self.assertIn("测试", voice_config.tags)
            self.assertIn("female_young", voice_config.tags)

            # 5. 验证DDSP配置
            ddsp_config = voice_config.ddsp_config
            self.assertIsNotNone(ddsp_config.model_path)
            self.assertIsNotNone(ddsp_config.config_path)
            self.assertTrue(ddsp_config.use_spk_mix)
            if ddsp_config.spk_mix_dict:
                self.assertEqual(len(ddsp_config.spk_mix_dict), 2)

                # 6. 验证权重归一化
                total_weight = sum(ddsp_config.spk_mix_dict.values())
                self.assertAlmostEqual(total_weight, 1.0, places=6)

            # 7. 保存音色
            self.voice_creator.save_voice_base(voice_config)

            # 8. 验证保存成功
            saved_voice = self.voice_manager.load_voice(voice_config.voice_id)
            self.assertEqual(saved_voice.name, voice_config.name)

        logger.info(f"角色声音基底创建测试完成，耗时: {creation_time:.2f}s")

    def test_voice_inheritance_workflow(self):
        """测试音色继承完整流程"""
        logger.info("开始测试音色继承工作流")

        # 1. 先创建一个基础音色
        base_voice = self._create_test_voice("基础音色")
        self.voice_manager.save_voice(base_voice)

        # 2. 创建新的配置
        new_ddsp_config = DDSPSVCConfig(
            model_path="new_model.pth",
            config_path="new_config.yaml",
            speaker_id=2,
            spk_mix_dict={"speaker_003": 0.6, "speaker_004": 0.4},
            use_spk_mix=True
        )

        new_index_config = IndexTTSConfig(
            model_path="new_index_model",
            config_path="new_index_config.yaml",
            speaker_name="new_speaker",
            emotion_strength=0.8
        )

        # 3. 执行继承
        inheritor = VoiceInheritor(self.voice_manager)
        inheritance_config = InheritanceConfig(inheritance_ratio=0.6)

        start_time = time.time()
        result = inheritor.inherit_from_voice(
            base_voice.voice_id,
            "继承音色",
            new_ddsp_config,
            new_index_config,
            inheritance_config
        )
        inheritance_time = time.time() - start_time

        # 4. 验证结果
        self.assertIsNotNone(result.new_voice_config)
        self.assertEqual(result.parent_voice_id, base_voice.voice_id)
        self.assertGreater(result.processing_time, 0)

        # 5. 验证继承关系
        inherited_voice = result.new_voice_config
        self.assertEqual(inherited_voice.name, "继承音色")
        self.assertIn(base_voice.voice_id, inherited_voice.parent_voice_ids)
        self.assertIn("继承音色", inherited_voice.tags)

        # 6. 验证权重融合
        self.assertIsNotNone(result.inheritance_weights)
        self.assertGreater(len(result.inheritance_weights.combined_weights), 0)

        # 7. 保存继承音色
        self.voice_manager.save_voice(inherited_voice)

        logger.info(f"音色继承测试完成，耗时: {inheritance_time:.2f}s")

    def test_voice_fusion_workflow(self):
        """测试音色融合完整流程"""
        logger.info("开始测试音色融合工作流")

        # 1. 创建多个基础音色
        voice1 = self._create_test_voice("音色1", {"speaker_001": 1.0})
        voice2 = self._create_test_voice("音色2", {"speaker_002": 1.0})
        voice3 = self._create_test_voice("音色3", {"speaker_003": 1.0})

        for voice in [voice1, voice2, voice3]:
            self.voice_manager.save_voice(voice)

        # 2. 创建融合源
        fusion_sources = [
            FusionSource(voice_config=voice1, weight=0.5, priority=1),
            FusionSource(voice_config=voice2, weight=0.3, priority=2),
            FusionSource(voice_config=voice3, weight=0.2, priority=3)
        ]

        # 3. 执行融合
        fuser = VoiceFuser(self.voice_manager)
        fusion_config = FusionConfig(max_speakers=5, min_weight_threshold=0.05)

        start_time = time.time()
        result = fuser.fuse_voices(fusion_sources, "融合音色", fusion_config)
        fusion_time = time.time() - start_time

        # 4. 验证结果
        self.assertIsNotNone(result.fused_voice_config)
        self.assertEqual(len(result.source_voices), 3)
        self.assertGreater(result.processing_time, 0)

        # 5. 验证融合配置
        fused_voice = result.fused_voice_config
        self.assertEqual(fused_voice.name, "融合音色")
        self.assertIn("融合音色", fused_voice.tags)
        self.assertEqual(len(fused_voice.parent_voice_ids), 3)

        # 6. 验证权重分布
        fusion_weights = result.fusion_weights
        self.assertGreater(len(fusion_weights.combined_weights), 0)

        # 验证权重总和
        total_weight = sum(fusion_weights.combined_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)

        # 7. 保存融合音色
        self.voice_manager.save_voice(fused_voice)

        logger.info(f"音色融合测试完成，耗时: {fusion_time:.2f}s")

    def test_complex_inheritance_chain(self):
        """测试复杂的继承链"""
        logger.info("开始测试复杂继承链")

        # 1. 创建基础音色
        base_voice = self._create_test_voice("基础音色")
        self.voice_manager.save_voice(base_voice)

        # 2. 创建继承链
        inheritor = VoiceInheritor(self.voice_manager)
        current_voice = base_voice

        for i in range(3):
            new_ddsp_config = DDSPSVCConfig(
                model_path=f"model_{i}.pth",
                config_path=f"config_{i}.yaml",
                speaker_id=i + 1,
                spk_mix_dict={f"speaker_{i:03d}": 1.0}
            )

            new_index_config = IndexTTSConfig(
                model_path=f"index_model_{i}",
                config_path=f"index_config_{i}.yaml",
                speaker_name=f"speaker_{i}"
            )

            inheritance_config = InheritanceConfig(inheritance_ratio=0.7)

            result = inheritor.inherit_from_voice(
                current_voice.voice_id,
                f"继承音色_{i+1}",
                new_ddsp_config,
                new_index_config,
                inheritance_config
            )

            self.assertTrue(result.new_voice_config is not None)
            self.voice_manager.save_voice(result.new_voice_config)
            current_voice = result.new_voice_config

        # 3. 验证继承链
        final_voice = current_voice
        self.assertEqual(len(final_voice.parent_voice_ids), 1)  # 直接父音色

        # 验证可以追溯到根音色
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 4)  # 基础 + 3个继承

        logger.info("复杂继承链测试完成")

    def _create_test_voice(self, name: str, speaker_weights: Optional[Dict[str, float]] = None) -> VoiceConfig:
        """创建测试音色"""
        if speaker_weights is None:
            speaker_weights = {"speaker_001": 0.6, "speaker_002": 0.4}

        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1,
            spk_mix_dict=speaker_weights,
            use_spk_mix=len(speaker_weights) > 1
        )

        index_config = IndexTTSConfig(
            model_path="test_index_model",
            config_path="test_index_config.yaml",
            speaker_name="test_speaker"
        )

        return VoiceConfig(
            name=name,
            description=f"测试音色: {name}",
            tags=["测试"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )


class TestSystemIntegration(EndToEndTestCase):
    """测试系统集成"""

    def test_voice_manager_persistence(self):
        """测试音色管理器持久化"""
        logger.info("开始测试音色管理器持久化")

        # 1. 创建并保存音色
        voice = self._create_test_voice("持久化测试音色")
        self.voice_manager.save_voice(voice)

        # 2. 创建新的管理器实例
        new_manager = VoiceManager(self.voices_dir)

        # 3. 验证可以加载音色
        loaded_voice = new_manager.load_voice(voice.voice_id)
        self.assertEqual(loaded_voice.name, voice.name)
        self.assertEqual(loaded_voice.voice_id, voice.voice_id)

        # 4. 验证列表功能
        voices = new_manager.list_voices()
        self.assertEqual(len(voices), 1)
        self.assertEqual(voices[0].voice_id, voice.voice_id)

        logger.info("音色管理器持久化测试完成")

    def test_configuration_management(self):
        """测试配置管理"""
        logger.info("开始测试配置管理")

        # 1. 创建配置文件
        config_path = self.temp_dir / "test_config.yaml"
        config_manager = ConfigManager(config_path)

        # 2. 保存配置
        config = Config()
        config.ui.port = 8080
        config.system.device = "cpu"
        config_manager.save_config(config)

        # 3. 验证文件存在
        self.assertTrue(config_path.exists())

        # 4. 重新加载配置
        new_manager = ConfigManager(config_path)
        loaded_config = new_manager.load_config()

        self.assertEqual(loaded_config.ui.port, 8080)
        self.assertEqual(loaded_config.system.device, "cpu")

        logger.info("配置管理测试完成")

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        logger.info("开始测试错误处理和恢复")

        # 1. 测试加载不存在的音色
        with self.assertRaises(Exception):
            self.voice_manager.load_voice("non_existent_id")

        # 2. 测试保存无效配置
        invalid_params = VoiceBaseCreationParams(
            voice_name="",  # 空名称
            selected_tag="non_existent_tag"
        )

        result = self.voice_creator.create_voice_base(invalid_params)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

        # 3. 测试系统恢复能力
        # 创建正常音色验证系统仍然工作
        valid_voice = self._create_test_voice("恢复测试音色")
        self.voice_manager.save_voice(valid_voice)

        loaded_voice = self.voice_manager.load_voice(valid_voice.voice_id)
        self.assertEqual(loaded_voice.name, valid_voice.name)

        logger.info("错误处理和恢复测试完成")

    def _create_test_voice(self, name: str) -> VoiceConfig:
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
            description=f"测试音色: {name}",
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )


# 移除独立的Mock类，使用内联Mock创建


def run_end_to_end_tests():
    """运行端到端测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        TestCompleteVoiceCreationWorkflow,
        TestSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_end_to_end_tests()
    sys.exit(0 if success else 1)
