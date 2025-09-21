"""音色合成集成测试

测试语音合成的完整流程和组件集成。
设计原则：
1. 流程完整性 - 测试从文本到音频的完整流程
2. 参数验证 - 测试各种参数组合的有效性
3. 质量保证 - 验证输出音频的基本质量指标
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.voice_manager import VoiceManager
from src.core.voice_synthesizer import VoiceSynthesizer, SynthesisParams, SynthesisResult
from src.integrations.ddsp_svc import DDSPSVCIntegration, DDSPSVCResult
from src.integrations.index_tts import IndexTTSIntegration, IndexTTSResult
from src.utils.audio_utils import AudioProcessor
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class VoiceSynthesisIntegrationTest(unittest.TestCase):
    """音色合成集成测试"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="DEBUG", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.voices_dir = self.temp_dir / "voices"
        self.outputs_dir = self.temp_dir / "outputs"
        self.audio_dir = self.temp_dir / "audio"

        for dir_path in [self.voices_dir, self.outputs_dir, self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.voice_manager = VoiceManager(self.voices_dir)

        # 创建模拟的集成组件
        self.ddsp_integration = self._create_mock_ddsp_integration()
        self.index_tts_integration = self._create_mock_index_tts_integration()

    def tearDown(self):
        """清理测试环境"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_mock_ddsp_integration(self) -> Mock:
        """创建模拟的DDSP-SVC集成"""
        mock = Mock(spec=DDSPSVCIntegration)

        def mock_infer(**kwargs):
            # 创建模拟的音频数据
            try:
                import numpy as np
                audio = np.random.randn(22050).astype(np.float32) * 0.1  # 小幅度避免削波
            except ImportError:
                audio = [0.0] * 22050

            return DDSPSVCResult(
                audio=audio,
                sample_rate=22050,
                processing_time=0.5,
                segments_count=1
            )

        def mock_save_audio(result, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).touch()

        mock.load_model = Mock()
        mock.infer = Mock(side_effect=mock_infer)
        mock.save_audio = Mock(side_effect=mock_save_audio)
        mock.get_model_info = Mock(return_value={
            "model_path": "test_model.pth",
            "sampling_rate": 22050,
            "device": "cpu"
        })

        return mock

    def _create_mock_index_tts_integration(self) -> Mock:
        """创建模拟的IndexTTS集成"""
        mock = Mock(spec=IndexTTSIntegration)

        def mock_infer(**kwargs):
            output_path = kwargs.get('output_path')
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_path).touch()

            try:
                import numpy as np
                audio_data = (22050, np.random.randn(22050).astype(np.float32) * 0.1)
            except ImportError:
                audio_data = (22050, [0.0] * 22050)

            return IndexTTSResult(
                audio_path=output_path,
                audio_data=audio_data if not output_path else None,
                processing_time=1.0,
                segments_count=1,
                emotion_info={
                    "control_method": kwargs.get('emotion_control_method', 'speaker'),
                    "emotion_weight": kwargs.get('emotion_weight', 0.65)
                }
            )

        mock.load_model = Mock()
        mock.infer = Mock(side_effect=mock_infer)
        mock.create_emotion_vector = Mock(return_value=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        mock.get_model_info = Mock(return_value={
            "model_dir": "test_model_dir",
            "device": "cpu"
        })

        return mock

    def test_basic_synthesis_workflow(self):
        """测试基本合成工作流"""
        logger.info("测试基本合成工作流")

        # 创建测试音色
        voice = self._create_test_voice("基本合成测试音色")
        self.voice_manager.save_voice(voice)

        # 创建合成器
        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        # 执行合成
        text = "这是一个基本的语音合成测试。"
        output_path = self.outputs_dir / "basic_synthesis.wav"

        # 创建合成参数
        params = SynthesisParams(
            text=text,
            voice_id=voice.voice_id,
            emotion_mode="speaker"
        )

        start_time = time.time()
        result = synthesizer.synthesize(params, str(output_path))
        synthesis_time = time.time() - start_time

        # 验证结果
        self.assertTrue(result.success)
        self.assertIsNotNone(result.audio_path)
        self.assertGreater(result.processing_time, 0)
        self.assertLessEqual(result.processing_time, synthesis_time + 1.0)  # 允许1秒误差

        # 验证文件生成
        self.assertTrue(Path(result.audio_path).exists())

        logger.info(f"基本合成测试完成，耗时: {synthesis_time:.2f}s")

    def test_emotion_control_synthesis(self):
        """测试情感控制合成"""
        logger.info("测试情感控制合成")

        voice = self._create_test_voice("情感控制测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        text = "这是一个情感控制的语音合成测试。"

        # 测试不同的情感控制方式
        emotion_configs = [
            ("speaker", {}),
            ("vector", {"emotion_vector": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]}),
            ("reference", {"emotion_reference_audio": str(self.audio_dir / "emotion_ref.wav")}),
            ("text", {"emotion_text": "高兴的"})
        ]

        for method, extra_params in emotion_configs:
            with self.subTest(emotion_method=method):
                output_path = self.outputs_dir / f"emotion_{method}.wav"

                # 为参考音频创建文件
                if method == "reference":
                    emotion_audio_path = Path(extra_params["emotion_reference_audio"])
                    emotion_audio_path.parent.mkdir(parents=True, exist_ok=True)
                    emotion_audio_path.touch()

                # 创建合成参数
                params = SynthesisParams(
                    text=text,
                    voice_id=voice.voice_id,
                    emotion_mode=method,
                    emotion_weight=0.8,
                    **extra_params
                )

                result = synthesizer.synthesize(params, str(output_path))

                self.assertTrue(result.success, f"情感控制 {method} 失败: {result.error_message}")
                if result.audio_path:
                    self.assertTrue(Path(result.audio_path).exists())

    def test_batch_synthesis(self):
        """测试批量合成"""
        logger.info("测试批量合成")

        voice = self._create_test_voice("批量合成测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        # 准备批量文本
        texts = [
            "第一段测试文本。",
            "第二段测试文本，内容稍长一些。",
            "第三段测试文本，包含更多的内容和标点符号！",
            "第四段测试文本？这是一个问句。",
            "第五段测试文本，最后一段。"
        ]

        # 执行批量合成（手动实现批量处理）
        start_time = time.time()
        results = []

        for i, text in enumerate(texts):
            output_path = self.outputs_dir / "batch" / f"batch_test_{i:03d}.wav"
            params = SynthesisParams(
                text=text,
                voice_id=voice.voice_id,
                emotion_mode="speaker"
            )
            result = synthesizer.synthesize(params, str(output_path))
            results.append(result)

        batch_time = time.time() - start_time

        # 验证结果
        self.assertEqual(len(results), len(texts))

        for i, result in enumerate(results):
            self.assertTrue(result.success, f"批量合成第{i+1}项失败: {result.error_message}")
            if result.audio_path:
                self.assertTrue(Path(result.audio_path).exists())

        logger.info(f"批量合成测试完成，{len(texts)}个文本，总耗时: {batch_time:.2f}s")

    def test_synthesis_parameters(self):
        """测试合成参数"""
        logger.info("测试合成参数")

        voice = self._create_test_voice("参数测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        text = "这是参数测试文本。"

        # 测试不同的IndexTTS参数
        param_configs = [
            {"speed": 0.8, "temperature": 0.6},
            {"speed": 1.2, "temperature": 0.9},
            {"top_k": 30, "top_p": 0.8},
            {"max_text_tokens_per_segment": 80}
        ]

        for i, param_dict in enumerate(param_configs):
            with self.subTest(params=param_dict):
                output_path = self.outputs_dir / f"params_{i}.wav"

                # 创建合成参数
                params = SynthesisParams(
                    text=text,
                    voice_id=voice.voice_id,
                    emotion_mode="speaker",
                    **param_dict
                )

                result = synthesizer.synthesize(params, str(output_path))

                self.assertTrue(result.success)
                if result.audio_path:
                    self.assertTrue(Path(result.audio_path).exists())

    def test_long_text_synthesis(self):
        """测试长文本合成"""
        logger.info("测试长文本合成")

        voice = self._create_test_voice("长文本测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        # 创建长文本
        long_text = "这是一个很长的测试文本。" * 50  # 重复50次

        output_path = self.outputs_dir / "long_text.wav"

        # 创建合成参数
        params = SynthesisParams(
            text=long_text,
            voice_id=voice.voice_id,
            emotion_mode="speaker",
            max_text_tokens_per_segment=100  # 限制分段大小
        )

        start_time = time.time()
        result = synthesizer.synthesize(params, str(output_path))
        long_text_time = time.time() - start_time

        # 验证结果
        self.assertTrue(result.success)
        self.assertTrue(Path(result.audio_path).exists())
        self.assertGreater(result.segments_count, 1)  # 应该被分段

        logger.info(f"长文本合成测试完成，文本长度: {len(long_text)}, 分段数: {result.segments_count}, 耗时: {long_text_time:.2f}s")

    def test_synthesis_error_handling(self):
        """测试合成错误处理"""
        logger.info("测试合成错误处理")

        voice = self._create_test_voice("错误处理测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        # 测试不存在的音色
        params = SynthesisParams(
            text="测试文本",
            voice_id="non_existent_voice",
            emotion_mode="speaker"
        )
        result = synthesizer.synthesize(params, str(self.outputs_dir / "error_test.wav"))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

        # 测试空文本
        params = SynthesisParams(
            text="",
            voice_id=voice.voice_id,
            emotion_mode="speaker"
        )
        result = synthesizer.synthesize(params, str(self.outputs_dir / "empty_text.wav"))

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)

        # 测试无效的输出路径
        params = SynthesisParams(
            text="测试文本",
            voice_id=voice.voice_id,
            emotion_mode="speaker"
        )
        result = synthesizer.synthesize(params, "/invalid/path/test.wav")

        # 这个可能成功也可能失败，取决于模拟实现
        # 主要是确保不会崩溃

    def test_synthesis_with_speaker_mixing(self):
        """测试说话人混合合成"""
        logger.info("测试说话人混合合成")

        # 创建多说话人音色
        voice = self._create_multi_speaker_voice("混合说话人测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        text = "这是说话人混合的语音合成测试。"
        output_path = self.outputs_dir / "speaker_mixing.wav"

        # 创建合成参数
        params = SynthesisParams(
            text=text,
            voice_id=voice.voice_id,
            emotion_mode="speaker"
        )

        result = synthesizer.synthesize(params, str(output_path))

        # 验证结果
        self.assertTrue(result.success)
        if result.audio_path:
            self.assertTrue(Path(result.audio_path).exists())

        # 验证IndexTTS集成被正确调用
        self.index_tts_integration.infer.assert_called()

    def test_audio_quality_validation(self):
        """测试音频质量验证"""
        logger.info("测试音频质量验证")

        voice = self._create_test_voice("质量验证测试音色")
        self.voice_manager.save_voice(voice)

        synthesizer = VoiceSynthesizer(
            voice_manager=self.voice_manager,
            index_tts_integration=self.index_tts_integration
        )

        text = "这是音频质量验证测试。"
        output_path = self.outputs_dir / "quality_test.wav"

        # 创建合成参数
        params = SynthesisParams(
            text=text,
            voice_id=voice.voice_id,
            emotion_mode="speaker",
            normalize_audio=True,
            trim_silence=True,
            apply_fade=True
        )

        result = synthesizer.synthesize(params, str(output_path))

        # 验证结果
        self.assertTrue(result.success)

        # 验证音频数据
        if result.audio_data:
            sample_rate, audio_data = result.audio_data
            self.assertGreater(sample_rate, 0)
            self.assertGreater(len(audio_data), 0)

    def _create_test_voice(self, name: str) -> VoiceConfig:
        """创建测试音色"""
        ddsp_config = DDSPSVCConfig(
            model_path="test_model.pth",
            config_path="test_config.yaml",
            speaker_id=1,
            f0_predictor="rmvpe",
            f0_min=50.0,
            f0_max=1100.0,
            threhold=-60.0
        )

        index_config = IndexTTSConfig(
            model_path="test_index_model",
            config_path="test_index_config.yaml",
            speaker_name="test_speaker",
            emotion_strength=0.65,
            speed=1.0,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )

        return VoiceConfig(
            name=name,
            description=f"测试音色: {name}",
            tags=["测试", "合成"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )

    def _create_multi_speaker_voice(self, name: str) -> VoiceConfig:
        """创建多说话人测试音色"""
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
            speaker_name="mixed_speaker"
        )

        return VoiceConfig(
            name=name,
            description=f"多说话人测试音色: {name}",
            tags=["测试", "多说话人"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )


def run_voice_synthesis_tests():
    """运行音色合成集成测试"""
    test_suite = unittest.TestSuite()

    # 添加测试方法
    test_methods = [
        'test_basic_synthesis_workflow',
        'test_emotion_control_synthesis',
        'test_batch_synthesis',
        'test_synthesis_parameters',
        'test_long_text_synthesis',
        'test_synthesis_error_handling',
        'test_synthesis_with_speaker_mixing',
        'test_audio_quality_validation'
    ]

    for method in test_methods:
        test_suite.addTest(VoiceSynthesisIntegrationTest(method))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_voice_synthesis_tests()
    sys.exit(0 if success else 1)
