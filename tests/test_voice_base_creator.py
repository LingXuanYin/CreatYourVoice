"""角色声音基底创建器测试

测试角色声音基底创建的完整工作流。
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.voice_base_creator import (
    VoiceBaseCreator, VoiceBaseCreationParams, VoiceBaseCreationResult
)
from src.core.voice_preset_manager import VoicePresetManager
from src.core.voice_manager import VoiceManager
from src.integrations.ddsp_svc import DDSPSVCIntegration
from src.integrations.index_tts import IndexTTSIntegration


class TestVoiceBaseCreator:
    """角色声音基底创建器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())

        # 创建模拟组件
        self.mock_preset_manager = Mock(spec=VoicePresetManager)
        self.mock_voice_manager = Mock(spec=VoiceManager)
        self.mock_ddsp_integration = Mock(spec=DDSPSVCIntegration)
        self.mock_index_tts_integration = Mock(spec=IndexTTSIntegration)

        self.creator = VoiceBaseCreator(
            preset_manager=self.mock_preset_manager,
            voice_manager=self.mock_voice_manager,
            ddsp_integration=self.mock_ddsp_integration,
            index_tts_integration=self.mock_index_tts_integration,
            temp_dir=self.temp_dir
        )

    def teardown_method(self):
        """测试后清理"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_get_available_tags(self):
        """测试获取可用音色标签"""
        # 模拟返回数据
        mock_tags = {
            "青年男": Mock(name="青年男", description="青年男性音色"),
            "青年女": Mock(name="青年女", description="青年女性音色")
        }
        self.mock_preset_manager.get_voice_tags.return_value = mock_tags

        # 执行测试
        result = self.creator.get_available_tags()

        # 验证结果
        assert result == mock_tags
        self.mock_preset_manager.get_voice_tags.assert_called_once()

    def test_validate_parameters_success(self):
        """测试参数验证成功"""
        # 准备测试数据
        from src.core.voice_preset_manager import VoiceTagInfo, SpeakerInfo

        mock_speaker = SpeakerInfo(
            id="test_speaker",
            name="测试说话人",
            model_path="test.pth",
            config_path="test.yaml"
        )

        mock_tag_info = VoiceTagInfo(
            name="青年男",
            description="青年男性音色",
            audio_file="test.wav",
            f0_range=[80, 250],
            speakers=[mock_speaker],
            default_ddsp_params={}
        )

        self.mock_preset_manager.get_voice_tag.return_value = mock_tag_info

        params = VoiceBaseCreationParams(
            voice_name="测试角色",
            selected_tag="青年男",
            speaker_weights={"test_speaker": 1.0}
        )

        # 执行测试（不应该抛出异常）
        self.creator.validate_parameters(params)

    def test_validate_parameters_empty_name(self):
        """测试参数验证 - 空名称"""
        params = VoiceBaseCreationParams(
            voice_name="",
            selected_tag="青年男",
            speaker_weights={"test_speaker": 1.0}
        )

        with pytest.raises(Exception) as exc_info:
            self.creator.validate_parameters(params)

        assert "音色名称不能为空" in str(exc_info.value)

    def test_validate_parameters_no_tag(self):
        """测试参数验证 - 未选择标签"""
        params = VoiceBaseCreationParams(
            voice_name="测试角色",
            selected_tag="",
            speaker_weights={"test_speaker": 1.0}
        )

        with pytest.raises(Exception) as exc_info:
            self.creator.validate_parameters(params)

        assert "必须选择音色标签" in str(exc_info.value)

    def test_validate_parameters_no_speakers(self):
        """测试参数验证 - 未选择说话人"""
        params = VoiceBaseCreationParams(
            voice_name="测试角色",
            selected_tag="青年男",
            speaker_weights={}
        )

        with pytest.raises(Exception) as exc_info:
            self.creator.validate_parameters(params)

        assert "必须选择至少一个说话人" in str(exc_info.value)

    @patch('src.core.voice_base_creator.time.time')
    def test_create_voice_base_success(self, mock_time):
        """测试创建角色声音基底成功"""
        # 模拟时间
        mock_time.side_effect = [1000.0, 1010.0]  # 开始和结束时间

        # 准备测试数据
        from src.core.voice_preset_manager import VoiceTagInfo, SpeakerInfo
        from src.integrations.ddsp_svc import DDSPSVCResult
        from src.integrations.index_tts import IndexTTSResult

        mock_speaker = SpeakerInfo(
            id="test_speaker",
            name="测试说话人",
            model_path="test.pth",
            config_path="test.yaml"
        )

        mock_tag_info = VoiceTagInfo(
            name="青年男",
            description="青年男性音色",
            audio_file="test.wav",
            f0_range=[80, 250],
            speakers=[mock_speaker],
            default_ddsp_params={"f0_min": 80.0, "f0_max": 250.0}
        )

        # 模拟DDSP结果
        mock_ddsp_result = DDSPSVCResult(
            audio=Mock(),
            sample_rate=22050,
            processing_time=5.0,
            segments_count=3
        )

        # 模拟TTS结果
        mock_tts_result = IndexTTSResult(
            audio_path="test_output.wav",
            audio_data=None,
            processing_time=3.0,
            segments_count=2
        )

        # 设置模拟返回值
        self.mock_preset_manager.get_voice_tag.return_value = mock_tag_info
        self.mock_preset_manager.get_audio_file_path.return_value = Path("test_input.wav")
        self.mock_ddsp_integration.infer.return_value = mock_ddsp_result
        self.mock_index_tts_integration.infer.return_value = mock_tts_result

        # 创建参数
        params = VoiceBaseCreationParams(
            voice_name="测试角色",
            selected_tag="青年男",
            speaker_weights={"test_speaker": 1.0},
            preview_text="测试文本"
        )

        # 执行测试
        result = self.creator.create_voice_base(params)

        # 验证结果
        assert result.success is True
        assert result.processing_time == 10.0
        assert result.ddsp_result == mock_ddsp_result
        assert result.tts_result == mock_tts_result
        assert result.voice_config is not None
        assert result.voice_config.name == "测试角色"

    def test_create_voice_base_validation_error(self):
        """测试创建角色声音基底 - 参数验证失败"""
        # 创建无效参数
        params = VoiceBaseCreationParams(
            voice_name="",  # 空名称
            selected_tag="青年男",
            speaker_weights={"test_speaker": 1.0}
        )

        # 执行测试
        result = self.creator.create_voice_base(params)

        # 验证结果
        assert result.success is False
        assert "音色名称不能为空" in result.error_message

    def test_save_voice_base(self):
        """测试保存角色声音基底"""
        from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig

        # 创建测试音色配置
        voice_config = VoiceConfig(
            name="测试角色",
            ddsp_config=DDSPSVCConfig(
                model_path="test.pth",
                config_path="test.yaml"
            ),
            index_tts_config=IndexTTSConfig(
                model_path="checkpoints",
                config_path="config.yaml"
            )
        )

        # 执行测试
        self.creator.save_voice_base(voice_config)

        # 验证调用
        self.mock_voice_manager.save_voice.assert_called_once_with(voice_config)

    def test_cleanup_temp_files(self):
        """测试清理临时文件"""
        # 创建一些临时文件
        test_file = self.temp_dir / "test.txt"
        test_file.write_text("test")

        assert test_file.exists()

        # 执行清理
        self.creator.cleanup_temp_files()

        # 验证文件被清理
        assert not test_file.exists()
        assert self.temp_dir.exists()  # 目录本身应该重新创建


class TestVoiceBaseCreationParams:
    """角色声音基底创建参数测试类"""

    def test_default_values(self):
        """测试默认值"""
        params = VoiceBaseCreationParams(voice_name="测试")

        assert params.voice_name == "测试"
        assert params.description == ""
        assert params.tags == []
        assert params.selected_tag == ""
        assert params.pitch_shift == 0.0
        assert params.formant_shift == 0.0
        assert params.vocal_register_shift == 0.0
        assert params.speaker_weights == {}
        assert params.preview_text == "你好，我是一个新的音色角色。"
        assert params.emotion_control == "speaker"
        assert params.emotion_weight == 0.65

    def test_custom_values(self):
        """测试自定义值"""
        params = VoiceBaseCreationParams(
            voice_name="自定义角色",
            description="自定义描述",
            tags=["标签1", "标签2"],
            selected_tag="青年男",
            pitch_shift=2.0,
            formant_shift=-1.0,
            vocal_register_shift=0.5,
            speaker_weights={"speaker1": 0.6, "speaker2": 0.4},
            preview_text="自定义预览文本",
            emotion_control="vector",
            emotion_weight=0.8
        )

        assert params.voice_name == "自定义角色"
        assert params.description == "自定义描述"
        assert params.tags == ["标签1", "标签2"]
        assert params.selected_tag == "青年男"
        assert params.pitch_shift == 2.0
        assert params.formant_shift == -1.0
        assert params.vocal_register_shift == 0.5
        assert params.speaker_weights == {"speaker1": 0.6, "speaker2": 0.4}
        assert params.preview_text == "自定义预览文本"
        assert params.emotion_control == "vector"
        assert params.emotion_weight == 0.8


if __name__ == "__main__":
    pytest.main([__file__])
