"""语音合成集成测试

测试整个语音合成工作流的集成功能。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.core.voice_synthesizer import VoiceSynthesizer, SynthesisParams, SynthesisResult
from src.core.voice_manager import VoiceManager
from src.core.emotion_manager import EmotionManager, EmotionVector
from src.core.synthesis_history import SynthesisHistory
from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.integrations.index_tts import IndexTTSIntegration, IndexTTSResult


class TestSynthesisIntegration:
    """语音合成集成测试类"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_voice_config(self):
        """模拟音色配置"""
        return VoiceConfig(
            name="测试音色",
            ddsp_config=DDSPSVCConfig(
                model_path="test_model.pt",
                config_path="test_config.yaml"
            ),
            index_tts_config=IndexTTSConfig(
                model_path="test_index_model",
                config_path="test_index_config.yaml"
            ),
            description="测试用音色",
            tags=["测试", "demo"]
        )

    @pytest.fixture
    def voice_manager(self, temp_dir, mock_voice_config):
        """创建音色管理器"""
        manager = VoiceManager(voices_dir=temp_dir / "voices")
        manager.save_voice(mock_voice_config)
        return manager

    @pytest.fixture
    def emotion_manager(self, temp_dir):
        """创建情感管理器"""
        presets_file = temp_dir / "emotion_presets.yaml"
        # 创建简单的预设文件
        presets_content = """
presets:
  - name: "开心"
    description: "快乐的情感"
    emotion_vector: [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]
    tags: ["积极"]
"""
        presets_file.write_text(presets_content, encoding='utf-8')

        return EmotionManager(
            presets_file=presets_file,
            cache_dir=temp_dir / "emotion_cache"
        )

    @pytest.fixture
    def synthesis_history(self, temp_dir):
        """创建合成历史管理器"""
        return SynthesisHistory(history_dir=temp_dir / "history")

    @pytest.fixture
    def mock_index_tts(self):
        """模拟IndexTTS集成"""
        mock_tts = Mock(spec=IndexTTSIntegration)

        # 模拟成功的推理结果
        mock_result = IndexTTSResult(
            audio_path=None,
            audio_data=(22050, np.random.randn(22050)),  # 1秒的随机音频
            processing_time=2.5,
            segments_count=1,
            emotion_info={
                "control_method": "speaker",
                "emotion_weight": 0.65,
                "emotion_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        )

        mock_tts.infer.return_value = mock_result
        return mock_tts

    @pytest.fixture
    def synthesizer(self, voice_manager, emotion_manager, mock_index_tts, temp_dir):
        """创建语音合成器"""
        return VoiceSynthesizer(
            voice_manager=voice_manager,
            emotion_manager=emotion_manager,
            index_tts_integration=mock_index_tts,
            output_dir=temp_dir / "outputs",
            temp_dir=temp_dir / "temp"
        )

    def test_basic_synthesis_workflow(self, synthesizer, mock_voice_config):
        """测试基本的语音合成工作流"""
        # 准备合成参数
        params = SynthesisParams(
            text="这是一个测试文本，用于验证语音合成功能。",
            voice_id=mock_voice_config.voice_id,
            emotion_mode="speaker"
        )

        # 模拟音频文件存在
        with patch('pathlib.Path.exists', return_value=True):
            with patch('src.core.voice_synthesizer.VoiceSynthesizer._get_speaker_audio',
                      return_value="test_audio.wav"):
                # 执行合成
                result = synthesizer.synthesize(params)

        # 验证结果
        assert isinstance(result, SynthesisResult)
        assert result.synthesis_params == params
        assert result.voice_config.voice_id == mock_voice_config.voice_id

    def test_emotion_preset_synthesis(self, synthesizer, mock_voice_config, emotion_manager):
        """测试情感预设模式的语音合成"""
        # 准备合成参数
        params = SynthesisParams(
            text="测试情感预设功能。",
            voice_id=mock_voice_config.voice_id,
            emotion_mode="preset",
            emotion_preset="开心"
        )

        # 模拟音频文件存在
        with patch('pathlib.Path.exists', return_value=True):
            with patch('src.core.voice_synthesizer.VoiceSynthesizer._get_speaker_audio',
                      return_value="test_audio.wav"):
                # 执行合成
                result = synthesizer.synthesize(params)

        # 验证结果
        assert result.synthesis_params.emotion_preset == "开心"
        assert result.final_emotion_vector is not None
        # 验证情感向量是开心预设的向量
        assert result.final_emotion_vector[0] > 0.5  # happy维度应该较高

    def test_emotion_vector_synthesis(self, synthesizer, mock_voice_config):
        """测试情感向量模式的语音合成"""
        # 自定义情感向量
        custom_emotion = [0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.3]

        params = SynthesisParams(
            text="测试自定义情感向量。",
            voice_id=mock_voice_config.voice_id,
            emotion_mode="vector",
            emotion_vector=custom_emotion
        )

        # 模拟音频文件存在
        with patch('pathlib.Path.exists', return_value=True):
            with patch('src.core.voice_synthesizer.VoiceSynthesizer._get_speaker_audio',
                      return_value="test_audio.wav"):
                # 执行合成
                result = synthesizer.synthesize(params)

        # 验证结果
        assert result.final_emotion_vector == custom_emotion

    def test_text_processing(self, synthesizer):
        """测试文本处理功能"""
        # 测试长文本分句
        long_text = "这是第一句话。这是第二句话！这是第三句话？还有更多内容，包含逗号分隔的部分，以及其他标点符号；最后是结尾。"

        segments = synthesizer._process_text(long_text, max_tokens_per_segment=50)

        # 验证分句结果
        assert len(segments) > 1
        assert all(isinstance(segment, str) for segment in segments)
        assert all(len(segment.strip()) > 0 for segment in segments)

    def test_parameter_validation(self, synthesizer, mock_voice_config):
        """测试参数验证功能"""
        # 测试有效参数
        valid_params = SynthesisParams(
            text="有效的测试文本",
            voice_id=mock_voice_config.voice_id,
            emotion_mode="speaker"
        )

        errors = synthesizer.validate_params(valid_params)
        assert len(errors) == 0

        # 测试无效参数
        invalid_params = SynthesisParams(
            text="",  # 空文本
            voice_id="不存在的音色ID",
            emotion_mode="vector",
            emotion_vector=[1.0, 2.0]  # 错误的向量长度
        )

        errors = synthesizer.validate_params(invalid_params)
        assert len(errors) > 0
        assert any("文本不能为空" in error for error in errors)
        assert any("音色不存在" in error for error in errors)
        assert any("情感向量必须包含8个值" in error for error in errors)

    def test_synthesis_history_integration(self, synthesizer, synthesis_history, mock_voice_config):
        """测试合成历史集成功能"""
        # 执行合成
        params = SynthesisParams(
            text="测试历史记录功能",
            voice_id=mock_voice_config.voice_id,
            emotion_mode="speaker"
        )

        with patch('pathlib.Path.exists', return_value=True):
            with patch('src.core.voice_synthesizer.VoiceSynthesizer._get_speaker_audio',
                      return_value="test_audio.wav"):
                result = synthesizer.synthesize(params)

        # 保存到历史
        history_record = synthesis_history.save_record(result)

        # 验证历史记录
        assert history_record.text == params.text
        assert history_record.voice_id == params.voice_id
        assert history_record.emotion_mode == params.emotion_mode

        # 测试历史记录搜索
        records = synthesis_history.search_records(text_pattern="测试历史")
        assert len(records) == 1
        assert records[0].record_id == history_record.record_id

        # 测试参数重建
        rebuilt_params = synthesis_history.recreate_synthesis_params(history_record)
        assert rebuilt_params.text == params.text
        assert rebuilt_params.voice_id == params.voice_id

    def test_emotion_reference_audio_conversion(self, emotion_manager, temp_dir):
        """测试情感参考音频转向量功能"""
        # 创建模拟音频文件
        test_audio = temp_dir / "test_emotion.wav"
        test_audio.touch()

        # 模拟音频分析
        with patch('src.core.emotion_manager.EmotionManager._analyze_audio_simple') as mock_analyze:
            mock_analyze.return_value = [0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]

            # 提取情感向量
            emotion_vector = emotion_manager.extract_emotion_from_audio(test_audio)

        # 验证结果
        assert isinstance(emotion_vector, EmotionVector)
        assert emotion_vector.happy > 0.5  # 应该检测到高兴情感
        assert sum(emotion_vector.to_list()) == pytest.approx(1.0, rel=1e-2)  # 归一化检查

    def test_emotion_text_analysis(self, emotion_manager):
        """测试情感文本分析功能"""
        # 测试包含情感关键词的文本
        happy_text = "我今天非常开心和兴奋！"
        emotion_vector = emotion_manager.analyze_emotion_from_text(happy_text)

        # 验证结果
        assert isinstance(emotion_vector, EmotionVector)
        assert emotion_vector.happy > 0.0  # 应该检测到高兴情感

        # 测试悲伤文本
        sad_text = "我感到很悲伤和难过。"
        emotion_vector = emotion_manager.analyze_emotion_from_text(sad_text)
        assert emotion_vector.sad > 0.0  # 应该检测到悲伤情感

    def test_emotion_vector_operations(self):
        """测试情感向量操作"""
        # 创建情感向量
        vector1 = EmotionVector(happy=0.8, calm=0.2)
        vector2 = EmotionVector(angry=0.6, afraid=0.4)

        # 测试归一化
        normalized = vector1.normalize()
        assert sum(normalized.to_list()) == pytest.approx(1.0, rel=1e-6)

        # 测试混合
        blended = vector1.blend(vector2, weight=0.5)
        assert isinstance(blended, EmotionVector)

        # 测试列表转换
        vector_list = vector1.to_list()
        assert len(vector_list) == 8
        assert vector_list[0] == 0.8  # happy
        assert vector_list[7] == 0.2  # calm

        # 测试从列表创建
        recreated = EmotionVector.from_list(vector_list)
        assert recreated.happy == vector1.happy
        assert recreated.calm == vector1.calm

    def test_error_handling(self, synthesizer, mock_voice_config):
        """测试错误处理"""
        # 测试不存在的音色
        params = SynthesisParams(
            text="测试错误处理",
            voice_id="不存在的音色ID",
            emotion_mode="speaker"
        )

        result = synthesizer.synthesize(params)
        assert not result.success
        assert result.error_message is not None
        assert "音色不存在" in result.error_message

    def test_statistics_generation(self, synthesis_history, mock_voice_config):
        """测试统计信息生成"""
        # 创建一些测试记录
        from src.core.synthesis_history import HistoryRecord
        from datetime import datetime

        records = [
            HistoryRecord(
                record_id="test1",
                synthesis_id="syn1",
                created_at=datetime.now(),
                text="测试文本1",
                voice_id=mock_voice_config.voice_id,
                voice_name=mock_voice_config.name,
                emotion_mode="speaker",
                emotion_vector=[0, 0, 0, 0, 0, 0, 0, 1],
                success=True,
                processing_time=2.5
            ),
            HistoryRecord(
                record_id="test2",
                synthesis_id="syn2",
                created_at=datetime.now(),
                text="测试文本2",
                voice_id=mock_voice_config.voice_id,
                voice_name=mock_voice_config.name,
                emotion_mode="preset",
                emotion_vector=[0.8, 0, 0, 0, 0, 0, 0.1, 0.1],
                success=True,
                processing_time=3.0
            )
        ]

        # 保存记录
        for record in records:
            synthesis_history._save_record_file(record)
            synthesis_history._update_index(record)

        # 获取统计信息
        stats = synthesis_history.get_statistics()

        # 验证统计结果
        assert stats['total_records'] == 2
        assert stats['success_rate'] == 1.0
        assert stats['avg_processing_time'] == 2.75
        assert mock_voice_config.name in [voice[0] for voice in stats['most_used_voices']]
        assert 'speaker' in stats['emotion_mode_distribution']
        assert 'preset' in stats['emotion_mode_distribution']


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
