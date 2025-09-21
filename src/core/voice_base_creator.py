"""角色声音基底创建器

这个模块实现角色声音基底创建的完整工作流。
设计原则：
1. 工作流驱动 - 按照用户描述的步骤顺序执行
2. 状态管理 - 支持迭代调整和实时预览
3. 错误恢复 - 每个步骤都有完善的错误处理
"""

import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
import logging

from .models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig, WeightInfo
from .voice_preset_manager import VoicePresetManager, VoiceTagInfo, SpeakerInfo
from .voice_manager import VoiceManager
from .weight_calculator import WeightCalculator
from ..integrations.ddsp_svc import DDSPSVCIntegration, DDSPSVCResult
from ..integrations.index_tts import IndexTTSIntegration, IndexTTSResult

logger = logging.getLogger(__name__)


@dataclass
class VoiceBaseCreationParams:
    """角色声音基底创建参数"""
    # 基本信息
    voice_name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 音色标签选择
    selected_tag: str = ""

    # DDSP-SVC变声器参数
    pitch_shift: float = 0.0  # 音调偏移（半音）
    formant_shift: float = 0.0  # 共振峰偏移（半音）
    vocal_register_shift: float = 0.0  # 声域偏移（半音）

    # 说话人选择和权重
    speaker_weights: Dict[str, float] = field(default_factory=dict)

    # IndexTTS参数（用于预览）
    preview_text: str = "你好，我是一个新的音色角色。"
    emotion_control: str = "speaker"  # speaker, reference, vector, text
    emotion_weight: float = 0.65


@dataclass
class VoiceBaseCreationResult:
    """角色声音基底创建结果"""
    voice_config: Optional[VoiceConfig] = None
    preview_audio_path: Optional[str] = None
    ddsp_result: Optional[DDSPSVCResult] = None
    tts_result: Optional[IndexTTSResult] = None
    processing_time: float = 0.0
    error_message: str = ""
    success: bool = False


class VoiceBaseCreatorError(Exception):
    """角色声音基底创建器异常基类"""
    pass


class InvalidParametersError(VoiceBaseCreatorError):
    """参数无效异常"""
    pass


class ProcessingError(VoiceBaseCreatorError):
    """处理异常"""
    pass


class VoiceBaseCreator:
    """角色声音基底创建器

    实现完整的角色声音基底创建工作流：
    1. 选择预设音色标签
    2. 配置DDSP-SVC变声器参数
    3. 选择说话人和权重
    4. 生成转换音频
    5. IndexTTS预览
    6. 迭代调整
    7. 保存配置
    """

    def __init__(
        self,
        preset_manager: Optional[VoicePresetManager] = None,
        voice_manager: Optional[VoiceManager] = None,
        ddsp_integration: Optional[DDSPSVCIntegration] = None,
        index_tts_integration: Optional[IndexTTSIntegration] = None,
        temp_dir: Optional[Union[str, Path]] = None
    ):
        """初始化角色声音基底创建器

        Args:
            preset_manager: 预设管理器
            voice_manager: 音色管理器
            ddsp_integration: DDSP-SVC集成
            index_tts_integration: IndexTTS集成
            temp_dir: 临时文件目录
        """
        self.preset_manager = preset_manager or VoicePresetManager()
        self.voice_manager = voice_manager or VoiceManager()
        self.ddsp_integration = ddsp_integration or DDSPSVCIntegration()
        self.index_tts_integration = index_tts_integration or IndexTTSIntegration()

        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "voice_creation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 当前创建状态
        self._current_params: Optional[VoiceBaseCreationParams] = None
        self._current_tag_info: Optional[VoiceTagInfo] = None
        self._current_speakers: List[SpeakerInfo] = []

        logger.info("角色声音基底创建器初始化完成")

    def get_available_tags(self) -> Dict[str, VoiceTagInfo]:
        """获取可用的音色标签

        Returns:
            音色标签字典
        """
        return self.preset_manager.get_voice_tags()

    def select_voice_tag(self, tag_name: str) -> VoiceTagInfo:
        """选择音色标签

        Args:
            tag_name: 音色标签名称

        Returns:
            音色标签信息

        Raises:
            InvalidParametersError: 标签不存在
        """
        tag_info = self.preset_manager.get_voice_tag(tag_name)
        if not tag_info:
            raise InvalidParametersError(f"音色标签不存在: {tag_name}")

        self._current_tag_info = tag_info
        self._current_speakers = tag_info.speakers

        logger.info(f"选择音色标签: {tag_name}, 包含 {len(self._current_speakers)} 个说话人")
        return tag_info

    def get_speakers_for_tag(self, tag_name: str) -> List[SpeakerInfo]:
        """获取指定标签的说话人列表

        Args:
            tag_name: 音色标签名称

        Returns:
            说话人信息列表
        """
        return self.preset_manager.get_speakers_by_tag(tag_name)

    def validate_parameters(self, params: VoiceBaseCreationParams) -> None:
        """验证创建参数

        Args:
            params: 创建参数

        Raises:
            InvalidParametersError: 参数无效
        """
        if not params.voice_name.strip():
            raise InvalidParametersError("音色名称不能为空")

        if not params.selected_tag:
            raise InvalidParametersError("必须选择音色标签")

        tag_info = self.preset_manager.get_voice_tag(params.selected_tag)
        if not tag_info:
            raise InvalidParametersError(f"音色标签不存在: {params.selected_tag}")

        if not params.speaker_weights:
            raise InvalidParametersError("必须选择至少一个说话人")

        # 验证说话人ID是否有效
        available_speakers = {s.id for s in tag_info.speakers}
        for speaker_id in params.speaker_weights.keys():
            if speaker_id not in available_speakers:
                raise InvalidParametersError(f"说话人ID无效: {speaker_id}")

        # 验证权重
        for speaker_id, weight in params.speaker_weights.items():
            if weight <= 0:
                raise InvalidParametersError(f"说话人权重必须大于0: {speaker_id}")

    def create_voice_base(
        self,
        params: VoiceBaseCreationParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> VoiceBaseCreationResult:
        """创建角色声音基底

        Args:
            params: 创建参数
            progress_callback: 进度回调函数

        Returns:
            创建结果
        """
        start_time = time.time()
        result = VoiceBaseCreationResult()

        try:
            # 验证参数
            if progress_callback:
                progress_callback(0.1, "验证参数...")
            self.validate_parameters(params)
            self._current_params = params

            # 选择音色标签
            if progress_callback:
                progress_callback(0.2, "加载音色标签...")
            tag_info = self.select_voice_tag(params.selected_tag)

            # 准备DDSP-SVC配置
            if progress_callback:
                progress_callback(0.3, "准备DDSP-SVC配置...")
            ddsp_config = self._create_ddsp_config(tag_info, params)

            # 执行DDSP-SVC变声
            if progress_callback:
                progress_callback(0.4, "执行DDSP-SVC变声...")
            ddsp_result = self._perform_ddsp_conversion(tag_info, params, ddsp_config)
            result.ddsp_result = ddsp_result

            # 准备IndexTTS配置
            if progress_callback:
                progress_callback(0.7, "准备IndexTTS配置...")
            index_tts_config = self._create_index_tts_config(params)

            # 执行IndexTTS预览
            if progress_callback:
                progress_callback(0.8, "生成音色预览...")
            tts_result = self._perform_tts_preview(ddsp_result, params, index_tts_config)
            result.tts_result = tts_result
            result.preview_audio_path = tts_result.audio_path

            # 创建音色配置
            if progress_callback:
                progress_callback(0.9, "创建音色配置...")
            voice_config = self._create_voice_config(params, ddsp_config, index_tts_config)
            result.voice_config = voice_config

            if progress_callback:
                progress_callback(1.0, "完成！")

            result.success = True
            result.processing_time = time.time() - start_time

            logger.info(f"角色声音基底创建成功: {params.voice_name}, 耗时: {result.processing_time:.2f}s")

        except Exception as e:
            result.error_message = str(e)
            result.success = False
            result.processing_time = time.time() - start_time
            logger.error(f"角色声音基底创建失败: {e}")

        return result

    def _create_ddsp_config(self, tag_info: VoiceTagInfo, params: VoiceBaseCreationParams) -> DDSPSVCConfig:
        """创建DDSP-SVC配置"""
        # 获取默认参数
        default_params = tag_info.default_ddsp_params

        # 选择主要说话人模型（权重最大的）
        main_speaker_id = max(params.speaker_weights.keys(), key=lambda k: params.speaker_weights[k])
        main_speaker = self.preset_manager.get_speaker(main_speaker_id)

        if not main_speaker:
            raise ProcessingError(f"找不到主要说话人: {main_speaker_id}")

        # 创建说话人混合字典
        normalized_weights = WeightCalculator.normalize_weights(params.speaker_weights)
        spk_mix_dict = normalized_weights.normalized_weights

        return DDSPSVCConfig(
            model_path=main_speaker.model_path,
            config_path=main_speaker.config_path,
            speaker_id=main_speaker.speaker_id,
            f0_predictor=default_params.get("f0_predictor", "rmvpe"),
            f0_min=default_params.get("f0_min", 50.0),
            f0_max=default_params.get("f0_max", 1100.0),
            threhold=default_params.get("threhold", -60.0),
            spk_mix_dict=spk_mix_dict,
            use_spk_mix=len(spk_mix_dict) > 1
        )

    def _perform_ddsp_conversion(
        self,
        tag_info: VoiceTagInfo,
        params: VoiceBaseCreationParams,
        ddsp_config: DDSPSVCConfig
    ) -> DDSPSVCResult:
        """执行DDSP-SVC变声"""
        # 获取输入音频文件
        audio_path = self.preset_manager.get_audio_file_path(params.selected_tag)
        if not audio_path or not audio_path.exists():
            raise ProcessingError(f"音色标签音频文件不存在: {audio_path}")

        # 加载DDSP-SVC模型
        self.ddsp_integration.load_model(ddsp_config.model_path)

        # 执行推理
        result = self.ddsp_integration.infer(
            audio=str(audio_path),
            speaker_id=ddsp_config.speaker_id,
            spk_mix_dict=ddsp_config.spk_mix_dict if ddsp_config.use_spk_mix else None,
            f0_predictor=ddsp_config.f0_predictor,
            f0_min=ddsp_config.f0_min,
            f0_max=ddsp_config.f0_max,
            threshold=ddsp_config.threhold,
            key_shift=params.pitch_shift,
            formant_shift=params.formant_shift,
            vocal_register_shift=params.vocal_register_shift
        )

        # 保存转换后的音频
        output_path = self.temp_dir / f"ddsp_output_{int(time.time())}.wav"
        self.ddsp_integration.save_audio(result, output_path)

        return result

    def _create_index_tts_config(self, params: VoiceBaseCreationParams) -> IndexTTSConfig:
        """创建IndexTTS配置"""
        return IndexTTSConfig(
            model_path="checkpoints",  # 使用默认模型路径
            config_path="checkpoints/config.yaml",
            speaker_name="default",
            emotion_strength=params.emotion_weight,
            speed=1.0,
            temperature=0.7
        )

    def _perform_tts_preview(
        self,
        ddsp_result: DDSPSVCResult,
        params: VoiceBaseCreationParams,
        index_tts_config: IndexTTSConfig
    ) -> IndexTTSResult:
        """执行IndexTTS预览"""
        # 保存DDSP结果作为说话人音频
        speaker_audio_path = self.temp_dir / f"speaker_audio_{int(time.time())}.wav"
        self.ddsp_integration.save_audio(ddsp_result, speaker_audio_path)

        # 生成预览音频
        preview_output_path = self.temp_dir / f"preview_{int(time.time())}.wav"

        # 加载IndexTTS模型
        self.index_tts_integration.load_model()

        # 执行TTS
        result = self.index_tts_integration.infer(
            text=params.preview_text,
            speaker_audio=str(speaker_audio_path),
            output_path=str(preview_output_path),
            emotion_control_method=params.emotion_control,
            emotion_weight=params.emotion_weight
        )

        return result

    def _create_voice_config(
        self,
        params: VoiceBaseCreationParams,
        ddsp_config: DDSPSVCConfig,
        index_tts_config: IndexTTSConfig
    ) -> VoiceConfig:
        """创建音色配置"""
        # 创建权重信息
        weight_info = WeightInfo(speaker_weights=params.speaker_weights)

        # 创建音色配置
        voice_config = VoiceConfig(
            name=params.voice_name,
            description=params.description,
            tags=params.tags + [params.selected_tag],
            ddsp_config=ddsp_config,
            index_tts_config=index_tts_config,
            weight_info=weight_info
        )

        return voice_config

    def save_voice_base(self, voice_config: VoiceConfig) -> None:
        """保存角色声音基底

        Args:
            voice_config: 音色配置
        """
        self.voice_manager.save_voice(voice_config)
        logger.info(f"角色声音基底保存成功: {voice_config.name}")

    def preview_with_different_text(
        self,
        voice_config: VoiceConfig,
        text: str,
        emotion_control: str = "speaker",
        emotion_weight: float = 0.65
    ) -> IndexTTSResult:
        """使用不同文本预览音色

        Args:
            voice_config: 音色配置
            text: 预览文本
            emotion_control: 情感控制方式
            emotion_weight: 情感权重

        Returns:
            TTS结果
        """
        # 需要重新生成说话人音频
        # 这里简化处理，实际应该缓存DDSP结果
        tag_info = self.preset_manager.get_voice_tag(voice_config.tags[-1])  # 假设最后一个标签是音色标签
        if not tag_info:
            raise ProcessingError("无法找到对应的音色标签")

        audio_path = self.preset_manager.get_audio_file_path(voice_config.tags[-1])
        if not audio_path:
            raise ProcessingError("无法找到音色标签音频文件")

        # 使用现有配置执行DDSP转换
        self.ddsp_integration.load_model(voice_config.ddsp_config.model_path)
        ddsp_result = self.ddsp_integration.infer(
            audio=str(audio_path),
            speaker_id=voice_config.ddsp_config.speaker_id,
            spk_mix_dict=voice_config.ddsp_config.spk_mix_dict if voice_config.ddsp_config.use_spk_mix else None,
            f0_predictor=voice_config.ddsp_config.f0_predictor,
            f0_min=voice_config.ddsp_config.f0_min,
            f0_max=voice_config.ddsp_config.f0_max,
            threshold=voice_config.ddsp_config.threhold
        )

        # 保存为临时说话人音频
        speaker_audio_path = self.temp_dir / f"temp_speaker_{int(time.time())}.wav"
        self.ddsp_integration.save_audio(ddsp_result, speaker_audio_path)

        # 执行TTS
        preview_output_path = self.temp_dir / f"temp_preview_{int(time.time())}.wav"
        result = self.index_tts_integration.infer(
            text=text,
            speaker_audio=str(speaker_audio_path),
            output_path=str(preview_output_path),
            emotion_control_method=emotion_control,
            emotion_weight=emotion_weight
        )

        return result

    def get_test_texts(self) -> List[str]:
        """获取测试文本列表

        Returns:
            测试文本列表
        """
        return self.preset_manager.get_test_texts()

    def cleanup_temp_files(self) -> None:
        """清理临时文件"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info("临时文件已清理")
