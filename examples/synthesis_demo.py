"""语音合成系统演示

这个文件展示了如何使用完整的语音合成系统，包括：
1. 音色管理
2. 情感控制
3. 语音合成
4. 历史记录
5. Web界面启动
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.voice_manager import VoiceManager
from src.core.emotion_manager import EmotionManager, EmotionVector
from src.core.voice_synthesizer import VoiceSynthesizer, SynthesisParams
from src.core.synthesis_history import SynthesisHistory
from src.webui.synthesis_tab import create_synthesis_interface
from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_demo_environment():
    """设置演示环境"""
    logger.info("设置演示环境...")

    # 创建必要的目录
    directories = [
        "data/voices",
        "data/synthesis_history",
        "outputs/synthesis",
        "temp/synthesis",
        "cache/emotions"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("演示环境设置完成")


def create_demo_voice():
    """创建演示用音色"""
    logger.info("创建演示音色...")

    # 创建音色管理器
    voice_manager = VoiceManager()

    # 检查是否已有演示音色
    existing_voices = voice_manager.list_voices()
    if existing_voices:
        logger.info(f"发现已有音色: {[v.name for v in existing_voices]}")
        return voice_manager, existing_voices[0]

    # 创建演示音色配置
    demo_voice = VoiceConfig(
        name="演示音色",
        description="用于演示语音合成功能的测试音色",
        tags=["演示", "测试", "demo"],
        ddsp_config=DDSPSVCConfig(
            model_path="models/ddsp_svc/demo_model.pt",
            config_path="models/ddsp_svc/demo_config.yaml",
            speaker_id=0,
            f0_predictor="rmvpe"
        ),
        index_tts_config=IndexTTSConfig(
            model_path="models/index_tts/demo_model",
            config_path="models/index_tts/demo_config.yaml",
            speaker_name="demo_speaker"
        )
    )

    # 保存音色
    voice_manager.save_voice(demo_voice)
    logger.info(f"演示音色创建完成: {demo_voice.name} ({demo_voice.voice_id})")

    return voice_manager, demo_voice


def demo_emotion_management():
    """演示情感管理功能"""
    logger.info("演示情感管理功能...")

    # 创建情感管理器
    emotion_manager = EmotionManager()

    # 列出可用的情感预设
    presets = emotion_manager.list_presets()
    logger.info(f"可用情感预设: {[p.name for p in presets]}")

    # 创建自定义情感向量
    custom_emotion = emotion_manager.create_emotion_vector(
        happy=0.6,
        surprised=0.3,
        calm=0.1,
        normalize=True
    )
    logger.info(f"自定义情感向量: {custom_emotion.to_list()}")

    # 演示情感向量混合
    if presets:
        preset_emotion = presets[0].emotion_vector
        blended_emotion = custom_emotion.blend(preset_emotion, weight=0.5)
        logger.info(f"混合情感向量: {blended_emotion.to_list()}")

    return emotion_manager


def demo_synthesis_workflow(voice_manager, emotion_manager, demo_voice):
    """演示完整的语音合成工作流"""
    logger.info("演示语音合成工作流...")

    # 创建语音合成器
    synthesizer = VoiceSynthesizer(
        voice_manager=voice_manager,
        emotion_manager=emotion_manager
    )

    # 创建历史管理器
    history = SynthesisHistory(emotion_manager=emotion_manager)

    # 演示不同的合成模式
    demo_texts = [
        "欢迎使用语音合成系统！这是一个功能强大的文本转语音工具。",
        "今天天气真好，我感到非常开心和兴奋！",
        "请注意，这只是一个演示，实际使用时需要配置正确的模型路径。"
    ]

    demo_modes = [
        ("speaker", "默认音色情感", {}),
        ("preset", "情感预设模式", {"emotion_preset": "开心"}),
        ("vector", "自定义情感向量", {
            "emotion_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.2]
        })
    ]

    results = []

    for i, (text, (mode, mode_desc, extra_params)) in enumerate(zip(demo_texts, demo_modes)):
        logger.info(f"演示 {i+1}: {mode_desc}")

        # 创建合成参数
        params = SynthesisParams(
            text=text,
            voice_id=demo_voice.voice_id,
            emotion_mode=mode,
            **extra_params
        )

        # 验证参数
        errors = synthesizer.validate_params(params)
        if errors:
            logger.warning(f"参数验证失败: {errors}")
            continue

        # 注意：这里不执行实际合成，因为需要真实的模型文件
        logger.info(f"参数验证通过，可以执行合成")
        logger.info(f"文本: {text[:50]}...")
        logger.info(f"情感模式: {mode}")

        # 模拟合成结果（实际使用时会调用 synthesizer.synthesize(params)）
        # result = synthesizer.synthesize(params)
        # results.append(result)

    return synthesizer, history


def demo_history_management(history):
    """演示历史管理功能"""
    logger.info("演示历史管理功能...")

    # 获取统计信息
    stats = history.get_statistics()
    logger.info(f"历史统计: {stats}")

    # 搜索历史记录
    recent_records = history.get_recent_records(days=7)
    logger.info(f"最近7天的记录数: {len(recent_records)}")

    # 演示搜索功能
    search_results = history.search_records(
        text_pattern="演示",
        limit=10
    )
    logger.info(f"包含'演示'的记录数: {len(search_results)}")


def launch_web_interface():
    """启动Web界面"""
    logger.info("准备启动Web界面...")

    try:
        # 创建所有组件
        voice_manager = VoiceManager()
        emotion_manager = EmotionManager()
        history = SynthesisHistory(emotion_manager=emotion_manager)

        # 创建界面
        interface = create_synthesis_interface(
            voice_manager=voice_manager,
            emotion_manager=emotion_manager,
            history=history
        )

        logger.info("Web界面创建成功")
        logger.info("注意: 需要安装gradio才能启动界面")
        logger.info("安装命令: pip install gradio")

        # 如果gradio可用，启动界面
        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                debug=True
            )
        except ImportError:
            logger.warning("Gradio未安装，无法启动Web界面")
            logger.info("请运行: pip install gradio")

    except Exception as e:
        logger.error(f"启动Web界面失败: {e}")


def main():
    """主函数"""
    logger.info("🎤 语音合成系统演示开始")

    try:
        # 1. 设置环境
        setup_demo_environment()

        # 2. 创建演示音色
        voice_manager, demo_voice = create_demo_voice()

        # 3. 演示情感管理
        emotion_manager = demo_emotion_management()

        # 4. 演示合成工作流
        synthesizer, history = demo_synthesis_workflow(
            voice_manager, emotion_manager, demo_voice
        )

        # 5. 演示历史管理
        demo_history_management(history)

        # 6. 显示系统信息
        logger.info("\n" + "="*50)
        logger.info("🎉 演示完成！系统组件状态:")
        logger.info(f"✅ 音色管理器: {len(voice_manager.list_voices())} 个音色")
        logger.info(f"✅ 情感管理器: {len(emotion_manager.list_presets())} 个预设")
        logger.info(f"✅ 语音合成器: 已就绪")
        logger.info(f"✅ 历史管理器: 已就绪")
        logger.info("="*50)

        # 7. 询问是否启动Web界面
        try:
            choice = input("\n是否启动Web界面? (y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                launch_web_interface()
            else:
                logger.info("演示结束")
        except KeyboardInterrupt:
            logger.info("\n演示被用户中断")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
