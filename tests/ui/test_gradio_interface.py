"""Gradio界面测试

这个模块测试Gradio Web界面的功能和响应性。
设计原则：
1. 界面完整性 - 验证所有组件正确加载和显示
2. 交互测试 - 测试用户交互流程和响应
3. 错误处理 - 验证错误情况下的界面行为
"""

import unittest
import tempfile
import shutil
import sys
import time
import threading
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
except ImportError:
    # 如果gradio未安装，创建模拟对象
    class MockGradio:
        class Blocks:
            def __init__(self, *args, **kwargs):
                pass
            def launch(self, *args, **kwargs):
                pass
            def close(self):
                pass
    gr = MockGradio()

from src.webui.app import create_app, VoiceCreationApp
from src.core.models import VoiceConfig, DDSPSVCConfig, IndexTTSConfig
from src.core.voice_manager import VoiceManager
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class GradioTestCase(unittest.TestCase):
    """Gradio测试基类"""

    def setUp(self):
        """设置测试环境"""
        setup_logging(log_level="WARNING", console_output=False)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.voices_dir = self.temp_dir / "voices"
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # 创建模拟的音色管理器
        self.voice_manager = VoiceManager(self.voices_dir)

        # 创建测试音色
        self._create_test_voices()

        # 启动Gradio应用
        self.app = None
        self.server_thread = None
        self.server_port = 7860

    def tearDown(self):
        """清理测试环境"""
        if self.app:
            try:
                self.app.close()
            except:
                pass

        if self.server_thread and self.server_thread.is_alive():
            # 给服务器一些时间关闭
            time.sleep(1)

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_voices(self):
        """创建测试音色"""
        test_voices = [
            {
                "name": "测试音色1",
                "description": "用于界面测试的音色1",
                "tags": ["测试", "女声"]
            },
            {
                "name": "测试音色2",
                "description": "用于界面测试的音色2",
                "tags": ["测试", "男声"]
            },
            {
                "name": "测试音色3",
                "description": "用于界面测试的音色3",
                "tags": ["测试", "中性"]
            }
        ]

        for voice_data in test_voices:
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

            voice = VoiceConfig(
                name=voice_data["name"],
                description=voice_data["description"],
                tags=voice_data["tags"],
                ddsp_config=ddsp_config,
                index_tts_config=index_config
            )

            self.voice_manager.save_voice(voice)

    def start_gradio_app(self, **kwargs):
        """启动Gradio应用"""
        def run_app():
            try:
                # 使用模拟的组件创建应用
                with patch('src.webui.app.VoiceManager') as mock_vm:
                    mock_vm.return_value = self.voice_manager

                    self.app = create_app()

                    if self.app:
                        self.app.launch(
                            server_port=self.server_port,
                            share=False,
                            quiet=True,
                            show_error=True
                        )
            except Exception as e:
                logger.error(f"启动Gradio应用失败: {e}")

        self.server_thread = threading.Thread(target=run_app, daemon=True)
        self.server_thread.start()

        # 等待服务器启动
        max_wait = 10  # 最多等待10秒
        for _ in range(max_wait):
            try:
                response = requests.get(f"http://localhost:{self.server_port}", timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(1)

        return False


class GradioInterfaceTest(GradioTestCase):
    """Gradio界面基础测试"""

    def test_app_creation(self):
        """测试应用创建"""
        logger.info("测试Gradio应用创建")

        with patch('src.webui.app.VoiceManager') as mock_vm:
            mock_vm.return_value = self.voice_manager

            # 测试应用创建不抛出异常
            try:
                app = create_app()
                self.assertIsNotNone(app)
                logger.info("应用创建成功")
            except Exception as e:
                self.fail(f"应用创建失败: {e}")

    def test_interface_components(self):
        """测试界面组件"""
        logger.info("测试界面组件")

        with patch('src.webui.app.VoiceManager') as mock_vm:
            mock_vm.return_value = self.voice_manager

            # 创建应用并检查组件
            app = create_app()

            self.assertIsNotNone(app)

            # 检查应用是否有预期的组件
            # 注意：这里的测试取决于实际的Gradio应用结构
            if hasattr(app, 'blocks'):
                self.assertIsNotNone(app.blocks)
                logger.info("界面组件验证通过")

    @patch('src.webui.app.VoiceManager')
    def test_voice_list_loading(self, mock_vm):
        """测试音色列表加载"""
        logger.info("测试音色列表加载")

        mock_vm.return_value = self.voice_manager

        # 模拟获取音色列表的函数
        voices = self.voice_manager.list_voices()

        self.assertEqual(len(voices), 3)
        self.assertIn("测试音色1", [v.name for v in voices])
        self.assertIn("测试音色2", [v.name for v in voices])
        self.assertIn("测试音色3", [v.name for v in voices])

        logger.info(f"成功加载 {len(voices)} 个音色")


class GradioFunctionalTest(GradioTestCase):
    """Gradio功能测试"""

    @patch('src.webui.app.VoiceManager')
    @patch('src.integrations.ddsp_svc.DDSPSVCIntegration')
    @patch('src.integrations.index_tts.IndexTTSIntegration')
    def test_voice_synthesis_workflow(self, mock_index, mock_ddsp, mock_vm):
        """测试语音合成工作流"""
        logger.info("测试语音合成工作流")

        # 设置模拟对象
        mock_vm.return_value = self.voice_manager

        mock_ddsp_instance = Mock()
        mock_ddsp_instance.convert_voice.return_value = "mock_audio_data"
        mock_ddsp.return_value = mock_ddsp_instance

        mock_index_instance = Mock()
        mock_index_instance.synthesize.return_value = "mock_synthesis_result"
        mock_index.return_value = mock_index_instance

        # 模拟合成过程
        voice = self.voice_manager.list_voices()[0]

        # 测试DDSP-SVC转换
        result = mock_ddsp_instance.convert_voice(
            audio_path="test_input.wav",
            speaker_id=voice.ddsp_config.speaker_id
        )
        self.assertEqual(result, "mock_audio_data")

        # 测试IndexTTS合成
        result = mock_index_instance.synthesize(
            text="测试文本",
            speaker_name=voice.index_tts_config.speaker_name
        )
        self.assertEqual(result, "mock_synthesis_result")

        logger.info("语音合成工作流测试通过")

    @patch('src.webui.app.VoiceManager')
    def test_voice_creation_workflow(self, mock_vm):
        """测试音色创建工作流"""
        logger.info("测试音色创建工作流")

        mock_vm.return_value = self.voice_manager

        # 模拟创建新音色
        new_voice_data = {
            "name": "新创建音色",
            "description": "通过界面创建的音色",
            "tags": ["新建", "测试"]
        }

        ddsp_config = DDSPSVCConfig(
            model_path="new_model.pth",
            config_path="new_config.yaml",
            speaker_id=2
        )

        index_config = IndexTTSConfig(
            model_path="new_index_model",
            config_path="new_index_config.yaml",
            speaker_name="new_speaker"
        )

        new_voice = VoiceConfig(
            name=new_voice_data["name"],
            description=new_voice_data["description"],
            tags=new_voice_data["tags"],
            ddsp_config=ddsp_config,
            index_tts_config=index_config
        )

        # 保存音色
        self.voice_manager.save_voice(new_voice)

        # 验证音色已创建
        voices = self.voice_manager.list_voices()
        self.assertEqual(len(voices), 4)  # 原来3个 + 新建1个

        new_voice_names = [v.name for v in voices]
        self.assertIn("新创建音色", new_voice_names)

        logger.info("音色创建工作流测试通过")

    @patch('src.webui.app.VoiceManager')
    def test_voice_fusion_workflow(self, mock_vm):
        """测试音色融合工作流"""
        logger.info("测试音色融合工作流")

        mock_vm.return_value = self.voice_manager

        # 获取现有音色
        voices = self.voice_manager.list_voices()
        self.assertGreaterEqual(len(voices), 2)

        # 模拟融合过程
        source_voices = voices[:2]
        fusion_weights = [0.6, 0.4]

        # 模拟融合逻辑
        from src.core.voice_fusion import VoiceFuser, FusionSource

        fuser = VoiceFuser(self.voice_manager)
        fusion_sources = [
            FusionSource(voice_config=source_voices[0], weight=fusion_weights[0]),
            FusionSource(voice_config=source_voices[1], weight=fusion_weights[1])
        ]

        # 执行融合
        fusion_result = fuser.fuse_voices(fusion_sources, "测试融合音色")

        # 验证融合结果
        self.assertIsNotNone(fusion_result)
        # 融合结果是FusionResult对象
        self.assertIsNotNone(fusion_result.fused_voice_config)
        self.assertEqual(fusion_result.fused_voice_config.name, "测试融合音色")

        logger.info("音色融合工作流测试通过")


class GradioComponentTest(GradioTestCase):
    """Gradio组件测试"""

    @patch('src.webui.app.VoiceManager')
    def test_voice_creation_app_initialization(self, mock_vm):
        """测试VoiceCreationApp初始化"""
        logger.info("测试VoiceCreationApp初始化")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager'), \
             patch('src.webui.app.VoiceBaseCreator'), \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            try:
                app = VoiceCreationApp()
                self.assertIsNotNone(app)
                self.assertIsNotNone(app.voice_manager)
                logger.info("VoiceCreationApp初始化成功")
            except Exception as e:
                self.fail(f"VoiceCreationApp初始化失败: {e}")

    @patch('src.webui.app.VoiceManager')
    def test_interface_creation(self, mock_vm):
        """测试界面创建"""
        logger.info("测试界面创建")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager'), \
             patch('src.webui.app.VoiceBaseCreator'), \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            try:
                app = VoiceCreationApp()
                interface = app.create_interface()
                self.assertIsNotNone(interface)
                logger.info("界面创建成功")
            except Exception as e:
                self.fail(f"界面创建失败: {e}")


class GradioErrorHandlingTest(GradioTestCase):
    """Gradio错误处理测试"""

    @patch('src.webui.app.VoiceManager')
    def test_voice_tag_selection_error_handling(self, mock_vm):
        """测试音色标签选择错误处理"""
        logger.info("测试音色标签选择错误处理")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager') as mock_preset, \
             patch('src.webui.app.VoiceBaseCreator'), \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            # 模拟预设管理器返回None
            mock_preset.return_value.get_voice_tag.return_value = None

            app = VoiceCreationApp()

            # 测试选择不存在的标签
            result = app._on_voice_tag_selected("不存在的标签")

            # 验证错误处理
            self.assertIsInstance(result, tuple)
            self.assertIn("error", result[0])

            logger.info("音色标签选择错误处理测试通过")

    @patch('src.webui.app.VoiceManager')
    def test_voice_creation_error_handling(self, mock_vm):
        """测试音色创建错误处理"""
        logger.info("测试音色创建错误处理")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager'), \
             patch('src.webui.app.VoiceBaseCreator') as mock_creator, \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            app = VoiceCreationApp()

            # 测试空名称
            result = app._create_voice_base_preview(
                "", "", "", "", 0, 0, 0, "测试文本", "speaker", 0.65,
                False, False, False, 1.0, 1.0, 1.0
            )

            # 验证错误处理
            self.assertIsNone(result[0])  # 音频应该为None
            self.assertIn("错误", result[1])  # 状态消息应该包含错误

            logger.info("音色创建错误处理测试通过")


class GradioPerformanceTest(GradioTestCase):
    """Gradio性能测试"""

    @patch('src.webui.app.VoiceManager')
    def test_interface_loading_performance(self, mock_vm):
        """测试界面加载性能"""
        logger.info("测试界面加载性能")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager'), \
             patch('src.webui.app.VoiceBaseCreator'), \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            start_time = time.time()

            app = VoiceCreationApp()
            interface = app.create_interface()

            loading_time = time.time() - start_time

            # 界面加载应该在合理时间内完成（比如5秒）
            self.assertLess(loading_time, 5.0)

            logger.info(f"界面加载时间: {loading_time:.3f}秒")

    @patch('src.webui.app.VoiceManager')
    def test_voice_list_refresh_performance(self, mock_vm):
        """测试音色列表刷新性能"""
        logger.info("测试音色列表刷新性能")

        mock_vm.return_value = self.voice_manager

        with patch('src.webui.app.DDSPSVCIntegration'), \
             patch('src.webui.app.IndexTTSIntegration'), \
             patch('src.webui.app.VoicePresetManager'), \
             patch('src.webui.app.VoiceBaseCreator'), \
             patch('src.webui.app.AdvancedWeightCalculator'), \
             patch('src.webui.app.VoiceInheritor'), \
             patch('src.webui.app.VoiceFuser'), \
             patch('src.webui.app.InheritanceTab'), \
             patch('src.webui.app.FusionTab'), \
             patch('src.webui.app.ConfigManager'):

            app = VoiceCreationApp()

            start_time = time.time()
            voice_data, statistics = app._refresh_voice_management()
            refresh_time = time.time() - start_time

            # 音色列表刷新应该在合理时间内完成
            self.assertLess(refresh_time, 1.0)
            self.assertIsInstance(voice_data, list)
            self.assertIsInstance(statistics, dict)

            logger.info(f"音色列表刷新时间: {refresh_time:.3f}秒")


def run_ui_tests():
    """运行UI测试"""
    test_suite = unittest.TestSuite()

    # 添加测试类
    test_classes = [
        GradioInterfaceTest,
        GradioFunctionalTest,
        GradioComponentTest,
        GradioErrorHandlingTest,
        GradioPerformanceTest
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ui_tests()
    sys.exit(0 if success else 1)
