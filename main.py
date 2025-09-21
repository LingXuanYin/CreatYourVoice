#!/usr/bin/env python3
"""CreatYourVoice 主启动脚本

这是音色创建和管理系统的主入口点。
设计原则：
1. 统一入口 - 所有功能通过一个脚本启动
2. 配置优先 - 优先使用配置文件，然后是命令行参数
3. 错误恢复 - 提供友好的错误信息和恢复建议
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.config import get_config_manager, ConfigError
    from src.utils.logging_config import setup_logging, get_logger, log_exception
    from src.core.gpu_manager_init import initialize_gpu_management, shutdown_gpu_management, get_gpu_management_status
except ImportError as e:
    print(f"错误：无法导入核心模块: {e}")
    print("请确保项目结构正确且依赖已安装")
    sys.exit(1)


def setup_environment():
    """设置运行环境"""
    try:
        # 加载配置
        config_manager = get_config_manager()
        config = config_manager.load_config(create_if_missing=True)

        # 设置日志
        log_file = None
        log_dir = "logs"
        if config.system.log_file:
            log_file_path = Path(config.system.log_file)
            if log_file_path.is_absolute():
                # 绝对路径，直接使用
                log_file = log_file_path.name
                log_dir = str(log_file_path.parent)
            else:
                # 相对路径，检查是否已包含目录
                if log_file_path.parent != Path("."):
                    log_dir = str(log_file_path.parent)
                    log_file = log_file_path.name
                else:
                    log_file = str(log_file_path)

        setup_logging(
            log_level=config.system.log_level,
            log_file=log_file,
            log_dir=log_dir,
            console_output=True,
            colored_output=True
        )

        # 创建必要目录
        config_manager.create_directories()

        logger = get_logger("main")
        logger.info("=== CreatYourVoice 启动 ===")
        logger.info(f"项目根目录: {PROJECT_ROOT}")
        logger.info(f"配置文件: {config_manager.config_path}")

        return config

    except ConfigError as e:
        print(f"配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"环境设置失败: {e}")
        sys.exit(1)


def check_dependencies():
    """检查依赖库"""
    logger = get_logger("dependencies")

    required_packages = [
        ("torch", "PyTorch"),
        ("librosa", "Librosa"),
        ("numpy", "NumPy"),
        ("soundfile", "SoundFile"),
        ("gradio", "Gradio"),
        ("yaml", "PyYAML")
    ]

    missing_packages = []

    for package, name in required_packages:
        try:
            __import__(package)
            logger.debug(f"✓ {name} 已安装")
        except ImportError:
            missing_packages.append(name)
            logger.warning(f"✗ {name} 未安装")

    if missing_packages:
        logger.error(f"缺少依赖库: {', '.join(missing_packages)}")
        logger.info("请运行以下命令安装依赖:")
        logger.info("pip install -r requirements.txt")
        return False

    logger.info("所有依赖库检查通过")
    return True


def check_models():
    """检查模型文件"""
    logger = get_logger("models")
    config = get_config_manager().get_config()

    # 检查DDSP-SVC模型目录
    ddsp_model_dir = Path(config.ddsp_svc.model_dir)
    if not ddsp_model_dir.exists():
        logger.warning(f"DDSP-SVC模型目录不存在: {ddsp_model_dir}")
        logger.info("请将DDSP-SVC模型文件放置在指定目录")
    else:
        logger.info(f"✓ DDSP-SVC模型目录: {ddsp_model_dir}")

    # 检查IndexTTS模型目录
    index_tts_model_dir = Path(config.index_tts.model_dir)
    if not index_tts_model_dir.exists():
        logger.warning(f"IndexTTS模型目录不存在: {index_tts_model_dir}")
        logger.info("请将IndexTTS模型文件放置在指定目录")
    else:
        logger.info(f"✓ IndexTTS模型目录: {index_tts_model_dir}")

    return True


def start_webui(args):
    """启动Web界面"""
    logger = get_logger("webui")

    try:
        logger.info("启动Web界面...")

        # 检查依赖
        if not check_dependencies():
            logger.error("依赖检查失败，无法启动Web界面")
            return False

        # 检查模型（非强制）
        check_models()

        # 导入并启动Gradio应用
        from src.webui.app import create_app

        config = get_config_manager().get_config()

        # 使用命令行参数覆盖配置
        host = args.host or config.ui.host
        port = args.port or config.ui.port
        share = args.share if args.share is not None else config.ui.share
        debug = args.debug if args.debug is not None else config.ui.debug

        logger.info(f"启动参数: host={host}, port={port}, share={share}, debug={debug}")

        # 创建并启动应用
        interface = create_app()
        interface.launch(
            server_name=host,
            server_port=port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )

        return True

    except ImportError as e:
        logger.error(f"导入Web界面模块失败: {e}")
        logger.info("请确保已安装gradio等Web界面依赖")
        return False
    except Exception as e:
        log_exception(logger, "启动Web界面失败")
        return False


def run_cli_command(args):
    """运行命令行命令"""
    logger = get_logger("cli")

    if args.command == "config":
        # 显示配置信息
        config = get_config_manager().get_config()
        print("=== 当前配置 ===")
        print(f"DDSP-SVC模型目录: {config.ddsp_svc.model_dir}")
        print(f"IndexTTS模型目录: {config.index_tts.model_dir}")
        print(f"音色目录: {config.system.voices_dir}")
        print(f"输出目录: {config.system.outputs_dir}")
        print(f"设备: {config.system.device}")
        print(f"日志级别: {config.system.log_level}")

    elif args.command == "check":
        # 系统检查
        print("=== 系统检查 ===")
        deps_ok = check_dependencies()
        models_ok = check_models()

        if deps_ok and models_ok:
            print("✓ 系统检查通过")
            return True
        else:
            print("✗ 系统检查发现问题")
            return False

    elif args.command == "voices":
        # 音色管理
        try:
            from src.core.voice_manager import VoiceManager
            config = get_config_manager().get_config()
            voice_manager = VoiceManager(config.system.voices_dir)

            voices = voice_manager.list_voices()
            if voices:
                print(f"=== 音色列表 ({len(voices)}个) ===")
                for voice in voices:
                    print(f"- {voice.name} ({voice.voice_id[:8]}...)")
                    print(f"  创建时间: {voice.created_at.strftime('%Y-%m-%d %H:%M')}")
                    print(f"  标签: {', '.join(voice.tags) if voice.tags else '无'}")
                    print()
            else:
                print("暂无音色")

        except Exception as e:
            logger.error(f"音色管理失败: {e}")
            return False

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CreatYourVoice - 音色创建和管理系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py                          # 启动Web界面
  python main.py --host 0.0.0.0 --port 7860  # 指定主机和端口
  python main.py --share                  # 创建公共链接
  python main.py config                   # 显示配置信息
  python main.py check                    # 系统检查
  python main.py voices                   # 查看音色列表
        """
    )

    # 全局参数
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # Web界面命令（默认）
    webui_parser = subparsers.add_parser("webui", help="启动Web界面")
    webui_parser.add_argument("--host", help="主机地址")
    webui_parser.add_argument("--port", type=int, help="端口号")
    webui_parser.add_argument("--share", action="store_true", help="创建公共链接")
    webui_parser.add_argument("--debug", action="store_true", help="调试模式")

    # 配置命令
    config_parser = subparsers.add_parser("config", help="显示配置信息")

    # 检查命令
    check_parser = subparsers.add_parser("check", help="系统检查")

    # 音色管理命令
    voices_parser = subparsers.add_parser("voices", help="音色管理")

    args = parser.parse_args()

    # 设置配置文件路径
    os.environ["CREATYOURVOICE_CONFIG"] = args.config

    try:
        # 设置环境
        config = setup_environment()

        # 应用命令行日志级别覆盖
        if args.log_level:
            from src.utils.logging_config import get_logging_manager
            get_logging_manager().set_level("", args.log_level)

        # 执行命令
        if args.command in ["config", "check", "voices"]:
            success = run_cli_command(args)
            sys.exit(0 if success else 1)
        else:
            # 默认启动Web界面
            if args.command == "webui" or args.command is None:
                # 如果没有指定子命令，为args添加默认的webui参数
                if args.command is None:
                    args.host = None
                    args.port = None
                    args.share = False
                    args.debug = False

                success = start_webui(args)
                sys.exit(0 if success else 1)
            else:
                print(f"未知命令: {args.command}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
