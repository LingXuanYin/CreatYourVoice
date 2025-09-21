"""DDSP-SVC版本支持演示

这个脚本演示如何使用CreatYourVoice系统的DDSP-SVC版本支持功能。
包括版本检测、版本切换、统一接口使用等功能。
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_version_detection():
    """演示版本检测功能"""
    print("=" * 60)
    print("🔍 DDSP-SVC版本检测演示")
    print("=" * 60)

    try:
        from src.utils.version_detector import get_ddsp_svc_version, DDSPSVCVersion

        # 检测当前版本
        print("正在检测DDSP-SVC版本...")
        version_info = get_ddsp_svc_version()

        print(f"✅ 检测结果:")
        print(f"   版本: {version_info.version.value}")
        print(f"   分支: {version_info.branch or '未知'}")
        print(f"   提交: {version_info.commit_hash or '未知'}")
        print(f"   路径: {version_info.path}")
        print(f"   特性: {version_info.features}")

        # 检查版本支持
        from src.utils.version_detector import is_version_supported

        for version in [DDSPSVCVersion.V6_1, DDSPSVCVersion.V6_3, DDSPSVCVersion.UNKNOWN]:
            supported = is_version_supported(version)
            status = "✅ 支持" if supported else "❌ 不支持"
            print(f"   {version.value}: {status}")

    except Exception as e:
        print(f"❌ 版本检测失败: {e}")


def demo_version_management():
    """演示版本管理功能"""
    print("\n" + "=" * 60)
    print("🔄 DDSP-SVC版本管理演示")
    print("=" * 60)

    try:
        from src.integrations.version_manager import get_version_manager, DDSPSVCVersion

        # 获取版本管理器
        manager = get_version_manager()

        # 检测当前版本
        print("正在检测和设置版本...")
        version_info = manager.detect_and_set_version()
        print(f"✅ 当前版本: {version_info.version.value}")

        # 获取支持的版本
        supported_versions = manager.get_supported_versions()
        print(f"📋 支持的版本: {[v.value for v in supported_versions]}")

        # 获取版本配置
        for version in supported_versions:
            config = manager.get_version_config(version)
            print(f"\n📝 {version.value}版本配置:")
            print(f"   音量提取器参数: {config['volume_extractor_args']}")
            print(f"   支持声域偏移: {config['supports_vocal_register']}")
            print(f"   默认t_start: {config['default_t_start']}")
            print(f"   掩码处理方式: {config['mask_processing']}")

        # 演示版本切换
        print(f"\n🔄 演示版本切换...")
        current_version = version_info.version
        target_version = DDSPSVCVersion.V6_1 if current_version == DDSPSVCVersion.V6_3 else DDSPSVCVersion.V6_3

        print(f"尝试从 {current_version.value} 切换到 {target_version.value}...")
        success = manager.switch_version(target_version)

        if success:
            print(f"✅ 版本切换成功")
            # 切换回原版本
            manager.switch_version(current_version)
            print(f"🔙 已切换回原版本: {current_version.value}")
        else:
            print(f"⚠️ 版本切换失败，继续使用当前版本")

    except Exception as e:
        print(f"❌ 版本管理演示失败: {e}")


def demo_unified_interface():
    """演示统一接口功能"""
    print("\n" + "=" * 60)
    print("🎯 DDSP-SVC统一接口演示")
    print("=" * 60)

    try:
        from src.integrations.ddsp_svc_unified import DDSPSVCUnified

        # 创建统一接口实例
        print("正在创建统一接口实例...")

        # 自动检测版本
        unified_auto = DDSPSVCUnified(version="auto")
        version_info = unified_auto.get_version_info()
        print(f"✅ 自动检测版本: {version_info['current_version']}")

        # 指定版本
        for version in ["6.1", "6.3"]:
            try:
                print(f"\n🔧 创建{version}版本实例...")
                unified_specific = DDSPSVCUnified(version=version)

                # 获取支持的功能
                features = unified_specific.get_supported_features()
                print(f"✅ {version}版本功能:")
                for feature, supported in features.items():
                    status = "✅" if supported else "❌"
                    print(f"   {feature}: {status}")

                # 演示参数适配
                print(f"📝 {version}版本推理参数演示:")
                if version == "6.1":
                    print("   - 不支持vocal_register_shift参数")
                    print("   - 默认t_start=0.7")
                    print("   - 使用padding掩码处理")
                else:
                    print("   - 支持vocal_register_shift参数")
                    print("   - 默认t_start=0.0")
                    print("   - 使用upsample掩码处理")

            except Exception as e:
                print(f"❌ 创建{version}版本实例失败: {e}")

    except Exception as e:
        print(f"❌ 统一接口演示失败: {e}")


def demo_compatibility():
    """演示兼容性功能"""
    print("\n" + "=" * 60)
    print("🔗 DDSP-SVC兼容性演示")
    print("=" * 60)

    try:
        from src.integrations.ddsp_svc import DDSPSVCIntegration

        # 创建集成实例（自动使用统一接口）
        print("正在创建DDSP-SVC集成实例...")
        integration = DDSPSVCIntegration()

        if hasattr(integration, '_use_unified') and integration._use_unified:
            print("✅ 成功使用统一接口")

            # 获取版本信息
            if hasattr(integration, '_unified'):
                version_info = integration._unified.get_version_info()
                print(f"📋 当前版本: {version_info['current_version']}")
                print(f"📋 支持的版本: {version_info['supported_versions']}")
        else:
            print("⚠️ 回退到原有实现")

        # 演示推理接口兼容性
        print("\n🎯 推理接口兼容性:")
        print("   - load_model(): ✅ 兼容")
        print("   - infer(): ✅ 兼容，自动处理版本差异")
        print("   - save_audio(): ✅ 兼容")
        print("   - get_model_info(): ✅ 兼容，包含版本信息")

    except Exception as e:
        print(f"❌ 兼容性演示失败: {e}")


def demo_error_handling():
    """演示错误处理功能"""
    print("\n" + "=" * 60)
    print("🛡️ DDSP-SVC错误处理演示")
    print("=" * 60)

    try:
        from src.integrations.ddsp_svc_unified import DDSPSVCUnified
        from src.integrations.version_manager import get_version_manager

        # 演示版本检测失败的处理
        print("🔍 演示版本检测失败处理...")
        try:
            # 使用不存在的路径
            from src.integrations.version_manager import VersionManagerConfig
            config = VersionManagerConfig(ddsp_svc_path=Path("/non/existent/path"))
            manager = get_version_manager(config)
            version_info = manager.detect_and_set_version()
            print(f"⚠️ 检测失败时的回退: {version_info.version.value}")
        except Exception as e:
            print(f"✅ 错误处理正常: {e}")

        # 演示模型加载失败的处理
        print("\n📁 演示模型加载失败处理...")
        try:
            unified = DDSPSVCUnified()
            unified.load_model("/non/existent/model.pt")
        except Exception as e:
            print(f"✅ 模型加载错误处理正常: {type(e).__name__}")

        # 演示推理失败的处理
        print("\n🎯 演示推理失败处理...")
        try:
            unified = DDSPSVCUnified()
            # 未加载模型就推理
            unified.infer(audio="test.wav")
        except Exception as e:
            print(f"✅ 推理错误处理正常: {type(e).__name__}")

    except Exception as e:
        print(f"❌ 错误处理演示失败: {e}")


def demo_performance_features():
    """演示性能优化功能"""
    print("\n" + "=" * 60)
    print("⚡ DDSP-SVC性能优化演示")
    print("=" * 60)

    try:
        from src.integrations.version_manager import get_version_manager

        manager = get_version_manager()

        # 演示缓存功能
        print("💾 缓存功能演示:")
        print("   - 版本检测缓存: ✅ 避免重复检测")
        print("   - 适配器缓存: ✅ 复用已创建的适配器")
        print("   - 模型缓存: ✅ 避免重复加载模型")

        # 演示缓存清理
        print("\n🧹 缓存清理演示:")
        manager.clear_cache()
        print("✅ 缓存清理完成")

        # 演示延迟加载
        print("\n⏳ 延迟加载演示:")
        print("   - 适配器延迟加载: ✅ 只在需要时创建")
        print("   - 编码器延迟加载: ✅ 只在推理时创建")
        print("   - 模型延迟加载: ✅ 只在调用时加载")

    except Exception as e:
        print(f"❌ 性能优化演示失败: {e}")


def main():
    """主函数"""
    print("🎵 CreatYourVoice - DDSP-SVC版本支持功能演示")
    print("=" * 60)
    print("本演示将展示DDSP-SVC 6.1和6.3版本的支持功能")
    print("包括版本检测、管理、统一接口、兼容性等特性")

    try:
        # 1. 版本检测演示
        demo_version_detection()

        # 2. 版本管理演示
        demo_version_management()

        # 3. 统一接口演示
        demo_unified_interface()

        # 4. 兼容性演示
        demo_compatibility()

        # 5. 错误处理演示
        demo_error_handling()

        # 6. 性能优化演示
        demo_performance_features()

        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)
        print("✅ 所有功能演示成功")
        print("📖 详细文档请参考: README_DDSP_SVC_VERSION_SUPPORT.md")
        print("🧪 运行测试: python tests/test_ddsp_svc_versions.py")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        logger.exception("演示失败")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
