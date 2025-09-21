"""Web界面模块 - Gradio界面

包含音色系统的完整Web界面，包括音色继承和融合功能。
"""

try:
    from .app import create_app
    from .inheritance_tab import create_inheritance_interface, InheritanceTab
    from .fusion_tab import create_fusion_interface, FusionTab

    __all__ = [
        "create_app",
        "create_inheritance_interface",
        "InheritanceTab",
        "create_fusion_interface",
        "FusionTab"
    ]
except ImportError:
    # 如果依赖库未安装，提供占位符
    def create_app():
        raise ImportError("请先安装gradio等依赖库")

    def create_inheritance_interface(voice_manager):
        raise ImportError("请先安装gradio等依赖库")

    def create_fusion_interface(voice_manager):
        raise ImportError("请先安装gradio等依赖库")

    class InheritanceTab:
        def __init__(self, voice_manager):
            raise ImportError("请先安装gradio等依赖库")

    class FusionTab:
        def __init__(self, voice_manager):
            raise ImportError("请先安装gradio等依赖库")

    __all__ = [
        "create_app",
        "create_inheritance_interface",
        "InheritanceTab",
        "create_fusion_interface",
        "FusionTab"
    ]
