# DDSP-SVC版本支持文档

## 概述

CreatYourVoice系统现在支持DDSP-SVC的6.1和6.3两个版本，提供自动版本检测、版本切换和统一的推理接口。这个功能确保用户可以使用不同版本的DDSP-SVC模型，而无需担心版本兼容性问题。

## 功能特性

### 🔍 自动版本检测
- **Git分支检测**: 自动检测当前Git分支确定版本
- **文件特征分析**: 通过代码特征识别版本差异
- **代码签名检测**: 分析API调用方式确定版本
- **缓存机制**: 避免重复检测，提高性能

### 🔄 版本管理
- **版本切换**: 支持在6.1和6.3版本间切换
- **配置管理**: 自动适配不同版本的配置参数
- **错误处理**: 版本切换失败时的回退机制
- **状态监控**: 实时显示当前版本状态

### 🎯 统一接口
- **版本无关API**: 提供一致的推理接口
- **自动路由**: 内部自动选择对应版本的实现
- **参数适配**: 自动处理版本间的参数差异
- **向后兼容**: 保持与现有代码的兼容性

### 🎨 用户界面
- **版本选择器**: 简单的版本选择界面
- **版本管理页**: 完整的版本管理功能
- **特性对比**: 直观显示版本间的差异
- **状态显示**: 实时显示版本信息和状态

## 版本差异对比

| 特性 | DDSP-SVC 6.1 | DDSP-SVC 6.3 |
|------|-------------|-------------|
| **声域偏移** | ❌ 不支持 | ✅ 支持 |
| **音量处理** | `Volume_Extractor(hop_size)` | `Volume_Extractor(hop_size, win_size)` |
| **掩码处理** | padding + 滑动窗口 | 直接upsample |
| **推理方式** | 直接返回音频 (`return_wav=True`) | 分离mel生成和vocoder推理 |
| **默认t_start** | 0.7 | 0.0 |
| **稳定性** | 稳定版本 | 最新功能 |

## 架构设计

### 核心组件

```
src/
├── utils/
│   └── version_detector.py      # 版本检测器
├── integrations/
│   ├── version_manager.py       # 版本管理器
│   ├── ddsp_svc_v61.py         # 6.1版本适配器
│   ├── ddsp_svc_v63.py         # 6.3版本适配器
│   ├── ddsp_svc_unified.py     # 统一接口
│   └── ddsp_svc.py             # 更新的集成模块
├── webui/
│   └── version_selection.py    # 版本选择界面
└── tests/
    └── test_ddsp_svc_versions.py # 版本测试
```

### 设计原则

1. **简洁至上**: 消除版本差异的复杂性，提供简单统一的接口
2. **自动适配**: 用户无需手动处理版本差异
3. **向后兼容**: 现有代码无需修改即可使用
4. **错误恢复**: 提供完善的错误处理和回退机制

## 使用方法

### 基本使用

```python
from src.integrations.ddsp_svc_unified import DDSPSVCUnified

# 自动检测版本
unified = DDSPSVCUnified(version="auto")

# 或指定版本
unified = DDSPSVCUnified(version="6.1")  # 使用6.1版本
unified = DDSPSVCUnified(version="6.3")  # 使用6.3版本

# 加载模型
unified.load_model("path/to/model.pt")

# 推理（统一接口，自动处理版本差异）
result = unified.infer(
    audio="input.wav",
    speaker_id=1,
    key_shift=2.0,
    vocal_register_shift=1.0  # 仅6.3版本支持，6.1版本会自动忽略
)

# 保存结果
unified.save_audio(result, "output.wav")
```

### 版本管理

```python
from src.integrations.version_manager import get_version_manager

# 获取版本管理器
manager = get_version_manager()

# 检测当前版本
version_info = manager.detect_and_set_version()
print(f"当前版本: {version_info.version.value}")

# 切换版本
success = manager.switch_version(DDSPSVCVersion.V6_1)
if success:
    print("版本切换成功")

# 获取版本配置
config = manager.get_version_config()
print(f"支持声域偏移: {config['supports_vocal_register']}")
```

### 便捷函数

```python
from src.integrations.ddsp_svc_unified import convert_voice

# 一键转换（自动处理版本）
result = convert_voice(
    input_path="input.wav",
    output_path="output.wav",
    model_path="model.pt",
    version="auto",  # 自动检测版本
    speaker_id=1,
    key_shift=2.0
)
```

### 兼容性使用

```python
from src.integrations.ddsp_svc import DDSPSVCIntegration

# 现有代码无需修改，自动使用统一接口
integration = DDSPSVCIntegration()  # 自动检测版本
integration.load_model("model.pt")

result = integration.infer(
    audio="input.wav",
    speaker_id=1,
    vocal_register_shift=1.0  # 自动处理版本兼容性
)
```

## 用户界面

### 版本管理页面

在Web界面中，用户可以：

1. **查看版本信息**: 显示当前检测到的版本、Git分支、提交哈希等
2. **版本切换**: 通过下拉菜单选择目标版本
3. **特性对比**: 查看不同版本的功能差异
4. **高级设置**: 自定义DDSP-SVC路径、清理缓存等

### 版本选择器

在其他功能页面中，提供简单的版本选择器：

```python
from src.webui.version_selection import create_version_selector

# 创建版本选择器
version_dropdown, switch_btn, status_text = create_version_selector()
```

## 配置说明

### 版本检测配置

```python
from src.integrations.version_manager import VersionManagerConfig

config = VersionManagerConfig(
    ddsp_svc_path=Path("custom/ddsp/path"),  # 自定义DDSP-SVC路径
    preferred_version=DDSPSVCVersion.V6_3,   # 首选版本
    auto_switch=True,                        # 自动切换
    cache_adapters=True                      # 缓存适配器
)
```

### 版本特定参数

不同版本的默认参数会自动设置：

- **6.1版本**: `t_start=0.7`, 不支持`vocal_register_shift`
- **6.3版本**: `t_start=0.0`, 支持`vocal_register_shift`

## 错误处理

### 常见问题

1. **版本检测失败**
   - 自动回退到6.3版本
   - 提供详细的错误信息
   - 支持手动指定版本

2. **版本切换失败**
   - 保持当前版本不变
   - 记录错误日志
   - 提供重试机制

3. **模型加载失败**
   - 检查模型文件是否存在
   - 验证模型与版本的兼容性
   - 提供详细的错误信息

### 调试方法

```python
# 启用详细日志
import logging
logging.getLogger('src.integrations').setLevel(logging.DEBUG)

# 强制刷新版本检测
version_info = manager.detect_and_set_version(force_refresh=True)

# 获取详细的版本信息
summary = manager.get_version_summary()
print(summary)
```

## 测试

### 运行测试

```bash
# 运行版本支持测试
python tests/test_ddsp_svc_versions.py

# 运行所有测试
python tests/run_all_tests.py
```

### 测试覆盖

- ✅ 版本检测功能
- ✅ 版本切换功能
- ✅ 统一接口功能
- ✅ 适配器功能
- ✅ 兼容性测试
- ✅ 错误处理测试

## 性能优化

### 缓存机制

1. **版本检测缓存**: 避免重复检测
2. **适配器缓存**: 复用已创建的适配器
3. **模型缓存**: 避免重复加载模型

### 内存管理

1. **自动清理**: 版本切换时自动清理GPU内存
2. **延迟加载**: 只在需要时加载组件
3. **资源释放**: 提供手动清理缓存的方法

## 扩展性

### 添加新版本

要添加对新版本的支持：

1. 创建新的适配器类（如`ddsp_svc_v64.py`）
2. 在版本检测器中添加检测逻辑
3. 在版本管理器中注册新版本
4. 更新统一接口以支持新版本

### 自定义适配器

```python
class CustomDDSPSVCAdapter:
    def __init__(self, ddsp_svc_path, version_info, device):
        # 自定义初始化逻辑
        pass

    def load_model(self, model_path):
        # 自定义模型加载逻辑
        pass

    def infer(self, **kwargs):
        # 自定义推理逻辑
        pass
```

## 最佳实践

### 版本选择建议

1. **新项目**: 推荐使用6.3版本，功能更完整
2. **现有项目**: 可以继续使用6.1版本，确保稳定性
3. **实验性功能**: 使用6.3版本体验声域偏移等新功能

### 性能优化建议

1. **启用缓存**: 保持`cache_adapters=True`
2. **避免频繁切换**: 减少不必要的版本切换
3. **及时清理**: 定期清理不需要的缓存

### 错误处理建议

1. **检查日志**: 遇到问题时查看详细日志
2. **版本验证**: 确保模型与版本兼容
3. **回退机制**: 利用自动回退功能确保稳定性

## 更新日志

### v1.0.0 (2024-01-XX)
- ✨ 新增DDSP-SVC 6.1和6.3版本支持
- ✨ 实现自动版本检测功能
- ✨ 提供统一的推理接口
- ✨ 添加版本管理用户界面
- ✨ 完整的测试覆盖
- 📝 详细的文档说明

## 贡献指南

欢迎为DDSP-SVC版本支持功能贡献代码：

1. Fork项目
2. 创建功能分支
3. 添加测试用例
4. 提交Pull Request

## 许可证

本功能遵循项目的整体许可证。

---

**注意**: 使用不同版本的DDSP-SVC时，请确保相应的依赖和预训练模型已正确安装。
