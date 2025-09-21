# 角色声音基底创建功能

这个文档详细介绍了角色声音基底创建功能的实现和使用方法。

## 功能概述

角色声音基底创建是整个音色定制系统的核心功能，允许用户通过以下步骤创建个性化的角色声音：

1. **选择预设音色标签** - 从预定义的音色类型中选择最接近目标的基础音色
2. **配置变声器参数** - 调整音调、共振峰、声域等参数
3. **选择说话人和权重** - 从多个说话人中选择并配置混合权重
4. **预览和调整** - 实时生成预览音频并进行参数调整
5. **保存配置** - 将满意的配置保存为角色声音基底

## 技术架构

### 核心组件

#### 1. 预设音色数据管理系统
- **VoicePresetManager** (`src/core/voice_preset_manager.py`)
  - 管理音色标签配置和说话人信息
  - 支持动态加载和缓存优化
  - 提供类型安全的数据访问接口

- **配置文件**
  - `src/data/presets/voice_tags.yaml` - 音色标签定义
  - `src/data/presets/speakers.yaml` - 说话人配置
  - `src/data/samples/` - 示例音频文件

#### 2. 角色声音基底创建器
- **VoiceBaseCreator** (`src/core/voice_base_creator.py`)
  - 实现完整的创建工作流
  - 集成DDSP-SVC和IndexTTS
  - 支持参数验证和错误处理
  - 提供进度回调和状态管理

#### 3. Gradio界面扩展
- **VoiceCreationTab** (`src/webui/voice_creation_tab.py`)
  - 直观的用户界面
  - 实时参数调整
  - 音频预览功能

### 数据流程

```
预设音频 → DDSP-SVC变声 → IndexTTS生成 → 用户预览 → 参数调整 → 保存配置
```

## 使用指南

### 1. 启动应用

```bash
python main.py --mode webui
```

### 2. 创建角色声音基底

1. **基本信息配置**
   - 输入角色名称和描述
   - 添加自定义标签

2. **选择音色标签**
   - 从下拉列表中选择最接近的音色类型
   - 查看标签信息和可用说话人

3. **配置变声器参数**
   - 音调偏移：调整音高（-12到+12半音）
   - 共振峰偏移：调整音色特征
   - 声域偏移：调整声音厚度

4. **说话人配置**
   - 选择要使用的说话人
   - 设置每个说话人的权重
   - 系统自动归一化权重

5. **预览和调整**
   - 输入预览文本
   - 选择情感控制方式
   - 生成预览音频
   - 根据效果调整参数

6. **保存配置**
   - 确认角色名称
   - 保存为音色配置

### 3. 管理音色

在"音色管理"标签页中可以：
- 查看所有创建的音色
- 删除不需要的音色
- 查看音色详情和统计信息

## 配置说明

### 音色标签配置 (voice_tags.yaml)

```yaml
voice_tags:
  青年男:
    description: "青年男性音色，成熟稳重"
    audio_file: "samples/young_male.wav"
    f0_range: [80, 250]
    speakers:
      - id: "young_male_01"
        name: "青年男1号"
        weight: 1.0
    default_ddsp_params:
      f0_min: 80.0
      f0_max: 250.0
      threhold: -60.0
      f0_predictor: "rmvpe"
```

### 说话人配置 (speakers.yaml)

```yaml
speakers:
  young_male_01:
    name: "青年男1号"
    model_path: "models/ddsp_svc/young_male_01.pth"
    config_path: "models/ddsp_svc/young_male_01.yaml"
    speaker_id: 0
    description: "成熟稳重的青年男性音色"
    tags: ["青年男", "成熟", "稳重"]
```

## API参考

### VoiceBaseCreationParams

创建参数类，包含所有必要的配置信息：

```python
@dataclass
class VoiceBaseCreationParams:
    voice_name: str                    # 角色名称
    description: str = ""              # 描述
    tags: List[str] = []              # 标签
    selected_tag: str = ""            # 选择的音色标签
    pitch_shift: float = 0.0          # 音调偏移
    formant_shift: float = 0.0        # 共振峰偏移
    vocal_register_shift: float = 0.0 # 声域偏移
    speaker_weights: Dict[str, float] # 说话人权重
    preview_text: str = "..."         # 预览文本
    emotion_control: str = "speaker"  # 情感控制
    emotion_weight: float = 0.65      # 情感权重
```

### VoiceBaseCreator主要方法

```python
# 获取可用音色标签
def get_available_tags() -> Dict[str, VoiceTagInfo]

# 选择音色标签
def select_voice_tag(tag_name: str) -> VoiceTagInfo

# 验证参数
def validate_parameters(params: VoiceBaseCreationParams) -> None

# 创建角色声音基底
def create_voice_base(
    params: VoiceBaseCreationParams,
    progress_callback: Optional[Callable] = None
) -> VoiceBaseCreationResult

# 保存音色配置
def save_voice_base(voice_config: VoiceConfig) -> None
```

## 扩展开发

### 添加新的音色标签

1. 在 `voice_tags.yaml` 中添加新标签定义
2. 在 `speakers.yaml` 中添加对应的说话人
3. 在 `src/data/samples/` 中添加示例音频文件

### 自定义变声器参数

可以在音色标签的 `default_ddsp_params` 中定义默认参数：

```yaml
default_ddsp_params:
  f0_min: 80.0
  f0_max: 250.0
  threhold: -60.0
  f0_predictor: "rmvpe"
  # 添加其他DDSP-SVC参数
```

### 集成新的TTS模型

1. 实现新的TTS集成类
2. 在 `VoiceBaseCreator` 中添加支持
3. 更新界面配置选项

## 故障排除

### 常见问题

1. **模型文件不存在**
   - 检查 `speakers.yaml` 中的路径配置
   - 确保DDSP-SVC模型文件已正确放置

2. **音频文件无法加载**
   - 检查 `voice_tags.yaml` 中的音频文件路径
   - 确保音频文件格式正确（WAV格式）

3. **预览生成失败**
   - 检查IndexTTS模型是否正确加载
   - 查看日志文件获取详细错误信息

4. **权重计算错误**
   - 确保至少选择一个说话人
   - 检查权重值是否大于0

### 日志调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

查看关键组件的日志：
- `VoiceBaseCreator` - 创建流程日志
- `VoicePresetManager` - 配置加载日志
- `DDSPSVCIntegration` - DDSP-SVC推理日志
- `IndexTTSIntegration` - IndexTTS推理日志

## 性能优化

### 缓存策略

1. **预设配置缓存** - 避免重复加载YAML文件
2. **模型缓存** - 复用已加载的模型
3. **音频缓存** - 缓存DDSP转换结果

### 内存管理

1. **及时清理临时文件**
2. **释放不用的模型内存**
3. **使用流式处理大音频文件**

### 并发处理

1. **异步音频处理**
2. **后台预加载模型**
3. **并行权重计算**

## 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_voice_base_creator.py

# 运行带覆盖率的测试
python -m pytest tests/ --cov=src/core/voice_base_creator
```

## 贡献指南

1. **代码风格** - 遵循项目的代码规范
2. **测试覆盖** - 新功能需要包含测试
3. **文档更新** - 更新相关文档
4. **性能考虑** - 注意内存和计算效率

## 许可证

本项目遵循项目根目录下的LICENSE文件。
