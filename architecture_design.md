# CreatYourVoice 统一WebUI架构设计

## 系统概述

本文档设计了一个统一的Gradio界面，将DDSP-SVC 6.3和IndexTTS v2整合到一个高效、可维护的系统中。

## 设计原则

### 1. 简洁至上
- 每个模块只负责一个功能
- 消除不必要的复杂性
- 避免过度设计

### 2. 数据结构优先
- 设计正确的数据流
- 统一的音频格式处理
- 清晰的状态管理

### 3. 错误处理统一
- 集中的异常处理机制
- 用户友好的错误信息
- 系统稳定性保证

## 系统架构

### 核心模块划分

```
unified_webui/
├── core/                    # 核心业务逻辑
│   ├── audio_processor.py   # 统一音频处理
│   ├── tts_engine.py       # TTS推理引擎
│   ├── svc_engine.py       # SVC推理引擎
│   └── pipeline.py         # TTS→SVC串联管道
├── ui/                     # 界面层
│   ├── components/         # 可复用组件
│   ├── tabs/              # 各功能Tab
│   └── app.py             # 主应用入口
├── config/                # 配置管理
│   ├── settings.py        # 统一配置
│   └── models.py          # 模型配置
└── utils/                 # 工具函数
    ├── audio_utils.py     # 音频工具
    └── error_handler.py   # 错误处理
```

### 数据流设计

#### 1. 音频数据结构
```python
@dataclass
class AudioData:
    """统一的音频数据结构"""
    data: torch.Tensor      # 音频数据
    sample_rate: int        # 采样率
    channels: int = 1       # 声道数
    dtype: torch.dtype = torch.float32

    def resample(self, target_sr: int) -> 'AudioData':
        """重采样到目标采样率"""
        pass

    def to_mono(self) -> 'AudioData':
        """转换为单声道"""
        pass
```

#### 2. 推理结果结构
```python
@dataclass
class InferenceResult:
    """推理结果统一结构"""
    audio: AudioData
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
```

### 核心引擎设计

#### 1. TTS引擎接口
```python
class TTSEngine:
    """TTS推理引擎基类"""

    def __init__(self, config_path: str, model_dir: str):
        self.model = None
        self.config = None

    def load_model(self) -> None:
        """加载模型"""
        pass

    def unload_model(self) -> None:
        """卸载模型释放内存"""
        pass

    def infer(self, text: str, reference_audio: AudioData, **kwargs) -> InferenceResult:
        """TTS推理接口"""
        pass
```

#### 2. SVC引擎接口
```python
class SVCEngine:
    """SVC推理引擎基类"""

    def __init__(self, model_path: str):
        self.model = None
        self.config = None

    def load_model(self) -> None:
        """加载模型"""
        pass

    def unload_model(self) -> None:
        """卸载模型释放内存"""
        pass

    def infer(self, audio: AudioData, **kwargs) -> InferenceResult:
        """SVC推理接口"""
        pass
```

#### 3. 统一音频处理器
```python
class AudioProcessor:
    """统一音频处理器"""

    @staticmethod
    def load_audio(file_path: str) -> AudioData:
        """加载音频文件"""
        pass

    @staticmethod
    def save_audio(audio: AudioData, file_path: str) -> None:
        """保存音频文件"""
        pass

    @staticmethod
    def normalize_audio(audio: AudioData) -> AudioData:
        """音频标准化"""
        pass

    @staticmethod
    def resample_audio(audio: AudioData, target_sr: int) -> AudioData:
        """音频重采样"""
        pass
```

### 串联管道设计

```python
class TTSSVCPipeline:
    """TTS→SVC串联管道"""

    def __init__(self, tts_engine: TTSEngine, svc_engine: SVCEngine):
        self.tts_engine = tts_engine
        self.svc_engine = svc_engine
        self.audio_processor = AudioProcessor()

    def process(self,
                text: str,
                tts_reference: AudioData,
                svc_target_speaker: int,
                **kwargs) -> InferenceResult:
        """完整的TTS→SVC处理流程"""

        # 1. TTS生成
        tts_result = self.tts_engine.infer(text, tts_reference, **kwargs.get('tts_params', {}))
        if not tts_result.success:
            return tts_result

        # 2. 音频格式统一
        tts_audio = self.audio_processor.normalize_audio(tts_result.audio)

        # 3. SVC转换
        svc_result = self.svc_engine.infer(tts_audio, spk_id=svc_target_speaker, **kwargs.get('svc_params', {}))

        return svc_result
```

## 界面设计

### Tab布局结构

```
主界面
├── TTS Tab                 # 纯TTS功能
│   ├── 文本输入
│   ├── 参考音频上传
│   ├── 情感控制
│   └── 生成按钮
├── SVC Tab                 # 纯SVC功能
│   ├── 音频上传
│   ├── 说话人选择
│   ├── 音调调节
│   └── 转换按钮
├── TTS→SVC Tab            # 串联功能
│   ├── 文本输入
│   ├── TTS参考音频
│   ├── SVC目标说话人
│   └── 一键生成按钮
└── 设置Tab               # 系统设置
    ├── 模型管理
    ├── 音频设置
    └── 性能设置
```

### 组件复用设计

```python
# 可复用的UI组件
class AudioUploadComponent:
    """音频上传组件"""
    pass

class SpeakerSelectComponent:
    """说话人选择组件"""
    pass

class ProgressComponent:
    """进度显示组件"""
    pass

class AudioPlayerComponent:
    """音频播放组件"""
    pass
```

## 配置管理

### 统一配置结构
```python
@dataclass
class UnifiedConfig:
    """统一配置类"""

    # 系统设置
    device: str = "cuda"
    max_audio_length: int = 30  # 秒
    temp_dir: str = "temp"

    # TTS设置
    tts_model_dir: str = "index-tts/checkpoints"
    tts_sample_rate: int = 22050

    # SVC设置
    svc_model_path: str = "DDSP-SVC/exp/model.pt"
    svc_sample_rate: int = 44100

    # 音频处理设置
    target_sample_rate: int = 44100  # 统一采样率
    audio_format: str = "wav"

    def load_from_file(self, config_path: str) -> None:
        """从文件加载配置"""
        pass

    def save_to_file(self, config_path: str) -> None:
        """保存配置到文件"""
        pass
```

## 错误处理机制

### 统一异常类
```python
class UnifiedWebUIError(Exception):
    """基础异常类"""
    pass

class ModelLoadError(UnifiedWebUIError):
    """模型加载错误"""
    pass

class AudioProcessingError(UnifiedWebUIError):
    """音频处理错误"""
    pass

class InferenceError(UnifiedWebUIError):
    """推理错误"""
    pass
```

### 错误处理装饰器
```python
def handle_errors(func):
    """统一错误处理装饰器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return InferenceResult(
                audio=None,
                metadata={},
                processing_time=0,
                success=False,
                error_message=str(e)
            )
    return wrapper
```

## 性能优化策略

### 1. 模型生命周期管理
- 懒加载：只在需要时加载模型
- 智能卸载：内存不足时自动卸载不常用模型
- 模型缓存：避免重复加载

### 2. 音频处理优化
- 流式处理：大文件分块处理
- 内存池：复用音频缓冲区
- 异步处理：非阻塞UI

### 3. 缓存策略
- 音频特征缓存：避免重复提取
- 结果缓存：相同输入直接返回
- 临时文件管理：自动清理

## 扩展性设计

### 1. 插件架构
- 支持新的TTS/SVC模型
- 可插拔的音频处理模块
- 自定义UI组件

### 2. API接口
- RESTful API支持
- WebSocket实时通信
- 批处理接口

### 3. 多语言支持
- 国际化框架
- 动态语言切换
- 本地化配置

## 开发计划

### 阶段1：核心架构（1-2周）
1. 实现统一的数据结构
2. 开发TTS/SVC引擎接口
3. 构建音频处理器
4. 设计配置管理系统

### 阶段2：界面开发（1-2周）
1. 创建可复用UI组件
2. 实现各功能Tab
3. 集成错误处理
4. 添加进度显示

### 阶段3：集成测试（1周）
1. 端到端测试
2. 性能优化
3. 错误处理完善
4. 文档编写

### 阶段4：部署优化（1周）
1. 容器化部署
2. 性能监控
3. 用户反馈收集
4. 持续优化

## 技术规范

### 代码规范
- 使用Type Hints
- 遵循PEP 8
- 单元测试覆盖率 > 80%
- 文档字符串完整

### 依赖管理
- 使用pyproject.toml
- 锁定版本号
- 最小化依赖

### 质量保证
- 代码审查
- 自动化测试
- 性能基准测试
- 安全扫描

## 总结

这个架构设计遵循了简洁性原则，通过正确的数据结构设计和清晰的模块划分，解决了现有代码的复杂性问题。重点关注：

1. **数据结构优先**：统一的音频数据格式
2. **接口简化**：每个组件职责单一
3. **错误处理统一**：集中的异常管理
4. **性能优化**：智能的资源管理
5. **扩展性**：支持未来功能扩展

这个设计将显著提高系统的可维护性、稳定性和用户体验。
