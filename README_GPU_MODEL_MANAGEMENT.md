# GPU模型管理系统

CreatYourVoice的GPU模型管理系统提供了统一的模型加载、卸载、内存监控和生命周期管理功能。

## 🎯 核心特性

### 1. 统一模型管理
- **多模型支持**：同时管理DDSP-SVC和IndexTTS模型
- **智能加载**：自动选择最优GPU设备
- **引用计数**：基于使用情况自动管理模型生命周期
- **内存估算**：预估模型内存需求，避免OOM错误

### 2. 实时内存监控
- **GPU状态监控**：实时显示GPU内存使用、利用率、温度
- **系统内存监控**：监控系统RAM使用情况
- **历史记录**：保存内存使用历史用于分析
- **智能警告**：内存使用率过高时自动警告

### 3. 自动化管理
- **自动清理**：空闲模型自动卸载释放内存
- **智能预加载**：根据任务需求预加载模型
- **内存优化**：自动清理GPU缓存和垃圾回收
- **错误恢复**：模型加载失败时自动重试或降级

### 4. 用户友好界面
- **直观控制**：一键加载/卸载模型
- **实时状态**：显示模型状态和内存使用
- **批量操作**：支持批量模型管理
- **高级功能**：任务预加载、自动管理配置

## 🏗️ 系统架构

```
GPU模型管理系统
├── 核心组件
│   ├── GPUModelManager      # 模型管理器
│   ├── ModelLifecycleManager # 生命周期管理
│   └── GPUManagerInitializer # 系统初始化
├── 工具模块
│   ├── GPUUtils            # GPU工具函数
│   ├── MemoryMonitor       # 内存监控
│   └── ErrorHandler       # 错误处理
└── 界面组件
    └── ModelManagementTab  # Gradio管理界面
```

## 🚀 快速开始

### 1. 系统初始化

```python
from src.core.gpu_manager_init import initialize_gpu_management

# 初始化GPU模型管理系统
success = initialize_gpu_management()
if success:
    print("GPU模型管理系统初始化成功")
```

### 2. 加载模型

```python
from src.core.gpu_model_manager import get_model_manager

manager = get_model_manager()

# 加载DDSP-SVC模型
ddsp_model_id = manager.load_ddsp_model(
    model_path="path/to/ddsp_model.pth",
    device="auto"  # 自动选择最优设备
)

# 加载IndexTTS模型
tts_model_id = manager.load_index_tts_model(
    model_dir="path/to/index_tts_model",
    device="cuda:0"
)
```

### 3. 使用模型

```python
# 获取模型实例
ddsp_model = manager.get_model(ddsp_model_id)
tts_model = manager.get_model(tts_model_id)

# 使用模型进行推理
if ddsp_model:
    result = ddsp_model.infer(audio_data, speaker_id=1)

if tts_model:
    result = tts_model.infer(text="Hello", speaker_audio="speaker.wav")
```

### 4. 卸载模型

```python
# 卸载单个模型
manager.unload_model(ddsp_model_id)

# 卸载所有模型
manager.unload_all_models()
```

## 📊 内存监控

### 启动监控

```python
from src.utils.memory_monitor import start_global_monitoring

# 启动全局内存监控
start_global_monitoring()
```

### 获取状态

```python
from src.utils.memory_monitor import get_memory_monitor

monitor = get_memory_monitor()

# 获取当前状态
status = monitor.get_current_status()
print(f"GPU内存使用: {status['gpu_memory']}")
print(f"系统内存使用: {status['system_memory']}")

# 获取内存历史
history = monitor.get_memory_history(minutes=10)
```

## 🔄 生命周期管理

### 任务配置

```python
from src.core.model_lifecycle import get_lifecycle_manager, TaskProfile, PreloadConfig, ModelType

lifecycle = get_lifecycle_manager()

# 定义语音合成任务
task_profile = TaskProfile(
    task_name="voice_synthesis",
    required_models=[
        PreloadConfig(
            model_type=ModelType.INDEX_TTS,
            model_path="path/to/index_tts",
            device="auto",
            priority=1
        )
    ],
    memory_limit_mb=4096,
    auto_cleanup=True
)

# 注册任务配置
lifecycle.register_task_profile(task_profile)
```

### 任务预加载

```python
# 为任务预加载模型
results = lifecycle.preload_for_task("voice_synthesis")
print(f"预加载结果: {results}")

# 任务完成后清理
cleanup_results = lifecycle.cleanup_for_task("voice_synthesis")
print(f"清理结果: {cleanup_results}")
```

## 🎛️ Web界面使用

### 启动应用

```bash
python main.py
```

### 访问GPU管理界面

1. 打开浏览器访问 `http://localhost:7860`
2. 点击 "🔧 GPU模型管理" 标签页
3. 使用界面进行模型管理操作

### 界面功能

- **模型控制**：加载/卸载模型
- **状态监控**：查看GPU和内存状态
- **模型列表**：查看已加载模型详情
- **GPU监控**：实时GPU信息显示
- **高级功能**：任务预加载、自动管理设置

## ⚙️ 配置选项

### GPU设置

在 `config.yaml` 中配置GPU管理参数：

```yaml
gpu:
  auto_cleanup_enabled: true
  cleanup_threshold_percent: 85.0
  idle_timeout_minutes: 30
  memory_monitoring_enabled: true
  memory_monitoring_interval: 2.0
  max_memory_usage_percent: 90.0
```

### 环境变量

```bash
# 设置设备
export CREATYOURVOICE_DEVICE=cuda:0

# 启用调试模式
export CREATYOURVOICE_DEBUG=true
```

## 🔧 命令行工具

### 检查GPU状态

```bash
python main.py gpu
```

输出示例：
```
=== GPU模型管理状态 ===
系统初始化: ✓
已加载模型: 2/3
自动清理: 启用
内存监控: 运行中
生命周期管理: 启用

=== GPU设备信息 ===
GPU 0: NVIDIA GeForce RTX 4090
  内存: 2048MB / 24576MB (8.3%)
  利用率: 15.2%
```

## 🛠️ 高级功能

### 自定义恢复策略

```python
from src.utils.error_handler import get_error_handler, RecoveryAction, RecoveryStrategy

error_handler = get_error_handler()

# 注册自定义恢复动作
def custom_recovery():
    # 自定义恢复逻辑
    return True

error_handler.register_recovery_action(
    "CustomError",
    RecoveryAction(
        strategy=RecoveryStrategy.RETRY,
        action=custom_recovery,
        description="自定义恢复策略",
        max_attempts=3
    )
)
```

### 内存警告回调

```python
from src.utils.memory_monitor import get_memory_monitor

def memory_alert_callback(alert):
    print(f"内存警告: {alert.message}")
    # 自定义处理逻辑

monitor = get_memory_monitor()
monitor.add_alert_callback(memory_alert_callback)
```

### 智能模型切换

```python
from src.core.gpu_manager_init import auto_load_for_synthesis, auto_load_for_conversion

# 为语音合成自动加载模型
tts_model_id = auto_load_for_synthesis(
    text="Hello world",
    speaker_audio="speaker.wav"
)

# 为音色转换自动加载模型
ddsp_model_id = auto_load_for_conversion(
    audio_path="input.wav",
    target_speaker="speaker1"
)
```

## 📈 性能优化

### 内存优化建议

1. **合理设置清理阈值**：根据GPU内存大小调整 `cleanup_threshold_percent`
2. **使用任务预加载**：为常用任务预加载模型减少等待时间
3. **监控内存使用**：定期检查内存使用情况，及时清理
4. **批量操作**：避免频繁的单个模型加载/卸载

### 设备选择策略

```python
from src.utils.gpu_utils import GPUUtils

# 获取最优设备
optimal_device = GPUUtils.get_optimal_device()

# 检查内存需求
if GPUUtils.check_memory_requirement(2048, optimal_device):
    # 内存充足，使用GPU
    device = optimal_device
else:
    # 内存不足，使用CPU
    device = "cpu"
```

## 🐛 故障排除

### 常见问题

1. **CUDA不可用**
   - 检查CUDA安装：`nvidia-smi`
   - 检查PyTorch CUDA支持：`torch.cuda.is_available()`

2. **内存不足**
   - 降低模型并发数量
   - 增加清理频率
   - 使用CPU模式

3. **模型加载失败**
   - 检查模型文件路径
   - 验证模型文件完整性
   - 查看错误日志

### 调试模式

```bash
# 启用详细日志
python main.py --log-level DEBUG

# 检查系统状态
python main.py check
```

## 📚 API参考

### GPUModelManager

主要方法：
- `load_ddsp_model(model_path, device, model_id)` - 加载DDSP模型
- `load_index_tts_model(model_dir, device, model_id)` - 加载IndexTTS模型
- `unload_model(model_id, force)` - 卸载模型
- `get_model(model_id)` - 获取模型实例
- `optimize_memory()` - 优化内存使用

### MemoryMonitor

主要方法：
- `start_monitoring()` - 启动监控
- `stop_monitoring()` - 停止监控
- `get_current_status()` - 获取当前状态
- `get_memory_history(minutes)` - 获取历史记录

### ModelLifecycleManager

主要方法：
- `register_task_profile(profile)` - 注册任务配置
- `preload_for_task(task_name)` - 任务预加载
- `cleanup_for_task(task_name)` - 任务清理
- `start_auto_management()` - 启动自动管理

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为GPU模型管理系统做出贡献的开发者和用户。

---

**注意**：GPU模型管理系统需要NVIDIA GPU和CUDA支持才能发挥最佳性能。在CPU模式下，某些功能可能受限。
