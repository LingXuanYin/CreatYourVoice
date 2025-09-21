# CreatYourVoice

基于DDSP-SVC和IndexTTS v2的音色创建和管理系统

## 项目概述

CreatYourVoice是一个复杂的音色定制系统，结合了DDSP-SVC的音色融合技术和IndexTTS v2的可控情感语音合成能力，为用户提供完整的音色创建、管理和语音合成解决方案。

### 核心功能

1. **角色声音基底创建**：通过DDSP-SVC音色融合和变声器，创建包含足够音高音素特征但情感特征少的基底音频
2. **情感语音合成**：使用IndexTTS v2的可控情感特性，基于角色声音基底合成带情感的语音
3. **音色融合和继承**：支持从现有音色创建新音色，支持多音色融合
4. **智能权重计算**：用户输入任意数字，系统自动归一化权重
5. **Web界面管理**：基于Gradio的直观Web界面

## 系统架构

```
src/
├── core/                    # 核心模块
│   ├── models.py           # 数据模型定义
│   ├── weight_calculator.py # 权重计算系统
│   └── voice_manager.py    # 音色管理器
├── integrations/           # 集成模块
│   ├── ddsp_svc.py        # DDSP-SVC集成
│   └── index_tts.py       # IndexTTS集成
├── utils/                  # 工具模块
│   ├── audio_utils.py     # 音频处理工具
│   ├── config.py          # 配置管理
│   └── logging_config.py  # 日志配置
└── webui/                 # Web界面
    └── app.py             # Gradio界面
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (推荐，用于GPU加速)
- 16GB+ RAM (推荐)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd CreatYourVoice
```

2. **安装依赖**
```bash
# 使用uv (推荐)
uv sync

# 或使用pip
pip install -e .
```

3. **下载模型**
```bash
# DDSP-SVC模型
# 将模型文件放置在 DDSP-SVC/exp/ 目录下

# IndexTTS模型
# 将模型文件放置在 index-tts/checkpoints/ 目录下
```

4. **启动应用**
```bash
# 启动Web界面
python main.py

# 或指定参数
python main.py --host 0.0.0.0 --port 7860 --share
```

### 命令行工具

```bash
# 显示配置信息
python main.py config

# 系统检查
python main.py check

# 查看音色列表
python main.py voices

# 启动Web界面
python main.py webui --host 0.0.0.0 --port 7860
```

## 使用指南

### 1. 音色创建

1. 在Web界面中选择"音色创建"标签页
2. 填写音色基本信息（名称、描述、标签）
3. 配置DDSP-SVC参数（模型路径、说话人ID等）
4. 配置IndexTTS参数（模型路径、情感强度等）
5. 点击"创建音色"完成创建

### 2. 音色融合

1. 选择"音色融合"标签页
2. 输入新音色名称和描述
3. 选择要融合的音色并设置权重
4. 系统自动计算归一化权重
5. 点击"创建融合音色"

### 3. 语音合成

1. 选择"语音合成"标签页
2. 选择要使用的音色
3. 输入要合成的文本
4. 选择情感控制方式：
   - 使用说话人情感
   - 情感参考音频
   - 情感向量控制
   - 情感文本描述
5. 调整高级参数（可选）
6. 点击"开始合成"

### 4. 音色管理

1. 选择"音色管理"标签页
2. 查看所有已创建的音色
3. 支持导出/导入音色配置
4. 查看统计信息

## 配置说明

系统使用YAML格式的配置文件，主要配置项包括：

```yaml
ddsp_svc:
  model_dir: "DDSP-SVC/exp"
  default_f0_predictor: "rmvpe"
  default_f0_min: 50.0
  default_f0_max: 1100.0

index_tts:
  model_dir: "index-tts/checkpoints"
  use_fp16: false
  use_cuda_kernel: false

system:
  device: "auto"  # auto, cpu, cuda, mps
  log_level: "INFO"
  voices_dir: "voices"
  outputs_dir: "outputs"

ui:
  host: "0.0.0.0"
  port: 7860
  share: false
```

## 开发指南

### 代码风格

项目遵循Linus Torvalds的代码哲学：

1. **好品味**：消除特殊情况，让代码简洁优雅
2. **最新实现**：避免向后兼容，保持最新实现
3. **实用主义**：解决实际问题，而非虚构的威胁
4. **简洁至上**：避免过高复杂度的函数

### 核心设计原则

1. **数据结构优先**："糟糕的程序员担心代码，优秀的程序员担心数据结构"
2. **单一职责**：每个模块只做一件事，并把它做好
3. **错误处理**：提供有意义的错误信息，避免静默失败
4. **性能优化**：关注实际性能瓶颈，避免过早优化

### 测试

```bash
# 运行核心功能测试
python tests/test_core.py

# 运行所有测试
python -m pytest tests/
```

### 日志

系统提供完整的日志记录：

- 控制台输出：彩色格式，便于开发调试
- 文件记录：详细格式，包含文件名和行号
- 错误日志：单独记录ERROR级别以上的日志
- 性能监控：记录关键操作的执行时间

## API参考

### 核心类

#### VoiceConfig
音色配置的核心数据结构，包含DDSP-SVC和IndexTTS的所有参数。

#### WeightCalculator
权重计算工具，支持：
- 自动归一化
- 多层级权重合并
- 权重插值

#### VoiceManager
音色管理器，提供：
- 音色保存/加载
- 音色搜索
- 音色融合
- 导入/导出

### 集成模块

#### DDSPSVCIntegration
DDSP-SVC集成接口，支持：
- 模型加载
- 音色转换
- 多说话人融合

#### IndexTTSIntegration
IndexTTS集成接口，支持：
- 情感语音合成
- 多种情感控制方式
- 文本情感分析

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件完整性
   - 查看日志文件获取详细错误信息

2. **GPU内存不足**
   - 减少batch size
   - 启用FP16模式
   - 使用CPU模式

3. **音频质量问题**
   - 检查输入音频质量
   - 调整F0参数范围
   - 优化权重配置

4. **Web界面无法访问**
   - 检查端口是否被占用
   - 确认防火墙设置
   - 查看控制台错误信息

### 性能优化

1. **GPU加速**
   - 使用CUDA版本的PyTorch
   - 启用CUDA内核（如果支持）
   - 使用FP16精度

2. **内存优化**
   - 定期清理缓存
   - 使用流式处理长音频
   - 优化模型加载策略

3. **并发处理**
   - 使用异步处理
   - 实现任务队列
   - 限制并发数量

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

请确保：
- 遵循项目代码风格
- 添加适当的测试
- 更新相关文档
- 通过所有检查

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 致谢

- [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC) - 音色转换技术
- [IndexTTS](https://github.com/X-T-E-R/IndexTTS) - 情感语音合成技术
- [Gradio](https://gradio.app/) - Web界面框架

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件
- 加入讨论群

---

**注意**：本项目仅供学习和研究使用，请遵守相关法律法规，不得用于非法用途。
