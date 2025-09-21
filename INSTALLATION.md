# CreatYourVoice 安装和使用指南

## 系统要求

- **Python**: 3.10 或更高版本
- **操作系统**: Windows 10/11, Linux, macOS
- **GPU**: 推荐使用 NVIDIA GPU (CUDA 支持)，也支持 CPU 运行
- **内存**: 推荐 16GB 或更多
- **存储**: 至少 10GB 可用空间

## 安装步骤

### 1. 克隆项目

```bash
git clone --recursive https://github.com/your-repo/CreatYourVoice.git
cd CreatYourVoice
```

### 2. 安装依赖

本项目使用 `uv` 包管理器。如果您还没有安装 `uv`，请先安装：

```bash
# 安装 uv
pip install uv
```

然后安装项目依赖：

```bash
# 安装所有依赖
uv sync

# 激活虚拟环境
uv shell
```

### 3. 下载预训练模型

#### DDSP-SVC 模型
请将 DDSP-SVC 模型文件放置在 `DDSP-SVC/exp/` 目录下。

#### IndexTTS 模型
请将 IndexTTS 模型文件放置在 `index-tts/checkpoints/` 目录下。

### 4. 配置设置

首次运行时，系统会自动创建 `config.yaml` 配置文件。您可以根据需要修改配置：

```yaml
system:
  device: "auto"  # auto, cpu, cuda
  log_level: "INFO"
  log_file: "logs/app.log"

ui:
  host: "0.0.0.0"
  port: 7860
  share: false
  debug: false

gpu:
  auto_cleanup_enabled: true
  memory_monitoring_enabled: true
```

## 使用方法

### 启动 Web 界面

```bash
# 启动默认 Web 界面
python main.py

# 指定主机和端口
python main.py --host 0.0.0.0 --port 7860

# 创建公共链接（用于远程访问）
python main.py --share

# 调试模式
python main.py --debug
```

### 命令行工具

```bash
# 显示配置信息
python main.py config

# 系统检查
python main.py check

# 查看音色列表
python main.py voices

# GPU 状态检查
python main.py gpu
```

## 功能说明

### 1. 角色声音基底创建
- 选择预设音色标签（童男、童女、少男、少女等）
- 配置 DDSP-SVC 变声器参数
- 多说话人音色融合
- 实时预览和参数调整

### 2. 语音合成
- 基于角色声音基底的语音合成
- 多种情感控制模式
- 长文本智能分句处理
- 合成历史管理

### 3. 音色融合和继承
- 从现有音色创建新音色
- 多音色权重融合
- 复杂的继承链管理

### 4. GPU 模型管理
- 智能模型加载/卸载
- GPU 内存监控
- 自动内存优化
- 多 GPU 支持

### 5. 版本支持
- 同时支持 DDSP-SVC 6.1 和 6.3 版本
- 自动版本检测
- 统一的推理接口

## 故障排除

### 常见问题

1. **日志路径错误**
   ```
   环境设置失败: [Errno 2] No such file or directory: 'logs/logs/app.log'
   ```
   **解决方案**: 这个问题已在最新版本中修复。请确保使用最新的代码。

2. **依赖缺失**
   ```
   No module named 'torchaudio'
   ```
   **解决方案**: 运行 `uv sync` 重新安装依赖。

3. **GPU 内存不足**
   ```
   CUDA out of memory
   ```
   **解决方案**:
   - 使用 GPU 模型管理功能手动卸载不需要的模型
   - 在配置中启用自动内存清理
   - 降低批处理大小

4. **模型文件未找到**
   ```
   模型目录不存在
   ```
   **解决方案**: 确保模型文件放置在正确的目录下，参考安装步骤第3步。

### 性能优化

1. **GPU 设置**
   - 确保 CUDA 正确安装
   - 使用 `python main.py gpu` 检查 GPU 状态
   - 启用自动内存管理

2. **内存优化**
   - 定期清理不使用的模型
   - 使用较小的批处理大小
   - 启用音频缓存

3. **网络设置**
   - 如需远程访问，使用 `--share` 参数
   - 调整 `--host` 和 `--port` 参数

## 开发和贡献

### 运行测试

```bash
# 运行所有测试
python tests/run_all_tests.py

# 运行特定测试
python -m pytest tests/test_core.py
```

### 代码质量检查

```bash
# 代码格式化
uv run ruff format .

# 代码检查
uv run ruff check .

# 类型检查
uv run mypy src/
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 支持

如果您遇到问题或有建议，请：

1. 查看本文档的故障排除部分
2. 检查 [GitHub Issues](https://github.com/your-repo/CreatYourVoice/issues)
3. 创建新的 Issue 描述您的问题

## 更新日志

### v1.0.0
- 完整的音色创建和管理系统
- GPU 模型管理功能
- DDSP-SVC 6.1/6.3 版本支持
- 完善的 Web 界面
- 详细的文档和测试
