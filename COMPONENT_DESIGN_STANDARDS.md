# CreatYourVoice 组件设计规范

## 设计哲学

遵循用户明确的需求：
- **单栏引导式设计**：用户逐步完成工作流
- **权重计算机械化**：用户输入任意数字，系统自动归一化
- **两阶段工作流**：角色声音基底创建 + 语音合成

## 核心组件规范

### 1. 权重输入组件

```python
class WeightInputComponent:
    """权重输入组件 - 支持任意数字输入"""

    def create_weight_input(self, label: str, info: str = "可输入任意数字") -> gr.Number:
        """创建权重输入框"""
        return gr.Number(
            label=label,
            value=0,
            minimum=0,
            info=info,
            elem_classes=["weight-input"]
        )

    def create_weight_display(self, label: str = "计算后的权重分布") -> gr.JSON:
        """创建权重显示组件"""
        return gr.JSON(
            label=label,
            value={},
            elem_classes=["weight-display"]
        )

    def calculate_normalized_weights(self, *weights) -> Dict[str, float]:
        """权重归一化计算"""
        valid_weights = [(i, w) for i, w in enumerate(weights) if w > 0]
        if not valid_weights:
            return {}

        total = sum(w for _, w in valid_weights)
        return {f"item_{i}": w/total for i, w in valid_weights}
```

### 2. 步骤引导组件

```python
class StepGuideComponent:
    """步骤引导组件 - 单栏布局的工作流引导"""

    def create_step_header(self, step_num: int, title: str, description: str = "") -> gr.HTML:
        """创建步骤标题"""
        html_content = f"""
        <div class="step-header">
            <div class="step-indicator">
                <span class="step-number">{step_num}</span>
                <span class="step-title">{title}</span>
            </div>
            {f'<p class="step-description">{description}</p>' if description else ''}
        </div>
        """
        return gr.HTML(html_content, elem_classes=["step-guide"])

    def create_progress_indicator(self, current_step: int, total_steps: int) -> gr.HTML:
        """创建进度指示器"""
        progress_html = f"""
        <div class="progress-container">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {(current_step/total_steps)*100}%"></div>
            </div>
            <span class="progress-text">步骤 {current_step} / {total_steps}</span>
        </div>
        """
        return gr.HTML(progress_html, elem_classes=["progress-indicator"])
```

### 3. 音频标签选择组件

```python
class AudioTagComponent:
    """音频标签选择组件"""

    def create_tag_selector(self) -> Tuple[gr.Dropdown, gr.Audio, gr.JSON]:
        """创建音频标签选择器"""

        # 标签下拉框
        tag_dropdown = gr.Dropdown(
            label="音频标签",
            choices=self._get_audio_tag_choices(),
            info="选择音色类型（童男、童女、少男、少女、青年男、青年女等）",
            elem_classes=["audio-tag-selector"]
        )

        # 标签音频预览
        tag_audio = gr.Audio(
            label="标签音频预览",
            visible=False,
            elem_classes=["tag-audio-preview"]
        )

        # 标签信息显示
        tag_info = gr.JSON(
            label="标签信息",
            value={},
            visible=False,
            elem_classes=["tag-info-display"]
        )

        return tag_dropdown, tag_audio, tag_info

    def _get_audio_tag_choices(self) -> List[Tuple[str, str]]:
        """获取音频标签选择列表"""
        return [
            ("童男 - 儿童男性音色", "child_male"),
            ("童女 - 儿童女性音色", "child_female"),
            ("少男 - 少年男性音色", "teen_male"),
            ("少女 - 少年女性音色", "teen_female"),
            ("青年男 - 青年男性音色", "young_male"),
            ("青年女 - 青年女性音色", "young_female"),
            ("中年男 - 中年男性音色", "middle_male"),
            ("中年女 - 中年女性音色", "middle_female"),
            ("老年男 - 老年男性音色", "elder_male"),
            ("老年女 - 老年女性音色", "elder_female")
        ]
```

### 4. DDSP-SVC参数组件

```python
class DDSPParameterComponent:
    """DDSP-SVC参数设置组件"""

    def create_ddsp_controls(self) -> Tuple[gr.Slider, gr.Slider]:
        """创建DDSP-SVC控制参数"""

        pitch_shift = gr.Slider(
            minimum=-12,
            maximum=12,
            value=0,
            step=0.1,
            label="音调偏移",
            info="正值升高音调，负值降低音调",
            elem_classes=["ddsp-param-slider"]
        )

        voice_thickness = gr.Slider(
            minimum=-12,
            maximum=12,
            value=0,
            step=0.1,
            label="声音粗细",
            info="调整声音的厚度和质感",
            elem_classes=["ddsp-param-slider"]
        )

        return pitch_shift, voice_thickness
```

### 5. 说话人选择组件

```python
class SpeakerSelectionComponent:
    """说话人选择和权重设置组件"""

    def create_speaker_rows(self, max_speakers: int = 8) -> List[Tuple]:
        """创建说话人选择行"""
        speaker_components = []

        for i in range(max_speakers):
            with gr.Row(visible=False, elem_classes=["speaker-row"]) as speaker_row:
                speaker_checkbox = gr.Checkbox(
                    label="选择",
                    value=False,
                    elem_classes=["speaker-checkbox"]
                )
                speaker_name = gr.Textbox(
                    label="说话人",
                    interactive=False,
                    scale=2,
                    elem_classes=["speaker-name"]
                )
                speaker_weight = gr.Number(
                    label="权重",
                    value=0,
                    scale=1,
                    info="可输入任意数字",
                    elem_classes=["speaker-weight"]
                )

            speaker_components.append((speaker_row, speaker_checkbox, speaker_name, speaker_weight))

        return speaker_components

    def update_speaker_display(self, tag_info: Dict, speaker_components: List) -> List[gr.update]:
        """更新说话人显示"""
        updates = []
        speakers = tag_info.get("speakers", [])

        for i, (row, checkbox, name, weight) in enumerate(speaker_components):
            if i < len(speakers):
                speaker = speakers[i]
                updates.extend([
                    gr.Row(visible=True),
                    gr.Checkbox(value=False),
                    gr.Textbox(value=f"{speaker['name']} ({speaker['id']})"),
                    gr.Number(value=0)
                ])
            else:
                updates.extend([
                    gr.Row(visible=False),
                    gr.Checkbox(value=False),
                    gr.Textbox(value=""),
                    gr.Number(value=0)
                ])

        return updates
```

### 6. 情感控制组件

```python
class EmotionControlComponent:
    """情感控制组件 - 用于语音合成"""

    def create_emotion_controls(self) -> Tuple:
        """创建情感控制界面"""

        # 情感模式选择
        emotion_mode = gr.Radio(
            choices=[
                ("普通模式（无情感）", "normal"),
                ("情感描述", "description"),
                ("情感参考音频", "reference"),
                ("高级模式（情感向量）", "vector")
            ],
            value="normal",
            label="情感模式",
            elem_classes=["emotion-mode-selector"]
        )

        # 情感描述组
        with gr.Group(visible=False, elem_classes=["emotion-group"]) as emotion_desc_group:
            emotion_description = gr.Textbox(
                label="情感描述",
                placeholder="例如：开心、激动、温柔、悲伤...",
                info="描述想要的情感特征",
                elem_classes=["emotion-description"]
            )

        # 情感参考组
        with gr.Group(visible=False, elem_classes=["emotion-group"]) as emotion_ref_group:
            emotion_reference = gr.Audio(
                label="情感参考音频",
                type="filepath",
                info="上传包含目标情感的音频文件",
                elem_classes=["emotion-reference"]
            )

        # 情感向量组
        with gr.Group(visible=False, elem_classes=["emotion-group"]) as emotion_vector_group:
            gr.Markdown("#### IndexTTS v2 情感向量参数")
            emotion_sliders = []
            emotion_names = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "平静", "激动"]

            with gr.Row():
                for i, name in enumerate(emotion_names[:4]):
                    slider = gr.Slider(
                        0, 1, 0,
                        label=name,
                        step=0.01,
                        elem_classes=["emotion-slider"]
                    )
                    emotion_sliders.append(slider)

            with gr.Row():
                for i, name in enumerate(emotion_names[4:]):
                    slider = gr.Slider(
                        0, 1, 0,
                        label=name,
                        step=0.01,
                        elem_classes=["emotion-slider"]
                    )
                    emotion_sliders.append(slider)

        return (emotion_mode, emotion_desc_group, emotion_ref_group,
                emotion_vector_group, emotion_description, emotion_reference, emotion_sliders)
```

### 7. 预览和保存组件

```python
class PreviewSaveComponent:
    """预览和保存组件"""

    def create_preview_section(self, preview_type: str = "voice_base") -> Tuple:
        """创建预览区域"""

        # 名称输入
        if preview_type == "voice_base":
            name_input = gr.Textbox(
                label="角色声音基底名称",
                placeholder="输入名称",
                elem_classes=["name-input"]
            )
        else:
            name_input = gr.Textbox(
                label="合成产物名称",
                placeholder="为这次合成起个名字（可选）",
                elem_classes=["name-input"]
            )

        # 操作按钮
        with gr.Row(elem_classes=["action-buttons"]):
            preview_btn = gr.Button(
                "🎧 生成预览",
                variant="secondary",
                elem_classes=["preview-button"]
            )
            save_btn = gr.Button(
                "💾 保存" + ("基底" if preview_type == "voice_base" else "产物"),
                variant="primary",
                visible=False,
                elem_classes=["save-button"]
            )

        # 状态显示
        status_display = gr.Textbox(
            label="状态",
            interactive=False,
            lines=3,
            elem_classes=["status-display"]
        )

        # 预览音频
        preview_audio = gr.Audio(
            label="预览音频",
            visible=False,
            elem_classes=["preview-audio"]
        )

        return name_input, preview_btn, save_btn, status_display, preview_audio
```

### 8. 产物管理组件

```python
class ProductManagementComponent:
    """语音产物和声音基底管理组件"""

    def create_product_selector(self, product_type: str = "all") -> gr.Dropdown:
        """创建产物选择器"""
        return gr.Dropdown(
            label="选择产物",
            choices=self._get_product_choices(product_type),
            info="选择现有的语音产物或声音基底",
            elem_classes=["product-selector"]
        )

    def create_product_info_display(self) -> gr.JSON:
        """创建产物信息显示"""
        return gr.JSON(
            label="产物信息",
            value={},
            elem_classes=["product-info"]
        )

    def _get_product_choices(self, product_type: str) -> List[Tuple[str, str]]:
        """获取产物选择列表"""
        # 这里应该从实际的数据库或文件系统获取
        return [
            ("示例声音基底1", "base_001"),
            ("示例语音产物1", "product_001"),
            ("示例声音基底2", "base_002")
        ]
```

## CSS样式规范

### 基础样式

```css
/* 全局容器 - 单栏布局 */
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 步骤引导样式 */
.step-guide {
    margin-bottom: 24px;
}

.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 8px;
    margin-bottom: 16px;
}

.step-indicator {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    font-weight: bold;
    margin-right: 12px;
    font-size: 16px;
}

.step-title {
    font-size: 18px;
    font-weight: 600;
}

.step-description {
    margin: 0;
    opacity: 0.9;
    font-size: 14px;
}

/* 进度指示器 */
.progress-container {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.progress-bar {
    flex: 1;
    height: 8px;
    background: #e5e7eb;
    border-radius: 4px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    transition: width 0.3s ease;
}

.progress-text {
    font-size: 14px;
    color: #6b7280;
    white-space: nowrap;
}

/* 权重相关组件 */
.weight-input input {
    text-align: center;
    font-weight: 500;
}

.weight-display {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px;
}

/* 说话人行样式 */
.speaker-row {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
}

.speaker-checkbox {
    margin-right: 8px;
}

.speaker-name input {
    background: transparent;
    border: none;
    font-weight: 500;
}

.speaker-weight input {
    background: white;
    border: 1px solid #d1d5db;
    border-radius: 4px;
}

/* 情感控制组件 */
.emotion-group {
    background: #fef7f0;
    border: 1px solid #fed7aa;
    border-radius: 6px;
    padding: 16px;
    margin-top: 12px;
}

.emotion-slider .gradio-slider {
    margin-bottom: 8px;
}

/* 操作按钮 */
.action-buttons {
    margin: 20px 0;
    gap: 12px;
}

.preview-button {
    background: #6b7280;
    border: none;
    color: white;
    font-weight: 500;
}

.save-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: 500;
}

/* 状态显示 */
.status-display textarea {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
    font-size: 13px;
}

/* 音频组件 */
.preview-audio {
    margin-top: 16px;
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 16px;
}

/* 响应式设计 */
@media (max-width: 768px) {
    .gradio-container {
        max-width: 100% !important;
        padding: 16px;
    }

    .step-header {
        padding: 12px 16px;
    }

    .step-number {
        width: 28px;
        height: 28px;
        font-size: 14px;
    }

    .action-buttons {
        flex-direction: column;
    }

    .gradio-row {
        flex-direction: column !important;
    }
}
```

## 事件处理规范

### 权重计算事件

```python
def bind_weight_calculation_events(self, weight_inputs: List[gr.Number],
                                 weight_display: gr.JSON):
    """绑定权重计算事件"""
    for
