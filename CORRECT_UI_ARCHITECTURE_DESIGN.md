# CreatYourVoice 正确的UI架构设计

## 林纳斯式重新分析

### 真实问题理解
**这是真实存在的问题还是想象出来的？**
用户明确了真实需求：两阶段工作流（1.创建角色声音基底 2.使用基底合成语音），我之前的简化分析偏离了实际需求。

**有没有更简单的方法？**
用户的工作流已经是最简单的方法：DDSP-SVC处理音色特征 → IndexTTS v2处理情感合成。

**这会破坏任何东西吗？**
需要保持用户定义的完整工作流，不能随意简化核心功能。

## 正确的需求分析

### 核心工作流
1. **角色声音基底创建**：
   - 从零开始：选择音频标签 → 设置DDSP-SVC参数 → 选择说话人权重 → 预览 → 保存
   - 从现有产物：加载历史产物 → 配置权重 → 添加新参数 → 重新计算 → 保存
   - 融合现有产物：选择多个产物 → 配置权重 → 计算融合参数 → 保存

2. **使用音色合成语音**：
   - 输入文本 → 选择角色声音基底 → 设置情感参数 → IndexTTS v2合成

### 关键数据结构

```python
@dataclass
class VoiceBase:
    """角色声音基底 - 核心数据结构"""
    name: str
    base_id: str

    # DDSP-SVC参数
    audio_tag: str  # 音频标签
    ddsp_params: Dict[str, float]  # 音调、声音粗细等
    speaker_weights: Dict[str, float]  # {speaker_id: weight}

    # IndexTTS参数（无情感的基础参数）
    index_tts_params: Dict[str, Any]

    # 元数据
    created_at: datetime
    base_audio_path: str  # 生成的基底音频文件

@dataclass
class SynthesisProduct:
    """语音合成产物"""
    product_id: str
    text: str
    voice_base_id: str

    # 情感参数
    emotion_mode: str  # "description", "reference", "vector"
    emotion_params: Dict[str, Any]

    # 结果
    audio_path: str
    created_at: datetime
```

### 权重计算规则

```python
def calculate_weights(user_inputs: List[float]) -> Dict[str, float]:
    """
    用户可以输入任意大的数字，系统机械计算权重
    例如：用户输入 A=1000, B=4000 → A=0.2, B=0.8
    """
    total = sum(user_inputs)
    if total == 0:
        return {}
    return {f"item_{i}": value/total for i, value in enumerate(user_inputs)}

def merge_voice_weights(old_weights: Dict[str, float], old_weight_ratio: float,
                       new_weights: Dict[str, float]) -> Dict[str, float]:
    """
    融合权重计算：
    1. 旧权重 * 旧权重比例
    2. 新权重 * (1 - 旧权重比例)
    3. 按speaker_id分别加和
    4. 最终归一化到总和为1
    """
    result = {}

    # 处理旧权重
    for speaker_id, weight in old_weights.items():
        result[speaker_id] = weight * old_weight_ratio

    # 处理新权重
    new_weight_ratio = 1.0 - old_weight_ratio
    for speaker_id, weight in new_weights.items():
        if speaker_id in result:
            result[speaker_id] += weight * new_weight_ratio
        else:
            result[speaker_id] = weight * new_weight_ratio

    # 归一化
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}

    return result
```

## UI架构设计

### 主界面结构（单栏设计）

```python
class CreatYourVoiceApp:
    """主应用 - 单栏引导式设计"""

    def create_interface(self):
        with gr.Blocks(title="CreatYourVoice", css=self._get_css()) as interface:
            # 标题
            gr.HTML("<h1>🎵 CreatYourVoice</h1>")

            # 主要工作流选择
            with gr.Tabs() as main_tabs:
                # Tab 1: 创建角色声音基底
                with gr.Tab("🎨 创建角色声音基底"):
                    self._create_voice_base_creation_tab()

                # Tab 2: 使用音色合成语音
                with gr.Tab("🎤 语音合成"):
                    self._create_speech_synthesis_tab()

                # Tab 3: 管理音色库
                with gr.Tab("📁 音色管理"):
                    self._create_voice_management_tab()

        return interface
```

### Tab 1: 创建角色声音基底

```python
def _create_voice_base_creation_tab(self):
    """创建角色声音基底Tab"""

    # 创建方式选择
    creation_mode = gr.Radio(
        choices=[
            ("从零开始创建", "from_scratch"),
            ("从现有产物创建", "from_existing"),
            ("融合现有产物", "merge_existing")
        ],
        value="from_scratch",
        label="创建方式"
    )

    # 从零开始创建
    with gr.Group(visible=True) as from_scratch_group:
        self._create_from_scratch_interface()

    # 从现有产物创建
    with gr.Group(visible=False) as from_existing_group:
        self._create_from_existing_interface()

    # 融合现有产物
    with gr.Group(visible=False) as merge_existing_group:
        self._create_merge_existing_interface()

    # 绑定模式切换
    creation_mode.change(
        fn=self._switch_creation_mode,
        inputs=[creation_mode],
        outputs=[from_scratch_group, from_existing_group, merge_existing_group]
    )

def _create_from_scratch_interface(self):
    """从零开始创建界面"""

    # 步骤1：选择音频标签
    gr.Markdown("### 步骤1：选择音频标签")
    audio_tag_dropdown = gr.Dropdown(
        label="音频标签",
        choices=self._get_audio_tag_choices(),
        info="选择音色类型（童男、童女、少男、少女、青年男、青年女等）"
    )

    # 显示标签对应的音频
    tag_audio_player = gr.Audio(label="标签音频预览", visible=False)

    # 步骤2：DDSP-SVC变声器参数
    gr.Markdown("### 步骤2：设置变声器参数")
    with gr.Row():
        pitch_shift = gr.Slider(-12, 12, 0, label="音调偏移", step=0.1)
        voice_thickness = gr.Slider(-12, 12, 0, label="声音粗细", step=0.1)

    # 步骤3：选择说话人和权重
    gr.Markdown("### 步骤3：选择说话人和权重")

    # 显示当前标签对应的说话人列表
    speakers_info = gr.JSON(label="可用说话人", value={})

    # 说话人选择和权重设置（动态生成）
    speaker_components = []
    for i in range(8):  # 最多8个说话人
        with gr.Row(visible=False) as speaker_row:
            speaker_checkbox = gr.Checkbox(label="选择", value=False)
            speaker_name = gr.Textbox(label="说话人", interactive=False, scale=2)
            speaker_weight = gr.Number(label="权重", value=0, scale=1, info="可输入任意数字")
        speaker_components.append((speaker_row, speaker_checkbox, speaker_name, speaker_weight))

    # 权重计算显示
    calculated_weights = gr.JSON(label="计算后的权重分布", value={})

    # 步骤4：预览和保存
    gr.Markdown("### 步骤4：预览和保存")

    voice_base_name = gr.Textbox(label="角色声音基底名称", placeholder="输入名称")

    with gr.Row():
        preview_btn = gr.Button("🎧 生成预览", variant="secondary")
        save_btn = gr.Button("💾 保存基底", variant="primary", visible=False)

    # 预览结果
    preview_audio = gr.Audio(label="预览音频", visible=False)
    creation_status = gr.Textbox(label="创建状态", interactive=False, lines=3)

    # 绑定事件
    self._bind_from_scratch_events(
        audio_tag_dropdown, tag_audio_player, speakers_info,
        speaker_components, calculated_weights, preview_btn, save_btn,
        preview_audio, creation_status, voice_base_name,
        pitch_shift, voice_thickness
    )

def _create_from_existing_interface(self):
    """从现有产物创建界面"""

    # 步骤1：选择现有产物
    gr.Markdown("### 步骤1：选择现有语音产物或声音基底")
    existing_product = gr.Dropdown(
        label="现有产物",
        choices=self._get_existing_products(),
        info="选择一个历史语音产物或现有声音基底"
    )

    # 显示现有产物信息
    existing_info = gr.JSON(label="现有产物信息", value={})

    # 步骤2：配置权重
    gr.Markdown("### 步骤2：配置现有产物权重")
    existing_weight = gr.Slider(
        0, 1, 0.5,
        label="现有产物权重",
        step=0.01,
        info="现有产物在新基底中的权重比例"
    )

    # 步骤3：添加新参数（复用从零开始的界面）
    gr.Markdown("### 步骤3：添加新的音色参数")
    # ... 复用从零开始的参数设置界面

    # 步骤4：权重计算显示
    gr.Markdown("### 步骤4：权重计算结果")
    final_weights = gr.JSON(label="最终权重分布", value={})

    # 预览和保存
    # ... 类似从零开始的预览保存界面

def _create_merge_existing_interface(self):
    """融合现有产物界面"""

    # 步骤1：选择多个产物
    gr.Markdown("### 步骤1：选择要融合的产物")

    merge_components = []
    for i in range(5):  # 最多融合5个产物
        with gr.Row():
            product_dropdown = gr.Dropdown(
                label=f"产物{i+1}",
                choices=self._get_existing_products(),
                scale=2
            )
            product_weight = gr.Number(
                label="权重",
                value=0,
                scale=1,
                info="可输入任意数字"
            )
        merge_components.append((product_dropdown, product_weight))

    # 步骤2：权重计算
    gr.Markdown("### 步骤2：权重计算结果")
    merge_weights = gr.JSON(label="融合权重分布", value={})

    # 步骤3：预览和保存
    # ... 预览保存界面
```

### Tab 2: 使用音色合成语音

```python
def _create_speech_synthesis_tab(self):
    """语音合成Tab"""

    # 步骤1：输入文本
    gr.Markdown("### 步骤1：输入要合成的文本")
    synthesis_text = gr.Textbox(
        label="合成文本",
        placeholder="请输入要朗读的文本...",
        lines=4
    )

    # 步骤2：选择角色声音基底
    gr.Markdown("### 步骤2：选择角色声音基底")
    voice_base_dropdown = gr.Dropdown(
        label="角色声音基底",
        choices=self._get_voice_bases(),
        info="选择已创建的角色声音基底"
    )

    # 显示基底信息
    base_info = gr.JSON(label="基底信息", value={})

    # 步骤3：情感控制
    gr.Markdown("### 步骤3：情感控制（可选）")

    emotion_mode = gr.Radio(
        choices=[
            ("普通模式（无情感）", "normal"),
            ("情感描述", "description"),
            ("情感参考音频", "reference"),
            ("高级模式（情感向量）", "vector")
        ],
        value="normal",
        label="情感模式"
    )

    # 情感描述模式
    with gr.Group(visible=False) as emotion_desc_group:
        emotion_description = gr.Textbox(
            label="情感描述",
            placeholder="例如：开心、激动、温柔、悲伤...",
            info="描述想要的情感特征"
        )

    # 情感参考模式
    with gr.Group(visible=False) as emotion_ref_group:
        emotion_reference = gr.Audio(
            label="情感参考音频",
            type="filepath",
            info="上传包含目标情感的音频文件"
        )

    # 高级模式（情感向量）
    with gr.Group(visible=False) as emotion_vector_group:
        gr.Markdown("#### IndexTTS v2 情感向量参数")
        # 根据IndexTTS v2的具体参数要求设置
        emotion_sliders = []
        emotion_names = ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶", "平静", "激动"]
        for name in emotion_names:
            slider = gr.Slider(0, 1, 0, label=name, step=0.01)
            emotion_sliders.append(slider)

    # 步骤4：合成
    gr.Markdown("### 步骤4：开始合成")

    synthesis_name = gr.Textbox(
        label="合成产物名称",
        placeholder="为这次合成起个名字（可选）"
    )

    with gr.Row():
        synthesize_btn = gr.Button("🎤 开始合成", variant="primary")
        save_product_btn = gr.Button("💾 保存产物", visible=False)

    # 合成结果
    synthesis_status = gr.Textbox(label="合成状态", interactive=False, lines=3)
    result_audio = gr.Audio(label="合成结果", visible=False)

    # 绑定事件
    self._bind_synthesis_events(
        synthesis_text, voice_base_dropdown, base_info,
        emotion_mode, emotion_desc_group, emotion_ref_group, emotion_vector_group,
        emotion_description, emotion_reference, emotion_sliders,
        synthesize_btn, save_product_btn, synthesis_status, result_audio
    )
```

## 响应式设计

### 单栏布局CSS

```css
/* 单栏引导式布局 */
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto;
    padding: 20px;
}

/* 步骤引导样式 */
.step-guide {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.step-number {
    display: inline-block;
    width: 30px;
    height: 30px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    text-align: center;
    line-height: 30px;
    margin-right: 10px;
    font-weight: bold;
}

/*
