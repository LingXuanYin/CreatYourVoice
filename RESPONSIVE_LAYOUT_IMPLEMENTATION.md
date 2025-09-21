# 自适应页面布局实现方案

## 设计原则

基于用户需求的单栏引导式设计：
- **移动优先**：从小屏幕开始设计，逐步适配大屏幕
- **单栏布局**：避免复杂的多栏布局，确保工作流清晰
- **渐进增强**：在大屏幕上提供更好的体验，但不破坏小屏幕的可用性

## 响应式断点设计

### 断点定义

```css
/* 断点定义 */
:root {
    --mobile-max: 767px;      /* 手机 */
    --tablet-min: 768px;      /* 平板开始 */
    --tablet-max: 1023px;     /* 平板结束 */
    --desktop-min: 1024px;    /* 桌面开始 */
    --desktop-max: 1439px;    /* 桌面结束 */
    --large-min: 1440px;      /* 大屏开始 */
}

/* 容器宽度 */
.gradio-container {
    width: 100%;
    margin: 0 auto;
    padding: 16px;
}

/* 手机端 (320px - 767px) */
@media (max-width: 767px) {
    .gradio-container {
        max-width: 100%;
        padding: 12px;
    }
}

/* 平板端 (768px - 1023px) */
@media (min-width: 768px) and (max-width: 1023px) {
    .gradio-container {
        max-width: 720px;
        padding: 20px;
    }
}

/* 桌面端 (1024px - 1439px) */
@media (min-width: 1024px) and (max-width: 1439px) {
    .gradio-container {
        max-width: 800px;
        padding: 24px;
    }
}

/* 大屏端 (1440px+) */
@media (min-width: 1440px) {
    .gradio-container {
        max-width: 900px;
        padding: 32px;
    }
}
```

## 组件响应式设计

### 1. 步骤引导组件

```css
/* 步骤引导 - 基础样式 */
.step-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
    margin-bottom: 16px;
    padding: 16px 20px;
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
    flex-shrink: 0;
}

.step-title {
    font-size: 18px;
    font-weight: 600;
    line-height: 1.3;
}

/* 手机端适配 */
@media (max-width: 767px) {
    .step-header {
        padding: 12px 16px;
        margin-bottom: 12px;
    }

    .step-number {
        width: 28px;
        height: 28px;
        font-size: 14px;
        margin-right: 8px;
    }

    .step-title {
        font-size: 16px;
    }

    .step-description {
        font-size: 13px;
        line-height: 1.4;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .step-header {
        padding: 14px 18px;
    }

    .step-number {
        width: 30px;
        height: 30px;
        font-size: 15px;
        margin-right: 10px;
    }

    .step-title {
        font-size: 17px;
    }
}
```

### 2. 权重输入组件

```css
/* 权重输入 - 基础样式 */
.weight-input-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    padding: 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
}

.weight-input-row .gradio-checkbox {
    flex-shrink: 0;
    margin-right: 8px;
}

.weight-input-row .gradio-textbox {
    flex: 2;
    min-width: 0;
}

.weight-input-row .gradio-number {
    flex: 1;
    min-width: 80px;
}

/* 手机端适配 */
@media (max-width: 767px) {
    .weight-input-row {
        flex-direction: column;
        align-items: stretch;
        gap: 8px;
        padding: 10px;
    }

    .weight-input-row .gradio-checkbox {
        margin-right: 0;
        margin-bottom: 4px;
    }

    .weight-input-row .gradio-textbox,
    .weight-input-row .gradio-number {
        flex: none;
        width: 100%;
    }

    /* 权重输入框在手机端使用更大的触摸目标 */
    .weight-input-row .gradio-number input {
        min-height: 44px;
        font-size: 16px;
        text-align: center;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .weight-input-row {
        gap: 10px;
        padding: 11px;
    }

    .weight-input-row .gradio-number {
        min-width: 90px;
    }
}
```

### 3. 音频组件

```css
/* 音频组件 - 基础样式 */
.audio-component {
    margin: 16px 0;
    border: 2px dashed #d1d5db;
    border-radius: 8px;
    padding: 16px;
    background: #fafafa;
}

.audio-component.has-audio {
    border-color: #667eea;
    background: #f0f4ff;
}

/* 手机端适配 */
@media (max-width: 767px) {
    .audio-component {
        margin: 12px 0;
        padding: 12px;
        border-radius: 6px;
    }

    /* 音频控件在手机端使用全宽 */
    .audio-component audio {
        width: 100%;
        min-height: 40px;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .audio-component {
        padding: 14px;
    }
}
```

### 4. 按钮组件

```css
/* 按钮组 - 基础样式 */
.action-buttons {
    display: flex;
    gap: 12px;
    margin: 20px 0;
    justify-content: center;
}

.action-buttons .gradio-button {
    min-width: 120px;
    height: 44px;
    font-weight: 500;
    border-radius: 6px;
    transition: all 0.2s ease;
}

.preview-button {
    background: #6b7280;
    border: none;
    color: white;
}

.preview-button:hover {
    background: #4b5563;
    transform: translateY(-1px);
}

.save-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
}

.save-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

/* 手机端适配 */
@media (max-width: 767px) {
    .action-buttons {
        flex-direction: column;
        gap: 8px;
        margin: 16px 0;
    }

    .action-buttons .gradio-button {
        width: 100%;
        min-width: auto;
        height: 48px;
        font-size: 16px;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .action-buttons {
        gap: 10px;
    }

    .action-buttons .gradio-button {
        min-width: 140px;
    }
}
```

### 5. 表单组件

```css
/* 表单组件 - 基础样式 */
.form-group {
    margin-bottom: 20px;
}

.form-group .gradio-textbox,
.form-group .gradio-dropdown,
.form-group .gradio-slider {
    margin-bottom: 8px;
}

.form-group label {
    font-weight: 500;
    color: #374151;
    margin-bottom: 4px;
    display: block;
}

/* 输入框样式 */
.form-group input,
.form-group textarea,
.form-group select {
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
    transition: border-color 0.2s ease;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* 手机端适配 */
@media (max-width: 767px) {
    .form-group {
        margin-bottom: 16px;
    }

    .form-group input,
    .form-group textarea,
    .form-group select {
        font-size: 16px; /* 防止iOS缩放 */
        padding: 10px 12px;
        min-height: 44px; /* 触摸友好 */
    }

    .form-group textarea {
        min-height: 100px;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .form-group {
        margin-bottom: 18px;
    }

    .form-group input,
    .form-group textarea,
    .form-group select {
        padding: 9px 12px;
    }
}
```

## 布局容器设计

### 主容器

```css
/* 主应用容器 */
.creat-your-voice-app {
    min-height: 100vh;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 内容容器 */
.app-content {
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}

/* 手机端适配 */
@media (max-width: 767px) {
    .creat-your-voice-app {
        background: white; /* 简化背景 */
    }

    .app-content {
        border-radius: 0;
        box-shadow: none;
        min-height: 100vh;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .app-content {
        border-radius: 8px;
        margin: 20px 0;
    }
}

/* 桌面端适配 */
@media (min-width: 1024px) {
    .app-content {
        margin: 40px 0;
    }
}
```

### Tab容器

```css
/* Tab容器 */
.gradio-tabs {
    background: white;
}

.gradio-tabs .tab-nav {
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    padding: 0;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
}

.gradio-tabs .tab-nav::-webkit-scrollbar {
    display: none;
}

.gradio-tabs .tab-nav button {
    padding: 12px 20px;
    border: none;
    background: transparent;
    color: #6b7280;
    font-weight: 500;
    white-space: nowrap;
    transition: all 0.2s ease;
}

.gradio-tabs .tab-nav button.selected {
    color: #667eea;
    background: white;
    border-bottom: 2px solid #667eea;
}

/* 手机端适配 */
@media (max-width: 767px) {
    .gradio-tabs .tab-nav {
        padding: 0 8px;
    }

    .gradio-tabs .tab-nav button {
        padding: 10px 16px;
        font-size: 14px;
        min-width: 100px;
    }
}

/* 平板端适配 */
@media (min-width: 768px) and (max-width: 1023px) {
    .gradio-tabs .tab-nav button {
        padding: 11px 18px;
    }
}
```

## JavaScript增强

### 响应式行为增强

```javascript
// 响应式行为增强脚本
function initResponsiveEnhancements() {
    // 检测设备类型
    const isMobile = window.innerWidth <= 767;
    const isTablet = window.innerWidth >= 768 && window.innerWidth <= 1023;
    const isDesktop = window.innerWidth >= 1024;

    // 添加设备类型类名
    document.body.classList.add(
        isMobile ? 'device-mobile' :
        isTablet ? 'device-tablet' : 'device-desktop'
    );

    // 移动端优化
    if (isMobile) {
        // 防止双击缩放
        let lastTouchEnd = 0;
        document.addEventListener('touchend', function (event) {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                event.preventDefault();
            }
            lastTouchEnd = now;
        }, false);

        // 优化滚动性能
        document.addEventListener('touchstart', function() {}, {passive: true});
        document.addEventListener('touchmove', function() {}, {passive: true});
    }

    // 窗口大小变化处理
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(function() {
            // 重新检测设备类型
            const newIsMobile = window.innerWidth <= 767;
            const newIsTablet = window.innerWidth >= 768 && window.innerWidth <= 1023;
            const newIsDesktop = window.innerWidth >= 1024;

            // 更新类名
            document.body.className = document.body.className.replace(/device-\w+/g, '');
            document.body.classList.add(
                newIsMobile ? 'device-mobile' :
                newIsTablet ? 'device-tablet' : 'device-desktop'
            );

            // 触发自定义事件
            window.dispatchEvent(new CustomEvent('deviceType
