# napari-mobilesam

[![License](https://img.shields.io/pypi/l/napari-mobilesam.svg?color=green)](https://github.com/yourusername/napari-mobilesam/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-mobilesam.svg?color=green)](https://pypi.org/project/napari-mobilesam)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-mobilesam.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-mobilesam)](https://napari-hub.org/plugins/napari-mobilesam)

## 简介

napari-mobilesam 是一个基于 MobileSAM 的 napari 插件，专为 Mac M 系列芯片优化，支持点标注与框选预测功能。该插件提供简单直观的用户界面，让用户能够使用 Segment Anything Model (SAM) 的轻量化版本进行交互式图像分割。

## 主要特点

- 🚀 基于 MobileSAM 的轻量级分割模型
- 💻 针对 Mac M 系列芯片优化，使用 MPS 后端加速
- ✏️ 支持点标注模式，可添加前景/背景点
- 🔲 支持框选标注模式，通过矩形框引导分割
- 🔄 生成多个掩码候选并进行选择
- 💾 支持掩码结果导出与保存
- 📊 便捷的标签图层添加功能
- 🔧 批量处理与自动命名功能

## 安装

使用 pip 安装:

```
pip install napari-mobilesam
```

或者从源码安装:

```
git clone https://github.com/yourusername/napari-mobilesam.git
cd napari-mobilesam
pip install -e .
```

## 依赖要求

- Python 3.8+
- napari 0.4.16+
- PyTorch 1.13+ (支持 MPS 后端)
- MobileSAM

## 使用方法

### 启动插件

1. 启动 napari
2. 从菜单中选择 `Plugins > MobileSAM Segmentation`

### 基本操作流程

1. **加载图像**：将图像导入 napari
2. **设置图像**：在插件面板中选择需要处理的图像图层，并点击"设置当前图像"
3. **选择标注模式**：
   - 点标注：在图像上添加前景点
   - 框选标注：在图像上绘制矩形框
4. **执行预测**：点击"执行分割预测"按钮
5. **查看结果**：查看分割掩码，可以从多个候选中选择最佳结果
6. **保存结果**：
   - 将掩码添加到 Labels 图层
   - 保存当前掩码或所有掩码到文件

### 批处理功能

1. 在"批处理"选项卡中设置输出目录和命名规则
2. 将图像添加到处理队列
3. 点击"处理队列中所有图像"按钮执行批处理

## 示例

![示例图片](https://example.com/napari-mobilesam-example.png)

## 贡献

欢迎提交 Issues 和 Pull Requests 来帮助改进这个项目。

## 许可证

该项目使用 [MIT 许可证](LICENSE)。

## 致谢

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - 提供轻量级 SAM 模型
- [napari](https://github.com/napari/napari) - 提供图像处理框架 