# napari-mobilesam 安装指南

本文档提供了安装 napari-mobilesam 插件的详细步骤，以及针对 Mac M 系列芯片的优化建议。

## 系统要求

- Python 3.8+
- macOS 11+ (对于 Mac M 系列芯片优化)
- 至少 4GB 内存

## 安装步骤

### 1. 创建虚拟环境（推荐）

首先创建一个新的 Python 虚拟环境：

```bash
# 使用 conda (推荐)
conda create -n napari-env python=3.9
conda activate napari-env

# 或者使用 venv
python -m venv napari-env
source napari-env/bin/activate  # macOS/Linux
```

### 2. 安装 PyTorch

对于 Mac M 系列芯片，建议安装支持 MPS 后端的 PyTorch 版本：

```bash
# 对于 Mac M 系列芯片
pip install torch torchvision torchaudio

# 验证 MPS 是否可用
python -c "import torch; print(torch.backends.mps.is_available())"
# 应该输出 True
```

### 3. 安装 napari

安装 napari 及其依赖：

```bash
pip install "napari[all]"
```

### 4. 安装 napari-mobilesam 插件

有两种安装方法：

#### 方法1: 使用 pip（如果已发布到 PyPI）

```bash
pip install napari-mobilesam
```

#### 方法2: 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/napari-mobilesam.git
cd napari-mobilesam

# 安装插件及其依赖
pip install -e .
```

## 验证安装

安装完成后，运行以下命令启动 napari 并验证插件是否已安装：

```bash
napari
```

在 napari 启动后，打开插件菜单，应该能看到 "MobileSAM Segmentation" 选项。

## 故障排除

### 问题: 找不到插件

确保插件正确安装，可以运行以下命令查看已安装的 napari 插件：

```bash
python -m pip list | grep napari
```

### 问题: PyTorch MPS 后端不可用

对于 Mac M 系列芯片，确保：
1. macOS 版本至少为 12.3+
2. PyTorch 版本至少为 1.13+
3. 安装了正确的 PyTorch 版本（见第2步）

### 问题: 导入 MobileSAM 失败

如果遇到导入 MobileSAM 包的问题，可以手动安装：

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## 附加资源

- [napari 文档](https://napari.org/stable/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [MobileSAM 仓库](https://github.com/ChaoningZhang/MobileSAM) 