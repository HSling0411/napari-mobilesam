"""
napari-mobilesam 插件演示脚本

此脚本演示如何在代码中使用 napari-mobilesam 插件进行图像分割。
"""

# 在导入任何PyTorch相关库之前设置环境变量，禁用MPS后端
import os
# 禁用PyTorch MPS后端，以避免段错误
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 强制回退到CPU

import numpy as np
import napari
from skimage import data
from napari_mobilesam import MobileSamWidget

print("加载示例图像...")
# 加载示例图像
image = data.astronaut()

print("创建napari查看器...")
# 创建 napari 查看器
viewer = napari.Viewer()

print("添加图像图层...")
# 添加图像图层
viewer.add_image(image, name='astronaut')

print("创建MobileSAM widget...")
# 创建 MobileSAM widget
mobilesam_widget = MobileSamWidget(viewer)

print("添加widget到viewer...")
# 将 widget 添加到 viewer
viewer.window.add_dock_widget(
    mobilesam_widget, 
    name='MobileSAM Segmentation',
    area='right'
)

# 使用说明
print("\n使用说明:")
print("1. 等待模型加载完成")
print("2. 点击 '设置当前图像' 按钮来选择图像")
print("3. 使用 'Shapes' 图层添加点或矩形标注")
print("4. 点击 '执行分割预测' 按钮进行预测")
print("5. 选择最佳掩码，并可以添加到 Labels 图层或保存")
print("\n注意: 当前使用CPU后端运行，以避免与MPS相关的崩溃问题")

if __name__ == '__main__':
    napari.run() 