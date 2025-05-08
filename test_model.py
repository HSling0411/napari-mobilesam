#!/usr/bin/env python
"""
MobileSAM 模型测试脚本
此脚本用于测试 MobileSAM 模型的基本功能，不使用 GUI 界面
"""

import numpy as np
import os
from skimage import data, io
import matplotlib.pyplot as plt

# 导入 MobileSAM 封装类
from napari_mobilesam.mobilesam_wrapper import MobileSamWrapper

def main():
    """测试 MobileSAM 模型的基本功能"""
    print("开始测试 MobileSAM 模型...")
    
    # 加载模型
    model = MobileSamWrapper()
    
    # 加载测试图像
    print("加载测试图像...")
    image = data.astronaut()
    print(f"图像形状: {image.shape}")
    
    # 设置图像
    print("设置图像到模型...")
    model.set_image(image)
    
    # 测试点预测
    print("\n测试点标注预测...")
    # 在图像中心添加一个前景点
    h, w = image.shape[:2]
    points = np.array([[w//2, h//2]])  # 中心点
    labels = np.array([1])  # 前景点
    
    masks, scores, best_idx = model.predict_from_points(
        points=points,
        labels=labels,
        multimask_output=True
    )
    
    print(f"生成了 {len(masks)} 个掩码候选")
    print(f"最佳掩码索引: {best_idx}, 得分: {scores[best_idx]:.4f}")
    
    # 测试框预测
    print("\n测试框选预测...")
    # 创建一个覆盖图像中心的框
    box = np.array([w//4, h//4, w*3//4, h*3//4])  # [x1, y1, x2, y2]
    
    masks_box, scores_box, best_idx_box = model.predict_from_box(
        box=box,
        multimask_output=True
    )
    
    print(f"生成了 {len(masks_box)} 个掩码候选")
    print(f"最佳掩码索引: {best_idx_box}, 得分: {scores_box[best_idx_box]:.4f}")
    
    # 保存结果
    print("\n保存测试结果...")
    # 创建输出目录
    os.makedirs("test_output", exist_ok=True)
    
    # 可视化点预测结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')
    plt.title("点标注")
    
    plt.subplot(122)
    plt.imshow(image)
    mask = masks[best_idx]
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title(f"点标注掩码 (得分: {scores[best_idx]:.4f})")
    
    plt.tight_layout()
    plt.savefig("test_output/point_prediction.png")
    
    # 可视化框预测结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(image)
    plt.plot([box[0], box[2], box[2], box[0], box[0]], 
             [box[1], box[1], box[3], box[3], box[1]], 'r-')
    plt.title("框选标注")
    
    plt.subplot(122)
    plt.imshow(image)
    mask_box = masks_box[best_idx_box]
    plt.imshow(mask_box, alpha=0.5, cmap='Reds')
    plt.title(f"框选掩码 (得分: {scores_box[best_idx_box]:.4f})")
    
    plt.tight_layout()
    plt.savefig("test_output/box_prediction.png")
    
    print(f"测试完成! 结果保存在 test_output 目录")

if __name__ == "__main__":
    main() 