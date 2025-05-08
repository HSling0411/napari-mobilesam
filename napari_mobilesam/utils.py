import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import datetime
from pathlib import Path
import uuid


def shapes_to_points(shapes: List[Dict], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    将napari的Shapes图层中的点转换为MobileSAM所需的格式
    
    参数:
        shapes: napari中Shapes图层的数据列表
        labels: 对应每个点的标签列表(1表示前景，0表示背景)
        
    返回:
        points: 标注点的坐标数组，形状为(N,2)
        labels: 标注点的标签数组，形状为(N,)
    """
    if not shapes:
        return np.empty((0, 2)), np.empty(0)
    
    # 只保留点类型的形状
    point_shapes = [s for s in shapes if s['shape_type'] == 'point']
    if not point_shapes:
        return np.empty((0, 2)), np.empty(0)
    
    # 提取点坐标 - 注意坐标转换
    points = np.array([s['data'][0] for s in point_shapes])
    
    # 确保点标签与点形状对应
    if labels and len(labels) == len(point_shapes):
        point_labels = np.array(labels)
    else:
        # 默认所有点都是前景点
        point_labels = np.ones(len(point_shapes), dtype=np.int32)
    
    return points, point_labels


def shapes_to_box(shapes: List[Dict]) -> np.ndarray:
    """
    将napari的Shapes图层中的矩形转换为MobileSAM所需的边界框格式
    
    参数:
        shapes: napari中Shapes图层的数据列表
        
    返回:
        box: 边界框坐标数组，形状为(4,)，格式为[x1, y1, x2, y2]
    """
    if not shapes:
        return np.array([])
    
    # 只保留矩形类型的形状
    rect_shapes = [s for s in shapes if s['shape_type'] == 'rectangle']
    if not rect_shapes:
        return np.array([])
    
    try:
        # 使用最新添加的矩形（假设是列表中的最后一个矩形）
        rect = rect_shapes[-1]
        rect_data = rect['data']
        
        # 确保矩形至少有4个顶点
        if len(rect_data) < 4:
            return np.array([])
        
        # 矩形顶点坐标
        x_coords = [p[1] for p in rect_data]
        y_coords = [p[0] for p in rect_data]
        
        # 检查坐标有效性
        if not x_coords or not y_coords:
            return np.array([])
        
        # 计算边界框 [x1, y1, x2, y2]
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        # 检查边界框是否有效（宽高都大于0）
        if x_max <= x_min or y_max <= y_min:
            return np.array([])
        
        box = np.array([x_min, y_min, x_max, y_max])
        
        return box
    except (IndexError, KeyError, ValueError) as e:
        # 捕获可能的错误并返回空数组
        print(f"处理边界框时出错: {str(e)}")
        return np.array([])


def mask_to_binary(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将概率掩码或整数掩码转换为二值掩码
    
    参数:
        mask: 概率掩码或整数掩码
        threshold: 二值化阈值
        
    返回:
        binary_mask: 二值掩码
    """
    # 处理整数掩码
    if mask.dtype in (np.uint8, np.int32, np.int64):
        return (mask > 0).astype(np.uint8)
    # 处理浮点掩码
    return (mask > threshold).astype(np.uint8)


def generate_unique_name(prefix: str = "mask") -> str:
    """
    生成唯一的标注名称
    
    参数:
        prefix: 名称前缀
        
    返回:
        name: 唯一名称
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


def save_masks(
    masks: np.ndarray, 
    scores: np.ndarray, 
    output_dir: str, 
    image_name: Optional[str] = None,
    base_name: Optional[str] = None
) -> List[str]:
    """
    保存分割掩码到文件
    
    参数:
        masks: 掩码数组，形状为(N,H,W)
        scores: 每个掩码的置信度，形状为(N,)
        output_dir: 输出目录
        image_name: 原始图像名称
        base_name: 掩码基础名称，如未指定则自动生成
        
    返回:
        saved_paths: 已保存文件的路径列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成基础名称
    if base_name is None:
        base_name = generate_unique_name()
    
    if image_name is not None:
        base_name = f"{os.path.splitext(image_name)[0]}_{base_name}"
    
    saved_paths = []
    
    # 保存每个掩码
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # 二值化掩码
        binary_mask = mask_to_binary(mask)
        
        # 构建文件路径
        mask_filename = f"{base_name}_{i:03d}.npy"
        mask_path = os.path.join(output_dir, mask_filename)
        
        # 保存为numpy数组
        np.save(mask_path, binary_mask)
        saved_paths.append(mask_path)
        
        # 保存元数据
        metadata = {
            "score": float(score),
            "mask_id": i,
            "base_name": base_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        metadata_filename = f"{base_name}_{i:03d}_meta.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return saved_paths


def batch_process_masks(
    masks_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    output_dir: str,
    image_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    批量处理和保存掩码
    
    参数:
        masks_list: 掩码数组列表
        scores_list: 分数数组列表
        output_dir: 输出目录
        image_names: 图像名称列表
        
    返回:
        saved_paths_dict: 图像名称到已保存文件路径的字典
    """
    saved_paths_dict = {}
    
    for i, (masks, scores) in enumerate(zip(masks_list, scores_list)):
        # 获取图像名称
        image_name = None
        if image_names and i < len(image_names):
            image_name = image_names[i]
        
        # 保存掩码
        base_name = generate_unique_name("batch")
        saved_paths = save_masks(
            masks=masks,
            scores=scores,
            output_dir=output_dir,
            image_name=image_name,
            base_name=base_name
        )
        
        # 添加到字典
        key = image_name if image_name else f"image_{i:03d}"
        saved_paths_dict[key] = saved_paths
    
    return saved_paths_dict 