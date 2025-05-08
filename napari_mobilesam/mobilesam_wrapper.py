import os
import torch
import numpy as np
from typing import Tuple, List, Optional
import cv2

# 确保可以直接导入MobileSAM
try:
    from mobile_sam import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    # 如果无法直接导入，则克隆仓库
    import subprocess
    import sys
    
    print("未找到mobile_sam，正在尝试下载...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/ChaoningZhang/MobileSAM.git"
    ])
    
    from mobile_sam import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

# 给SamPredictor添加reset_image方法
def add_reset_method_to_predictor():
    if not hasattr(SamPredictor, 'reset_image'):
        def reset_image(self):
            self.is_image_set = False
            self.features = None
            self.orig_h = None
            self.orig_w = None
            self.input_h = None
            self.input_w = None
        SamPredictor.reset_image = reset_image

# 确保有reset_image方法
add_reset_method_to_predictor()

class MobileSamWrapper:
    """MobileSAM模型的封装类，提供点和框预测功能"""
    
    def __init__(self, model_path: str = None, force_device: str = None):
        """
        初始化MobileSAM模型
        
        参数:
            model_path: 模型权重路径，如未指定则使用默认路径
            force_device: 强制使用的设备，可选值: "cpu", "cuda", "mps"，若为None则自动选择
        """
        # 设置默认模型路径
        if model_path is None:
            # 查找当前目录下的模型权重
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_dir = os.path.dirname(current_dir)
            model_path = os.path.join(workspace_dir, "mobile_sam_weights", "mobile_sam.pt")
            
            # 如果找不到则使用预训练模型
            if not os.path.exists(model_path):
                model_path = "mobile_sam"
                print(f"未找到本地模型权重，将下载预训练模型")
        
        # 检测是否为Mac M系列芯片
        is_mac_m_chip = False
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                is_mac_m_chip = True
                if force_device == "mps":
                    print("警告: 在Mac M系列芯片上使用MPS后端可能导致崩溃")
                    print("如果程序崩溃，请重启程序并选择CPU后端")
                elif force_device is None or force_device == "自动":
                    # 如果未指定设备或选择自动，在Mac M系列上默认使用CPU
                    force_device = "cpu"
                    print("检测到Mac M系列芯片，默认使用CPU后端避免MPS崩溃")
        except:
            # 如果无法检测，则继续使用指定的设备
            pass
        
        # 根据force_device参数选择设备
        if force_device is not None:
            if force_device in ["cpu", "cuda", "mps"]:
                self.device = force_device
                print(f"强制使用{self.device}后端进行推理")
            else:
                print(f"不支持的设备类型: {force_device}，将自动选择设备")
                force_device = None
        
        # 如果没有强制指定设备，则自动选择
        if force_device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built() and not is_mac_m_chip:
                self.device = "mps"
                print("使用MPS后端进行推理 - 注意：在M系列芯片上可能会出现段错误，如遇问题请切换到CPU后端")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("使用CUDA后端进行推理")
            else:
                self.device = "cpu"
                print("使用CPU后端进行推理")
        
        # 加载模型
        try:
            self.sam_type = "vit_t"
            if model_path == "mobile_sam":
                # 使用预训练模型
                self.model = sam_model_registry[self.sam_type](checkpoint=None)
            else:
                # 使用自定义模型权重
                self.model = sam_model_registry[self.sam_type](checkpoint=model_path)
            
            # 将模型移到指定设备
            try:
                self.model.to(device=self.device)
            except Exception as e:
                # 如果移到指定设备失败，使用CPU
                print(f"无法将模型移到{self.device}设备: {str(e)}")
                self.device = "cpu"
                self.model.to(device="cpu")
            
            # 初始化预测器
            try:
                self.predictor = SamPredictor(self.model)
            except Exception as e:
                print(f"初始化预测器失败: {str(e)}，尝试使用CPU")
                self.device = "cpu"
                self.model.to(device="cpu")
                self.predictor = SamPredictor(self.model)
            
        except Exception as e:
            # 如果使用指定设备加载失败，尝试使用CPU
            print(f"使用{self.device}设备加载模型失败: {str(e)}")
            print("切换到CPU后端重试...")
            self.device = "cpu"
            
            # 重新加载模型
            if model_path == "mobile_sam":
                self.model = sam_model_registry[self.sam_type](checkpoint=None)
            else:
                self.model = sam_model_registry[self.sam_type](checkpoint=model_path)
            
            self.model.to(device="cpu")
            self.predictor = SamPredictor(self.model)
        
        self.mask_generator = None
        self.image_embeddings = None
        self.current_image = None
    
    def set_image(self, image: np.ndarray) -> None:
        """
        设置当前图像并计算图像嵌入
        
        参数:
            image: RGB格式的图像数组
        """
        try:
            # 输入检查
            if image is None:
                raise ValueError("输入图像不能为None")
            
            # 复制图像数据以避免修改原始数据
            image = image.copy()
            
            # 转换图像格式
            if image.ndim == 2:
                # 灰度图转RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                if image.shape[2] > 3:
                    # 多通道图像只保留RGB通道
                    image = image[:, :, :3]
                # 确保数据类型为uint8
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
            
            # 在CPU上预处理图像，避免MPS错误
            if self.device == "mps":
                try:
                    # 计算图像嵌入前先将图像传输到CPU，然后再传回MPS
                    # 这有助于避免某些MPS错误
                    self.current_image = image
                    with torch.no_grad():
                        self.predictor.reset_image()
                        # 在CPU上设置图像
                        self.predictor.model.to("cpu")
                        self.predictor.set_image(image)
                        # 将模型移回MPS设备
                        self.predictor.model.to(self.device)
                        self.image_embeddings = True  # 标记已计算嵌入
                except Exception as e:
                    print(f"在MPS设备上设置图像时出错: {str(e)}")
                    print("切换到CPU后端...")
                    self.device = "cpu"
                    self.model.to("cpu")
                    self.predictor = SamPredictor(self.model)
                    self.predictor.set_image(image)
                    self.image_embeddings = True
            else:
                # 对于CPU和CUDA后端，直接设置图像
                self.current_image = image
                self.predictor.set_image(image)
                self.image_embeddings = True  # 标记已计算嵌入
            
        except Exception as e:
            self.image_embeddings = False
            raise ValueError(f"设置图像失败: {str(e)}")
    
    def predict_from_points(
        self, 
        points: np.ndarray, 
        labels: np.ndarray, 
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        根据点标注预测分割掩码
        
        参数:
            points: 点坐标数组，形状为(N,2)
            labels: 点标注类型（1表示前景，0表示背景），形状为(N,)
            multimask_output: 是否输出多个掩码候选
            
        返回:
            masks: 预测的掩码数组
            scores: 每个掩码的置信度分数
            best_idx: 最佳掩码的索引
        """
        if self.image_embeddings is None:
            raise ValueError("请先使用set_image()方法设置图像")
        
        # 确保输入数据类型和设备正确
        input_points = points.astype(np.float32)
        input_labels = labels.astype(np.int32)
        
        # 预测掩码
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output,
        )
        
        # 找出最佳掩码
        best_idx = np.argmax(scores)
        
        return masks, scores, best_idx
    
    def predict_from_box(
        self, 
        box: np.ndarray, 
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        根据边界框预测分割掩码
        
        参数:
            box: 边界框坐标数组，形状为(4,)，格式为[x1, y1, x2, y2]
            multimask_output: 是否输出多个掩码候选
            
        返回:
            masks: 预测的掩码数组
            scores: 每个掩码的置信度分数
            best_idx: 最佳掩码的索引
        """
        if self.image_embeddings is None:
            raise ValueError("请先使用set_image()方法设置图像")
        
        # 确保输入数据类型正确
        input_box = box.astype(np.float32)
        
        # 确保框的格式正确
        if input_box.shape != (4,):
            raise ValueError("边界框应为形状(4,)的数组，格式为[x1, y1, x2, y2]")
        
        # 预测掩码
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # 添加批次维度
            multimask_output=multimask_output,
        )
        
        # 找出最佳掩码
        best_idx = np.argmax(scores)
        
        return masks, scores, best_idx
    
    def predict_from_box_and_points(
        self, 
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        结合边界框和点标注预测分割掩码
        
        参数:
            box: 边界框坐标数组，形状为(4,)，格式为[x1, y1, x2, y2]
            points: 点坐标数组，形状为(N,2)
            labels: 点标注类型（1表示前景，0表示背景），形状为(N,)
            multimask_output: 是否输出多个掩码候选
            
        返回:
            masks: 预测的掩码数组
            scores: 每个掩码的置信度分数
            best_idx: 最佳掩码的索引
        """
        if self.image_embeddings is None:
            raise ValueError("请先使用set_image()方法设置图像")
        
        # 确保输入数据类型正确
        input_box = box.astype(np.float32)
        input_points = points.astype(np.float32)
        input_labels = labels.astype(np.int32)
        
        # 确保框的格式正确
        if input_box.shape != (4,):
            raise ValueError("边界框应为形状(4,)的数组，格式为[x1, y1, x2, y2]")
        
        # 预测掩码
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box[None, :],  # 添加批次维度
            multimask_output=multimask_output,
        )
        
        # 找出最佳掩码
        best_idx = np.argmax(scores)
        
        return masks, scores, best_idx
    
    def generate_all_masks(
        self, 
        image: Optional[np.ndarray] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ) -> List[dict]:
        """
        生成图像中的所有分割掩码
        
        参数:
            image: 可选，直接提供图像
            points_per_side: 每边采样点数
            pred_iou_thresh: 预测IoU阈值
            stability_score_thresh: 稳定性分数阈值
            min_mask_region_area: 最小掩码区域面积
            
        返回:
            masks: 掩码列表，每个掩码包含'segmentation'、'area'、'bbox'等信息
        """
        if image is not None:
            self.set_image(image)
        elif self.current_image is None:
            raise ValueError("请先使用set_image()方法设置图像或提供图像参数")
        
        # 延迟初始化MaskGenerator
        if self.mask_generator is None:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        
        # 生成所有掩码
        masks = self.mask_generator.generate(self.current_image)
        return masks 