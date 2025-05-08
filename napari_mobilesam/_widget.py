from typing import List, Optional, Dict, Tuple, Any, Union, Callable
import os
import numpy as np
import threading
from pathlib import Path
import time
import json
import colorsys
import cv2

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFileDialog, QProgressBar, QTabWidget, QLineEdit, QMessageBox,
    QRadioButton, QButtonGroup, QToolButton, QDialog, QTextBrowser,
    QGridLayout, QScrollArea
)
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QColor

import napari
from napari.layers import Image, Shapes, Labels
from napari.utils.notifications import show_info, show_warning, show_error
from napari.types import LayerDataTuple
import torch

from .mobilesam_wrapper import MobileSamWrapper
from .utils import (
    shapes_to_points, shapes_to_box, mask_to_binary, 
    generate_unique_name, save_masks, batch_process_masks
)

class MobileSamWidget(QWidget):
    """MobileSAM napari插件小部件"""
    
    # 信号定义
    progress_signal = Signal(int)
    finished_signal = Signal()
    error_signal = Signal(str)
    point_type_changed_signal = Signal(int)  # 添加点击类型变更信号
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # 初始化模型
        self.model = None
        self.model_thread = None
        self.model_loaded = False
        
        # 初始化变量
        self.current_image = None
        self.current_layer = None
        self.prediction_mode = "点标注"  # 默认为点标注模式
        self.result_masks = []
        self.result_scores = []
        self.selected_mask_idx = 0
        
        # 文件夹导入相关变量
        self.image_folder_path = ""
        self.image_files = []
        self.current_image_index = -1
        self.point_type = 1  # 1表示前景点，0表示背景点
        
        # 键盘状态变量
        self.shift_pressed = False
        self.ctrl_pressed = False
        
        # 点大小设置
        self.point_size = 10
        
        # 批处理相关变量
        self.processing_queue = []
        
        # 添加标签管理相关变量
        self.label_names = {}  # 标签ID到名称的映射
        self.label_colors = {}  # 标签ID到颜色的映射
        self.next_label_id = 1  # 下一个可用的标签ID
        
        # 初始化UI
        self._init_ui()
        
        # 连接信号
        self._connect_signals()
        
        # 设置键盘事件和鼠标事件处理
        self._setup_event_handlers()
        
        # 异步加载模型
        self._load_model_async()
    
    def _init_ui(self):
        """初始化用户界面，使用与napari匹配的深色主题风格"""
        # 设置全局样式 - 匹配napari深色主题
        self.setStyleSheet("""
            QWidget {
                font-family: 'Helvetica Neue', 'Arial', sans-serif;
                font-size: 12px;
                color: #f0f0f0;
                background-color: #2d2d2d;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
                color: #f0f0f0;
                font-weight: medium;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #7d7d7d;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                color: #6d6d6d;
                background-color: #353535;
                border: 1px solid #454545;
            }
            QComboBox {
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 3px 8px;
                min-height: 24px;
                background-color: #3d3d3d;
                color: #f0f0f0;
                selection-background-color: #00a6ff;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #5d5d5d;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QComboBox:on {
                background-color: #404040;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                selection-background-color: #00a6ff;
                selection-color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #5d5d5d;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 8px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: -7px;
                padding: 0 5px;
                background-color: #333333;
                color: #00a6ff;
            }
            QCheckBox {
                spacing: 6px;
                color: #f0f0f0;
                font-weight: medium;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #5d5d5d;
                background-color: #3d3d3d;
            }
            QCheckBox::indicator:checked {
                background-color: #00a6ff;
                border: 1px solid #00a6ff;
            }
            QCheckBox::indicator:unchecked:hover {
                border: 1px solid #00a6ff;
            }
            QLineEdit {
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 3px 8px;
                background-color: #3d3d3d;
                color: #f0f0f0;
                selection-background-color: #00a6ff;
            }
            QLineEdit:focus {
                border: 1px solid #00a6ff;
            }
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #3d3d3d;
                height: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00a6ff;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                top: -1px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #5d5d5d;
                border-bottom: none;
                padding: 5px 10px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #c0c0c0;
            }
            QTabBar::tab:selected {
                background-color: #333333;
                color: #00a6ff;
                font-weight: bold;
                border-bottom: none;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QRadioButton {
                spacing: 6px;
                color: #f0f0f0;
                font-weight: medium;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 1px solid #5d5d5d;
                background-color: #3d3d3d;
            }
            QRadioButton::indicator:checked {
                background-color: #00a6ff;
                border: 1px solid #00a6ff;
                width: 10px;
                height: 10px;
                margin: 3px;
            }
            QRadioButton::indicator:unchecked:hover {
                border: 1px solid #00a6ff;
            }
            QLabel {
                color: #f0f0f0;
            }
            QLabel[labelType="heading"] {
                font-weight: bold;
                font-size: 12px;
                color: #00a6ff;
            }
            QLabel[labelType="info"] {
                color: #a0a0a0;
                font-size: 11px;
            }
            QLabel[labelType="highlight"] {
                color: #00a6ff;
                font-weight: bold;
            }
            QLabel[labelType="value"] {
                color: #ffffff;
                font-weight: bold;
                background-color: #3d3d3d;
                border-radius: 3px;
                padding: 2px 6px;
            }
            QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 3px;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #00a6ff;
            }
            QSpinBox, QDoubleSpinBox {
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 3px;
                background-color: #3d3d3d;
                color: #f0f0f0;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #3d3d3d;
                width: 16px;
                border-left: 1px solid #5d5d5d;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4d4d4d;
            }
        """)
        
        # 主布局设置
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)  # 进一步减小外边距
        layout.setSpacing(8)  # 减小组件间距
        
        # 创建选项卡Widget
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setDocumentMode(True)
        
        # 标注Tab
        annotation_tab = QWidget()
        annotation_layout = QVBoxLayout()
        annotation_layout.setContentsMargins(8, 10, 8, 8)  # 减小内边距
        annotation_layout.setSpacing(8)  # 减小组件间距
        
        # 模型状态组
        model_group = QGroupBox("模型")
        model_group.setCheckable(True)
        model_group.setChecked(True)
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(8, 12, 8, 8)  # 减小内边距
        model_layout.setSpacing(6)  # 调整组件间距
        
        # 模型状态布局
        model_status_layout = QHBoxLayout()
        self.model_status_label = QLabel("状态: 正在加载模型...")
        model_status_layout.addWidget(self.model_status_label)
        model_layout.addLayout(model_status_layout)
        
        # 添加设备选择下拉框
        device_layout = QHBoxLayout()
        device_label = QLabel("设备:")
        device_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "MPS", "CUDA", "自动"])
        self.device_combo.setToolTip("选择推理设备，Mac M系列芯片推荐使用CPU避免崩溃")
        # 设置默认选择为CPU，避免在Mac上使用MPS导致崩溃
        self.device_combo.setCurrentText("CPU")
        device_layout.addWidget(self.device_combo)
        model_layout.addLayout(device_layout)
        
        # 进度条和按钮
        progress_layout = QHBoxLayout()
        progress_layout.setSpacing(8)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # 隐藏文本
        
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.setEnabled(False)  # 初始禁用
        
        self.load_custom_model_btn = QPushButton("自定义模型")
        self.load_custom_model_btn.setFixedWidth(110)
        self.load_model_btn.setFixedWidth(110)
        
        progress_layout.addWidget(self.progress_bar, 1)
        progress_layout.addWidget(self.load_model_btn)
        progress_layout.addWidget(self.load_custom_model_btn)
        model_layout.addLayout(progress_layout)
        
        model_group.setLayout(model_layout)
        annotation_layout.addWidget(model_group)
        
        # 图像选择组
        image_group = QGroupBox("图像")
        image_group.setCheckable(True)
        image_group.setChecked(True)
        image_layout = QVBoxLayout()
        image_layout.setContentsMargins(8, 12, 8, 8)  # 减小内边距
        image_layout.setSpacing(6)  # 调整组件间距
        
        # 文件夹导入部分
        folder_import_layout = QHBoxLayout()
        self.import_folder_btn = QPushButton("导入文件夹")
        
        self.folder_path_label = QLabel("未选择文件夹")
        self.folder_path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.folder_path_label.setWordWrap(False)
        self.folder_path_label.setMaximumWidth(220)
        
        folder_import_layout.addWidget(self.import_folder_btn)
        folder_import_layout.addWidget(self.folder_path_label, 1)
        image_layout.addLayout(folder_import_layout)
        
        # 图像导航部分
        image_nav_layout = QHBoxLayout()
        image_nav_layout.setSpacing(8)
        
        # 导航按钮组
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        
        self.prev_image_btn = QPushButton("◀")
        self.prev_image_btn.setFixedSize(32, 32)
        
        self.image_counter_label = QLabel("0/0")
        self.image_counter_label.setFixedWidth(50)
        self.image_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.next_image_btn = QPushButton("▶")
        self.next_image_btn.setFixedSize(32, 32)
        
        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addWidget(self.image_counter_label)
        nav_layout.addWidget(self.next_image_btn)
        
        # 图像选择部分
        self.image_combo = QComboBox()
        self.refresh_image_btn = QPushButton("刷新")
        self.refresh_image_btn.setFixedWidth(70)
        
        self.set_image_btn = QPushButton("设置")
        self.set_image_btn.setFixedWidth(70)
        
        image_nav_layout.addLayout(nav_layout)
        image_nav_layout.addWidget(self.image_combo, 1)
        image_nav_layout.addWidget(self.refresh_image_btn)
        image_nav_layout.addWidget(self.set_image_btn)
        
        image_layout.addLayout(image_nav_layout)
        
        image_group.setLayout(image_layout)
        annotation_layout.addWidget(image_group)
        
        # 预测模式组
        predict_group = QGroupBox("标注设置")
        predict_group.setCheckable(True)
        predict_group.setChecked(True)
        predict_layout = QVBoxLayout()
        predict_layout.setContentsMargins(8, 12, 8, 8)  # 减小内边距
        predict_layout.setSpacing(6)  # 调整组件间距
        
        # 模式和点类型在一行
        mode_point_layout = QGridLayout()
        mode_point_layout.setHorizontalSpacing(8)
        mode_point_layout.setVerticalSpacing(4)
        mode_point_layout.setColumnStretch(1, 1)  # 设置第2列可伸缩
        mode_point_layout.setColumnStretch(3, 1)  # 设置第4列可伸缩
        
        # 模式选择
        mode_label = QLabel("模式:")
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["点标注", "框选标注"])
        self.mode_combo.setFixedHeight(32)
        
        # 点类型状态
        point_label = QLabel("当前:")
        
        self.point_type_status = QLabel("前景点")
        self.point_type_status.setStyleSheet("""
            color: #4caf50; 
            font-weight: 600;
            background-color: #3d3d3d;
            border-radius: 4px;
            padding: 4px 8px;
        """)  # 绿色表示前景点
        
        mode_point_layout.addWidget(mode_label, 0, 0)
        mode_point_layout.addWidget(self.mode_combo, 0, 1)
        mode_point_layout.addWidget(point_label, 0, 2)
        mode_point_layout.addWidget(self.point_type_status, 0, 3)
        
        predict_layout.addLayout(mode_point_layout)
        
        # 点大小和自动预测在一行
        size_auto_layout = QGridLayout()
        size_auto_layout.setHorizontalSpacing(8)
        size_auto_layout.setVerticalSpacing(4)
        size_auto_layout.setColumnStretch(1, 1)  # 设置第2列可伸缩
        
        # 点大小
        size_label = QLabel("点大小:")
        
        self.point_size_value = QLabel(f"{self.point_size}")
        self.point_size_value.setStyleSheet("""
            font-weight: medium;
            font-size: 12px;
            color: #ffffff;
            background-color: #00a6ff;
            border-radius: 3px;
            padding: 2px 5px;
        """)  # 使用蓝色背景
        
        ctrl_scroll_label = QLabel("(Ctrl+滚轮)")
        ctrl_scroll_label.setStyleSheet("color: #a0a0a0; font-size: 12px;")  # 浅灰色
        
        size_layout = QHBoxLayout()
        size_layout.setSpacing(6)
        size_layout.addWidget(self.point_size_value)
        size_layout.addWidget(ctrl_scroll_label)
        
        # 自动预测
        self.auto_predict_check = QCheckBox("自动预测")
        self.auto_predict_check.setChecked(True)  # 默认启用
        
        size_auto_layout.addWidget(size_label, 0, 0)
        size_auto_layout.addLayout(size_layout, 0, 1)
        size_auto_layout.addWidget(self.auto_predict_check, 0, 2)
        
        predict_layout.addLayout(size_auto_layout)
        
        # 多掩码和预测按钮
        mask_predict_layout = QHBoxLayout()
        mask_predict_layout.setSpacing(8)
        
        # 多掩码设置
        multimask_layout = QHBoxLayout()
        multimask_layout.setSpacing(4)
        
        self.multimask_check = QCheckBox("多掩码候选")
        self.multimask_check.setChecked(True)
        
        self.multimask_help_btn = QToolButton()
        self.multimask_help_btn.setText("?")
        self.multimask_help_btn.setToolTip("查看有关多掩码生成机制和得分依据的说明")
        self.multimask_help_btn.setFixedSize(24, 24)
        
        multimask_layout.addWidget(self.multimask_check)
        multimask_layout.addWidget(self.multimask_help_btn)
        
        # 预测按钮
        self.predict_btn = QPushButton("执行预测")
        self.predict_btn.setEnabled(False)  # 初始禁用
        
        mask_predict_layout.addLayout(multimask_layout)
        mask_predict_layout.addStretch(1)
        mask_predict_layout.addWidget(self.predict_btn)
        
        predict_layout.addLayout(mask_predict_layout)
        
        # 隐藏不使用的控件
        self.positive_point_radio = QRadioButton("正点击(前景)")
        self.negative_point_radio = QRadioButton("负点击(背景)")
        self.positive_point_radio.setChecked(True)
        self.positive_point_radio.hide()
        self.negative_point_radio.hide()
        self.point_type_radio_group = QButtonGroup(self)
        self.point_type_radio_group.addButton(self.positive_point_radio, 1)
        self.point_type_radio_group.addButton(self.negative_point_radio, 0)
        
        predict_group.setLayout(predict_layout)
        annotation_layout.addWidget(predict_group)
        
        # 结果处理组
        result_group = QGroupBox("掩码操作")
        result_group.setCheckable(True)
        result_group.setChecked(True)
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(8, 12, 8, 8)  # 减小内边距
        result_layout.setSpacing(6)  # 调整组件间距
        
        # 掩码选择
        mask_control_layout = QHBoxLayout()
        mask_control_layout.setSpacing(6)
        
        mask_label = QLabel("掩码:")
        
        self.mask_combo = QComboBox()
        self.mask_combo.setEnabled(False)
        
        mask_preview_btn = QPushButton("预览")
        mask_preview_btn.setFixedWidth(70)
        mask_preview_btn.setToolTip("在标签图层上预览选中的掩码")
        mask_preview_btn.clicked.connect(self._preview_selected_mask)
        
        mask_control_layout.addWidget(mask_label)
        mask_control_layout.addWidget(self.mask_combo, 1)
        mask_control_layout.addWidget(mask_preview_btn)
        
        result_layout.addLayout(mask_control_layout)
        
        # 掩码调整
        mask_adjust_layout = QHBoxLayout()
        mask_adjust_layout.setSpacing(6)
        
        self.erode_mask_btn = QPushButton("收缩")
        self.erode_mask_btn.setToolTip("收缩当前掩码边界")
        self.erode_mask_btn.setEnabled(False)
        self.erode_mask_btn.clicked.connect(lambda: self._adjust_mask_boundary(-1))
        
        self.dilate_mask_btn = QPushButton("扩张")
        self.dilate_mask_btn.setToolTip("扩张当前掩码边界") 
        self.dilate_mask_btn.setEnabled(False)
        self.dilate_mask_btn.clicked.connect(lambda: self._adjust_mask_boundary(1))
        
        mask_adjust_layout.addWidget(self.erode_mask_btn)
        mask_adjust_layout.addWidget(self.dilate_mask_btn)
        
        result_layout.addLayout(mask_adjust_layout)
        
        # 标签名称
        label_name_layout = QHBoxLayout()
        label_name_layout.setSpacing(6)
        
        label_name_label = QLabel("标签:")
        
        self.label_name_combo = QComboBox()
        self.label_name_combo.setEditable(True)
        self.label_name_combo.setInsertPolicy(QComboBox.InsertPolicy.InsertAtBottom)
        
        label_name_layout.addWidget(label_name_label)
        label_name_layout.addWidget(self.label_name_combo, 1)
        
        result_layout.addLayout(label_name_layout)
        
        # 添加到标签和保存按钮
        add_save_layout = QGridLayout()
        add_save_layout.setHorizontalSpacing(6)
        add_save_layout.setVerticalSpacing(6)
        add_save_layout.setColumnStretch(0, 1)
        add_save_layout.setColumnStretch(1, 1)
        
        self.add_to_labels_btn = QPushButton("添加到标签")
        self.add_to_labels_btn.setEnabled(False)
        
        self.save_current_btn = QPushButton("保存掩码")
        self.save_current_btn.setEnabled(False)
        
        self.save_all_btn = QPushButton("保存所有掩码")
        self.save_all_btn.setEnabled(False)
        
        add_save_layout.addWidget(self.add_to_labels_btn, 0, 0)
        add_save_layout.addWidget(self.save_current_btn, 0, 1)
        add_save_layout.addWidget(self.save_all_btn, 1, 0, 1, 2)
        
        result_layout.addLayout(add_save_layout)
        
        # 标签管理按钮
        label_manage_layout = QHBoxLayout()
        label_manage_layout.setSpacing(6)
        
        self.export_labels_btn = QPushButton("导出标签")
        self.export_labels_btn.setEnabled(False)
        
        self.clear_labels_btn = QPushButton("清除标签")
        self.clear_labels_btn.setEnabled(False)
        
        label_manage_layout.addWidget(self.export_labels_btn)
        label_manage_layout.addWidget(self.clear_labels_btn)
        
        result_layout.addLayout(label_manage_layout)
        
        result_group.setLayout(result_layout)
        annotation_layout.addWidget(result_group)
        
        # 添加拉伸因子，确保紧凑布局
        annotation_layout.addStretch(1)
        
        annotation_tab.setLayout(annotation_layout)
        tabs.addTab(annotation_tab, "标注")
        
        # 批处理Tab
        batch_tab = QWidget()
        batch_layout = QVBoxLayout()
        batch_layout.setContentsMargins(10, 10, 10, 10)  # 减小边距
        batch_layout.setSpacing(8)  # 减小间距
        
        # 添加处理进度提示
        def _process_queue(self):
            # 开始处理时
            self.batch_progress_bar.setFormat("处理中... %p%")
            # 处理完成时
            self.batch_progress_bar.setFormat("完成 %p%")
            self.batch_progress_bar.setStyleSheet("""
                QProgressBar::chunk {
                    background-color: #00a6ff;  /* napari蓝色 */
                }
            """)
        
        # 批处理设置 - 更紧凑
        batch_setting_group = QGroupBox("批处理设置")
        batch_setting_layout = QVBoxLayout()
        batch_setting_layout.setContentsMargins(5, 5, 5, 5)
        batch_setting_layout.setSpacing(3)
        
        # 输入目录
        input_dir_layout = QHBoxLayout()
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setReadOnly(True)
        self.input_dir_btn = QPushButton("选择输入...")
        self.input_dir_btn.setFixedWidth(80)
        
        input_dir_layout.addWidget(QLabel("输入:"))
        input_dir_layout.addWidget(self.input_dir_edit, 1)
        input_dir_layout.addWidget(self.input_dir_btn)
        
        batch_setting_layout.addLayout(input_dir_layout)
        
        # 输出目录
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("选择输出...")
        self.output_dir_btn.setFixedWidth(80)
        
        output_dir_layout.addWidget(QLabel("输出:"))
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        output_dir_layout.addWidget(self.output_dir_btn)
        
        batch_setting_layout.addLayout(output_dir_layout)
        
        # 命名前缀和自动命名
        prefix_auto_layout = QHBoxLayout()
        self.prefix_edit = QLineEdit("mask")
        
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("前缀:"))
        prefix_layout.addWidget(self.prefix_edit)
        
        self.auto_naming_check = QCheckBox("使用图像名称")
        self.auto_naming_check.setChecked(True)
        
        prefix_auto_layout.addLayout(prefix_layout)
        prefix_auto_layout.addWidget(self.auto_naming_check)
        
        batch_setting_layout.addLayout(prefix_auto_layout)
        
        batch_setting_group.setLayout(batch_setting_layout)
        batch_layout.addWidget(batch_setting_group)
        
        # 批处理控制 - 更紧凑
        batch_control_group = QGroupBox("批处理控制")
        batch_control_layout = QVBoxLayout()
        batch_control_layout.setContentsMargins(5, 5, 5, 5)
        batch_control_layout.setSpacing(3)
        
        # 按钮在同一行
        batch_btn_layout = QHBoxLayout()
        self.add_to_queue_btn = QPushButton("添加到队列")
        self.process_queue_btn = QPushButton("处理队列")
        self.process_queue_btn.setEnabled(False)
        
        batch_btn_layout.addWidget(self.add_to_queue_btn)
        batch_btn_layout.addWidget(self.process_queue_btn)
        
        batch_control_layout.addLayout(batch_btn_layout)
        
        # 队列状态
        self.queue_status_label = QLabel("队列状态: 0 个图像")
        batch_control_layout.addWidget(self.queue_status_label)
        
        # 批处理进度条
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        batch_control_layout.addWidget(self.batch_progress_bar)
        
        batch_control_group.setLayout(batch_control_layout)
        batch_layout.addWidget(batch_control_group)
        
        # 添加拉伸因子，确保紧凑布局
        batch_layout.addStretch(1)
        
        batch_tab.setLayout(batch_layout)
        tabs.addTab(batch_tab, "批处理")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
        
        # 将关键功能按钮添加特殊样式
        
        # 执行预测按钮 - 使用强调色
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #00a6ff;
                color: white;
                font-weight: medium;
                border: none;
                border-radius: 4px;
                padding: 4px 10px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:pressed {
                background-color: #0077b3;
            }
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #a0a0a0;
            }
        """)
        
        # 添加到标签按钮 - 使用绿色
        self.add_to_labels_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: medium;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #43a047;
            }
            QPushButton:pressed {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #a0a0a0;
            }
        """)
        
        # 清除标签按钮 - 使用红色
        self.clear_labels_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: medium;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
            QPushButton:pressed {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #a0a0a0;
            }
        """)
        
        # 设置图像按钮 - 使用强调色
        self.set_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #00a6ff;
                color: white;
                font-weight: medium;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #0088cc;
            }
            QPushButton:pressed {
                background-color: #0077b3;
            }
            QPushButton:disabled {
                background-color: #5a5a5a;
                color: #a0a0a0;
            }
        """)
        
        # 状态栏显示快捷键信息
        self.viewer.status = "快捷键: [Shift+点击]背景点 | [点击]前景点 | [空格]预测 | [F/B]切换前景/背景 | [Ctrl+滚轮]调整点大小"
    
    def _connect_signals(self):
        """连接信号与槽"""
        # 模型加载相关
        self.load_model_btn.clicked.connect(self._load_model_async)
        self.load_custom_model_btn.clicked.connect(self._load_custom_model)
        
        # 图像选择相关
        self.refresh_image_btn.clicked.connect(self._refresh_image_layers)
        self.set_image_btn.clicked.connect(self._set_current_image)
        
        # 文件夹导入相关
        self.import_folder_btn.clicked.connect(self._import_image_folder)
        self.prev_image_btn.clicked.connect(self._load_prev_image)
        self.next_image_btn.clicked.connect(self._load_next_image)
        
        # 预测相关
        self.mode_combo.currentTextChanged.connect(self._update_prediction_mode)
        self.predict_btn.clicked.connect(self._run_prediction)
        self.positive_point_radio.toggled.connect(self._update_point_type)
        
        # 添加快捷键信号的连接
        self.point_type_changed_signal.connect(self._update_point_type_display)
        
        # 结果处理相关
        self.mask_combo.currentIndexChanged.connect(self._update_selected_mask)
        self.add_to_labels_btn.clicked.connect(self._add_mask_to_labels)
        self.save_current_btn.clicked.connect(self._save_current_mask)
        self.save_all_btn.clicked.connect(self._save_all_masks)
        
        # 批处理相关
        self.input_dir_btn.clicked.connect(self._select_input_directory)
        self.output_dir_btn.clicked.connect(self._select_output_directory)
        self.add_to_queue_btn.clicked.connect(self._add_to_queue)
        self.process_queue_btn.clicked.connect(self._process_queue)
        
        # 异步信号
        self.progress_signal.connect(self._update_progress)
        self.finished_signal.connect(self._model_loading_finished)
        self.error_signal.connect(self._handle_error)
        
        # Viewer事件
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        
        # 监听Shapes图层的变化，用于自动预测
        self._connect_shapes_layer_events()
        
        # 添加帮助按钮信号连接
        self.multimask_help_btn.clicked.connect(self._show_multimask_help)
        
        # 标签管理相关
        self.export_labels_btn.clicked.connect(self._export_label_info)
        self.clear_labels_btn.clicked.connect(self._clear_all_labels)
    
    def _setup_event_handlers(self):
        """设置键盘和鼠标事件处理"""
        try:
            # 添加快捷键，使用try-except避免版本兼容性问题
            try:
                # 添加空格键作为预测快捷键
                self.viewer.bind_key(" ", self._run_prediction)
                
                # 添加F和B键作为前景点和背景点快捷键
                self.viewer.bind_key("f", self._set_positive_point)
                self.viewer.bind_key("b", self._set_negative_point)
            except Exception as e:
                show_warning(f"设置快捷键失败: {str(e)}，可能是napari版本兼容性问题")
            
            # 尝试监听键盘事件
            if hasattr(self.viewer, 'events'):
                # 通过napari的事件系统连接键盘事件
                if hasattr(self.viewer.events, 'key_press'):
                    self.viewer.events.key_press.connect(self._on_key_press)
                if hasattr(self.viewer.events, 'key_release'):
                    self.viewer.events.key_release.connect(self._on_key_release)
            
            # 设置鼠标事件
            self._connect_viewer_mouse_events()
        
        except Exception as e:
            show_warning(f"设置事件处理器失败: {str(e)}")
    
    def _connect_viewer_mouse_events(self):
        """连接查看器的鼠标事件"""
        try:
            # 尝试使用事件系统监听鼠标滚轮事件
            if hasattr(self.viewer, 'events') and hasattr(self.viewer.events, 'mouse_wheel'):
                self.viewer.events.mouse_wheel.connect(self._on_mouse_wheel)
        except Exception as e:
            show_warning(f"设置鼠标滚轮事件失败: {str(e)}")
    
    def _on_mouse_wheel(self, event):
        """处理鼠标滚轮事件"""
        try:
            # 如果按住Ctrl键，调整点大小
            if self.ctrl_pressed:
                # 尝试获取滚轮方向，兼容不同版本的事件格式
                delta = 0
                if hasattr(event, 'delta') and isinstance(event.delta, (list, tuple)) and len(event.delta) > 1:
                    delta = event.delta[1]
                elif hasattr(event, 'delta') and not isinstance(event.delta, (list, tuple)):
                    delta = event.delta
                
                # 根据滚轮方向调整点大小
                if delta > 0:  # 向上滚动，增大点
                    self.point_size = min(50, self.point_size + 2)
                else:  # 向下滚动，减小点
                    self.point_size = max(1, self.point_size - 2)
                
                # 更新点大小显示
                self.point_size_value.setText(f"{self.point_size}")
                self.point_size_value.setStyleSheet(f"""
                    font-weight: medium;
                    font-size: 12px;
                    color: #ffffff;
                    background-color: #00a6ff;
                    border-radius: 3px;
                    padding: 2px 5px;
                """)
                
                # 找到shapes图层并应用新的点大小
                for layer in self.viewer.layers:
                    if isinstance(layer, Shapes) and layer.name == "标注":
                        layer.size = self.point_size
                        layer.refresh()
                        break
                
                # 在状态栏显示点大小
                self.viewer.status = f"点大小已调整为: {self.point_size}"
                
                # 标记事件已处理
                if hasattr(event, 'handled'):
                    event.handled = True
                return True  # 返回True表示已处理事件
        except Exception as e:
            show_warning(f"处理鼠标滚轮事件失败: {str(e)}")
        return False
    
    def _on_key_press(self, event):
        """处理键盘按下事件"""
        try:
            # 尝试获取按键，兼容不同版本的事件格式
            key = None
            if hasattr(event, 'key'):
                key = event.key
            
            # 检测Shift键
            if key == "Shift":
                self.shift_pressed = True
                self.point_type = 0  # 切换为背景点
                self.point_type_changed_signal.emit(0)
            
            # 检测Ctrl键
            elif key in ["Control", "Meta"]:  # 兼容Mac的Command键
                self.ctrl_pressed = True
        except Exception as e:
            pass  # 静默处理按键错误
    
    def _on_key_release(self, event):
        """处理键盘释放事件"""
        try:
            # 尝试获取按键，兼容不同版本的事件格式
            key = None
            if hasattr(event, 'key'):
                key = event.key
            
            # 检测Shift键释放
            if key == "Shift":
                self.shift_pressed = False
                self.point_type = 1  # 切换回前景点
                self.point_type_changed_signal.emit(1)
            
            # 检测Ctrl键释放
            elif key in ["Control", "Meta"]:  # 兼容Mac的Command键
                self.ctrl_pressed = False
        except Exception as e:
            pass  # 静默处理按键错误
    
    def _update_point_type_display(self, point_type):
        """更新点类型显示"""
        if point_type == 1:
            self.point_type_status.setText("前景点")
            self.point_type_status.setStyleSheet("""
                color: #4caf50;  /* 绿色 */
                font-weight: medium;
                font-size: 12px;
                background-color: #3d3d3d;
                border-radius: 3px;
                padding: 3px 6px;
            """)
            self.positive_point_radio.setChecked(True)
            
            # 在状态栏显示当前模式
            self.viewer.status = "前景点标注模式 | 按Shift切换到背景点 | F/B键快速切换"
            
            # 显示提示
            if hasattr(self, '_last_point_type') and self._last_point_type != point_type:
                show_info("已切换到前景点模式")
        else:
            self.point_type_status.setText("背景点")
            self.point_type_status.setStyleSheet("""
                color: #f44336;  /* 红色 */
                font-weight: medium;
                font-size: 12px;
                background-color: #3d3d3d;
                border-radius: 3px;
                padding: 3px 6px;
            """)
            self.negative_point_radio.setChecked(True)
            
            # 在状态栏显示当前模式
            self.viewer.status = "背景点标注模式 | 释放Shift切换到前景点 | F/B键快速切换"
            
            # 显示提示
            if hasattr(self, '_last_point_type') and self._last_point_type != point_type:
                show_info("已切换到背景点模式")
        
        # 记录上一次的点类型，用于状态变化检测
        self._last_point_type = point_type
        
        # 找到shapes图层并更新默认颜色
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes) and layer.name == "标注":
                # 更新下一个点的颜色
                if point_type == 1:
                    layer.current_edge_color = [0.3, 0.8, 0.3, 1]  # 绿色边缘
                    if hasattr(layer, 'current_face_color'):
                        layer.current_face_color = [0.3, 0.8, 0.3, 0.5]  # 绿色填充
                else:
                    layer.current_edge_color = [0.95, 0.3, 0.2, 1]  # 红色边缘
                    if hasattr(layer, 'current_face_color'):
                        layer.current_face_color = [0.95, 0.3, 0.2, 0.5]  # 红色填充
                break
    
    def _load_model_async(self):
        """异步加载模型"""
        if self.model_thread is not None and self.model_thread.is_alive():
            # 已经在加载中
            return
        
        # 禁用按钮并更新状态
        self.load_model_btn.setEnabled(False)
        self.load_custom_model_btn.setEnabled(False)
        self.model_status_label.setText("状态: 正在加载模型...")
        self.progress_bar.setValue(0)
        
        # 启动线程
        self.model_thread = threading.Thread(target=self._load_model_thread)
        self.model_thread.daemon = True
        self.model_thread.start()
    
    def _load_model_thread(self, custom_path=None):
        """模型加载线程函数"""
        try:
            # 模拟加载进度
            for i in range(0, 90, 10):
                time.sleep(0.2)  # 模拟加载时间
                self.progress_signal.emit(i)
            
            # 获取选择的设备
            device = None
            device_text = self.device_combo.currentText()
            if device_text == "CPU":
                device = "cpu"
            elif device_text == "MPS":
                device = "mps"
                # 添加警告信息
                print("警告：选择了MPS后端，在Mac M系列芯片上可能导致崩溃")
            elif device_text == "CUDA":
                device = "cuda"
            elif device_text == "自动":
                device = None
            
            # 尝试加载模型
            try:
                self.model = MobileSamWrapper(model_path=custom_path, force_device=device)
            except Exception as e:
                # 如果使用指定设备失败，尝试使用CPU
                print(f"使用{device}后端加载模型失败: {str(e)}")
                print("尝试使用CPU后端加载模型...")
                self.model = MobileSamWrapper(model_path=custom_path, force_device="cpu")
            
            # 完成加载
            self.progress_signal.emit(100)
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def _load_custom_model(self):
        """加载自定义模型"""
        # 打开文件对话框
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择MobileSAM模型文件", "", "PyTorch模型 (*.pt *.pth)"
        )
        
        if file_path:
            # 异步加载自定义模型
            self.load_model_btn.setEnabled(False)
            self.load_custom_model_btn.setEnabled(False)
            self.model_status_label.setText("状态: 正在加载自定义模型...")
            self.progress_bar.setValue(0)
            
            # 启动线程
            self.model_thread = threading.Thread(
                target=self._load_model_thread, 
                args=(file_path,)
            )
            self.model_thread.daemon = True
            self.model_thread.start()
    
    def _update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def _model_loading_finished(self):
        """模型加载完成"""
        self.model_loaded = True
        self.model_status_label.setText(f"状态: 模型已加载 (设备: {self.model.device})")
        self.model_status_label.setStyleSheet("""
            color: #4caf50;  /* 绿色 */
            font-weight: medium;
            font-size: 12px;
        """)
        
        # 启用按钮
        self.load_model_btn.setEnabled(True)
        self.load_custom_model_btn.setEnabled(True)
        
        # 刷新图像列表
        self._refresh_image_layers()
        
        # 根据当前设备提供额外提示
        device_msg = ""
        if self.model.device == "mps":
            device_msg = " | 注意：MPS后端在Mac M系列芯片上可能不稳定"
        elif self.model.device == "cpu":
            device_msg = " | 使用CPU后端，性能较慢但稳定"
        
        # 在状态栏显示提示
        self.viewer.status = f"模型加载完成，使用{self.model.device.upper()}设备{device_msg} | 请选择图像并添加点标注或框选"
        
        # 如果检测到Mac M系列芯片且用户选择了MPS，显示额外警告
        try:
            import platform
            if platform.system() == "Darwin" and platform.machine() == "arm64" and self.model.device == "mps":
                show_warning("您正在Mac M系列芯片上使用MPS后端，这可能导致程序崩溃。\n"
                            "如果遇到问题，请在设备选项中选择CPU并重新加载模型。")
        except:
            pass
    
    def _handle_error(self, error_msg):
        """处理错误信息"""
        self.model_status_label.setText(f"状态: 加载失败")
        self.model_status_label.setStyleSheet("""
            color: #f44336;  /* 红色 */
            font-weight: medium;
            font-size: 12px;
        """)
        self.progress_bar.setValue(0)
        
        # 启用按钮
        self.load_model_btn.setEnabled(True)
        self.load_custom_model_btn.setEnabled(True)
        
        # 显示错误
        show_error(f"模型加载错误: {error_msg}")
    
    def _refresh_image_layers(self):
        """刷新图像图层列表"""
        # 清空当前列表
        self.image_combo.clear()
        
        # 获取所有Image类型的图层
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        
        if image_layers:
            # 添加到下拉列表
            self.image_combo.addItems(image_layers)
        else:
            # 未找到图像图层
            show_warning("未找到图像图层，请先添加图像")
    
    def _on_layer_change(self, event):
        """当图层变化时更新图层列表和标签名称"""
        self._refresh_image_layers()
        
        # 检查并更新标签信息
        if "分割标签" in self.viewer.layers:
            labels_layer = self.viewer.layers["分割标签"]
            if hasattr(labels_layer, 'metadata'):
                if 'label_names' in labels_layer.metadata:
                    self.label_names = labels_layer.metadata['label_names']
                    self._update_label_name_combo()
                    
                    # 更新下一个ID
                    if self.label_names:
                        self.next_label_id = max(self.label_names.keys()) + 1
                
                if 'label_colors' in labels_layer.metadata:
                    self.label_colors = labels_layer.metadata['label_colors']
                
                # 启用标签管理按钮
                self.export_labels_btn.setEnabled(True)
                self.clear_labels_btn.setEnabled(True)
        else:
            # 禁用标签管理按钮
            self.export_labels_btn.setEnabled(False)
            self.clear_labels_btn.setEnabled(False)
    
    def _set_current_image(self):
        """设置当前处理的图像"""
        if not self.model_loaded:
            show_warning("模型尚未加载，请先加载模型")
            return
        
        if self.image_combo.count() == 0:
            show_warning("未找到图像图层，请先添加图像")
            return
        
        # 获取选中的图像图层
        selected_layer_name = self.image_combo.currentText()
        selected_layer = self.viewer.layers[selected_layer_name]
        
        if not isinstance(selected_layer, Image):
            show_error(f"图层 '{selected_layer_name}' 不是图像类型")
            return
        
        # 获取图像数据
        image_data = selected_layer.data
        
        # 设置当前图像
        try:
            # 显示加载提示
            self.viewer.status = "正在处理图像，请稍候..."
            
            # 设置图像到模型
            self.model.set_image(image_data)
            
            # 更新状态
            self.current_image = image_data
            self.current_layer = selected_layer
            
            # 启用预测按钮
            self.predict_btn.setEnabled(True)
            
            show_info(f"已设置图像 '{selected_layer_name}'")
            
            # 在状态栏显示提示
            self.viewer.status = "图像已设置 | 添加点标注或框选，按空格键执行预测"
            
        except Exception as e:
            # 显示详细错误信息
            error_msg = f"设置图像失败: {str(e)}"
            show_error(error_msg)
            
            # 检查是否为MPS错误
            error_str = str(e).lower()
            if "mps" in error_str or "metal" in error_str or "gpu" in error_str:
                # 提示用户切换到CPU
                show_warning("检测到可能是MPS后端问题，请尝试在设备选项中选择'CPU'并重新加载模型")
                # 自动切换到CPU
                self.device_combo.setCurrentText("CPU")
            
            # 更新状态
            self.viewer.status = "图像设置失败，请重试"
    
    def _update_prediction_mode(self, mode):
        """更新预测模式"""
        self.prediction_mode = mode
        
        # 根据模式提示用户
        if mode == "点标注":
            show_info("点标注模式: 请在Shapes图层中添加点标注")
            # 在状态栏显示操作提示
            self.viewer.status = "点标注模式 | 绿色: 前景点 | 红色: 背景点 | 空格键预测"
        else:
            show_info("框选标注模式: 请在Shapes图层中添加矩形框选")
            # 在状态栏显示操作提示
            self.viewer.status = "框选标注模式 | 使用矩形工具添加框选 | 空格键预测"
    
    def _run_prediction(self):
        # 在预测开始时
        self.viewer.status = "正在执行预测..."
        
        if not self.model_loaded or self.current_image is None:
            show_warning("请先加载模型并设置图像")
            return
        
        # 检查Shapes图层
        shapes_layers = [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]
        if not shapes_layers:
            # 创建新的Shapes图层，设置点颜色
            shapes_layer = self.viewer.add_shapes(
                name="标注", 
                ndim=self.current_image.ndim,
                face_color='transparent',
                edge_color='green',
                symbol='o',
                size=self.point_size
            )
            show_info("已创建新的Shapes图层用于标注")
        else:
            # 使用第一个Shapes图层
            shapes_layer = shapes_layers[0]
            # 更新点大小
            shapes_layer.size = self.point_size
        
        # 获取标注数据
        shapes_data = shapes_layer.data
        shapes_types = shapes_layer.shape_type
        
        if len(shapes_data) == 0:
            show_warning("未找到标注数据，请先添加点或框标注")
            return
        
        # 转换shape数据为所需格式
        shapes_metadata = [
            {'shape_type': shape_type} 
            for shape_type in shapes_types
        ]
        
        shapes = []
        for i, (data, metadata) in enumerate(zip(shapes_data, shapes_metadata)):
            shapes.append({
                'data': data,
                'shape_type': metadata['shape_type']
            })
        
        try:
            # 根据模式执行预测
            if self.prediction_mode == "点标注":
                # 进行点标注模式的预测
                self._predict_with_points(shapes, shapes_layer)
            elif self.prediction_mode == "框选标注":
                # 进行框选标注模式的预测
                self._predict_with_box_and_points(shapes, shapes_layer)
            
        except Exception as e:
            show_error(f"预测失败: {str(e)}")
        
        # 预测结束后
        self.viewer.status = "预测完成 | 使用掩码下拉框选择结果 | 按住Shift+点击添加背景点优化"
    
    def _predict_with_points(self, shapes, shapes_layer):
        """使用点标注进行预测"""
        # 获取点标注和它们的类型
        # 使用shapes_layer的feature属性获取点类型
        point_types = []
        if hasattr(shapes_layer, 'features') and 'point_type' in shapes_layer.features:
            point_types = shapes_layer.features['point_type'].tolist()
        
        # 如果没有点类型特征，则为每个点使用当前设置的点类型
        if not point_types or len(point_types) != len([s for s in shapes if s['shape_type'] == 'point']):
            point_types = []
            # 只处理点形状
            for s in shapes:
                if s['shape_type'] == 'point':
                    # 默认使用当前选择的点类型
                    point_types.append(self.point_type)
        
        # 获取点和标签
        points, labels = shapes_to_points(shapes, point_types)
        
        if len(points) == 0:
            show_warning("未找到点标注")
            return
        
        # 执行预测
        multimask = self.multimask_check.isChecked()
        masks, scores, best_idx = self.model.predict_from_points(
            points=points,
            labels=labels,
            multimask_output=multimask
        )
        
        # 保存结果
        self.result_masks = masks
        self.result_scores = scores
        self.selected_mask_idx = best_idx
        
        # 更新UI
        self._update_mask_list()
        
        # 显示最佳掩码
        self._display_mask(masks[best_idx])
        
        # 更新shapes_layer的feature属性，记录点类型
        # 确保shapes_layer有features属性
        if not hasattr(shapes_layer, 'features'):
            shapes_layer.features = {}
        
        # 更新point_type特征
        point_shapes = [s for s in shapes if s['shape_type'] == 'point']
        if point_shapes:
            # 使用numpy数组保存点类型
            shapes_layer.features['point_type'] = np.array(point_types)
    
    def _predict_with_box_and_points(self, shapes, shapes_layer):
        """使用框选和点标注相结合进行预测"""
        # 分离框和点
        box_shapes = [s for s in shapes if s['shape_type'] == 'rectangle']
        point_shapes = [s for s in shapes if s['shape_type'] == 'point']
        
        # 检查是否有框
        if not box_shapes:
            show_warning("未找到框选标注")
            return
        
        # 获取框选
        box = shapes_to_box(box_shapes)
        
        if box.size == 0:
            show_warning("框选格式无效")
            return
        
        # 检查是否有点标注用于优化掩码
        has_points = len(point_shapes) > 0
        
        if has_points:
            # 获取点标注和它们的类型
            point_types = []
            if hasattr(shapes_layer, 'features') and 'point_type' in shapes_layer.features:
                # 提取与点对应的类型
                all_point_indices = [i for i, t in enumerate(shapes_layer.shape_type) if t == 'point']
                if len(all_point_indices) <= len(shapes_layer.features['point_type']):
                    point_types = shapes_layer.features['point_type'].tolist()
            
            # 如果没有点类型特征，则为每个点使用当前设置的点类型
            if not point_types or len(point_types) != len(point_shapes):
                point_types = []
                for _ in point_shapes:
                    # 默认使用当前选择的点类型
                    point_types.append(self.point_type)
            
            # 获取点和标签
            points, labels = shapes_to_points(point_shapes, point_types)
            
            # 执行带点的框选预测
            multimask = self.multimask_check.isChecked()
            masks, scores, best_idx = self.model.predict_from_box_and_points(
                box=box,
                points=points,
                labels=labels,
                multimask_output=multimask
            )
        else:
            # 执行纯框选预测
            multimask = self.multimask_check.isChecked()
            masks, scores, best_idx = self.model.predict_from_box(
                box=box,
                multimask_output=multimask
            )
        
        # 保存结果
        self.result_masks = masks
        self.result_scores = scores
        self.selected_mask_idx = best_idx
        
        # 更新UI
        self._update_mask_list()
        
        # 显示最佳掩码
        self._display_mask(masks[best_idx])
        
        # 更新shapes_layer的feature属性，记录点类型
        if has_points:
            # 确保shapes_layer有features属性
            if not hasattr(shapes_layer, 'features'):
                shapes_layer.features = {}
            
            # 更新point_type特征
            shapes_layer.features['point_type'] = np.array(point_types)
    
    def _update_mask_list(self):
        """更新掩码列表"""
        # 清空当前列表
        self.mask_combo.clear()
        
        # 添加所有掩码到列表
        for i, score in enumerate(self.result_scores):
            self.mask_combo.addItem(f"掩码 {i+1} (得分: {score:.3f})")
        
        # 选择最佳掩码
        self.mask_combo.setCurrentIndex(self.selected_mask_idx)
        
        # 启用控件
        self.mask_combo.setEnabled(True)
        self.add_to_labels_btn.setEnabled(True)
        self.save_current_btn.setEnabled(True)
        self.save_all_btn.setEnabled(True)
        
        # 启用掩码调整按钮
        self.erode_mask_btn.setEnabled(True)
        self.dilate_mask_btn.setEnabled(True)
    
    def _update_selected_mask(self, index):
        """更新选中的掩码"""
        # 修复NumPy数组布尔求值错误
        if (index < 0 or 
            self.result_masks is None or 
            len(self.result_masks) == 0 or 
            index >= len(self.result_masks)):
            return
        
        # 更新选中的掩码索引
        self.selected_mask_idx = index
        
        # 显示选中的掩码
        self._display_mask(self.result_masks[index])
        
        # 更新预览图层(如果存在)
        if "掩码预览" in self.viewer.layers and self.result_masks is not None:
            mask = self.result_masks[index]
            binary_mask = mask_to_binary(mask)
            self.viewer.layers["掩码预览"].data = binary_mask.astype(np.uint8) * 255
    
    def _display_mask(self, mask):
        """显示掩码"""
        # 检查是否已有掩码图层
        mask_layer_name = "MobileSAM掩码"
        if mask_layer_name in self.viewer.layers:
            # 更新现有图层
            self.viewer.layers[mask_layer_name].data = mask
        else:
            # 创建新的图层
            self.viewer.add_image(
                mask,
                name=mask_layer_name,
                colormap="red",
                opacity=0.5,
                blending="additive"
            )
    
    def _add_mask_to_labels(self):
        """将当前掩码添加到Labels图层"""
        if self.result_masks is None or len(self.result_masks) == 0 or self.selected_mask_idx >= len(self.result_masks):
            show_warning("没有可用的掩码结果")
            return
        
        # 获取当前掩码
        mask = self.result_masks[self.selected_mask_idx]
        
        # 转换为二值掩码
        binary_mask = mask_to_binary(mask)
        
        # 获取标签名称
        label_name = self.label_name_combo.currentText().strip()
        if not label_name:
            label_name = f"对象{self.next_label_id}"
        
        # 检查是否已有Labels图层
        labels_layer_name = "分割标签"
        if labels_layer_name in self.viewer.layers:
            # 获取现有的Labels图层
            labels_layer = self.viewer.layers[labels_layer_name]
            
            # 获取新的标签ID
            if hasattr(labels_layer, 'metadata') and 'label_names' in labels_layer.metadata:
                # 从图层的元数据中恢复标签名称映射
                self.label_names = labels_layer.metadata['label_names']
                
                # 如果有颜色映射，也恢复它
                if 'label_colors' in labels_layer.metadata:
                    self.label_colors = labels_layer.metadata['label_colors']
                
                # 根据名称查找对应的ID
                label_id = None
                for key, value in self.label_names.items():
                    if value == label_name:
                        label_id = key
                        break
                
                if label_id is None:
                    # 新标签，分配新ID
                    label_id = max(self.label_names.keys(), default=0) + 1
                    self.label_names[label_id] = label_name
                    # 生成新的颜色
                    self.label_colors[label_id] = self._generate_label_color(label_id)
                    show_info(f"创建新标签: {label_name} (ID: {label_id})")
                else:
                    show_info(f"使用现有标签: {label_name} (ID: {label_id})")
            else:
                # 初始化元数据
                self.label_names = {}
                label_id = self.next_label_id
                self.label_names[label_id] = label_name
                
                # 生成颜色
                self.label_colors = {label_id: self._generate_label_color(label_id)}
                
                # 更新下一个可用ID
                self.next_label_id = label_id + 1
                
                show_info(f"创建标签: {label_name} (ID: {label_id})")
            
            # 更新Labels数据
            labels_data = labels_layer.data.copy()
            
            # 创建重用记录
            if not hasattr(self, 'label_mask_history'):
                self.label_mask_history = {}
                
            # 记录这个标签ID的当前掩码，用于重用
            self.label_mask_history[label_id] = {
                'mask': binary_mask.copy(),
                'name': label_name,
                'mask_idx': self.selected_mask_idx
            }
            
            # 应用掩码 - 先清除该ID的旧掩码
            # 这允许用户修改之前的标签
            labels_data[labels_data == label_id] = 0
            labels_data[binary_mask > 0] = label_id
            
            # 更新图层
            labels_layer.data = labels_data
            
            # 更新元数据
            if not hasattr(labels_layer, 'metadata'):
                labels_layer.metadata = {}
            labels_layer.metadata['label_names'] = self.label_names
            labels_layer.metadata['label_colors'] = self.label_colors
            
            # 尝试设置color
            if hasattr(labels_layer, 'color'):
                color_dict = {id: tuple(color) for id, color in self.label_colors.items()}
                labels_layer.color = color_dict
            
            # 更新标签名称下拉框
            self._update_label_name_combo()
            
            # 在状态栏显示信息
            self.viewer.status = f"已更新标签: {label_name} (ID: {label_id}) - 可以继续添加更多标签或修改现有标签"
            
        else:
            # 创建新的Labels图层
            labels_data = np.zeros_like(mask, dtype=np.int32)
            
            # 分配新标签ID
            label_id = 1
            self.label_names = {label_id: label_name}
            # 生成颜色
            self.label_colors = {label_id: self._generate_label_color(label_id)}
            self.next_label_id = label_id + 1
            
            # 设置标签
            labels_data[binary_mask > 0] = label_id
            
            # 创建图层及元数据
            labels_layer = self.viewer.add_labels(
                labels_data,
                name=labels_layer_name,
                metadata={
                    'label_names': self.label_names,
                    'label_colors': self.label_colors
                }
            )
            
            # 初始化标签历史记录
            self.label_mask_history = {
                label_id: {
                    'mask': binary_mask.copy(),
                    'name': label_name,
                    'mask_idx': self.selected_mask_idx
                }
            }
            
            # 尝试设置颜色
            if hasattr(labels_layer, 'color'):
                color_dict = {id: tuple(color) for id, color in self.label_colors.items()}
                labels_layer.color = color_dict
            
            # 更新标签名称下拉框
            self._update_label_name_combo()
            
            show_info(f"已创建新的Labels图层并添加掩码 (标签: {label_name}, ID: {label_id})")
        
        # 启用标签管理按钮
        self.export_labels_btn.setEnabled(True)
        self.clear_labels_btn.setEnabled(True)
    
    def _update_label_name_combo(self):
        """更新标签名称下拉框"""
        # 暂存当前文本
        current_text = self.label_name_combo.currentText()
        
        # 清空下拉框
        self.label_name_combo.clear()
        
        # 添加现有标签名称
        unique_names = sorted(set(self.label_names.values()))
        self.label_name_combo.addItems(unique_names)
        
        # 如果当前文本不在列表中，添加它
        if current_text and current_text not in unique_names:
            self.label_name_combo.addItem(current_text)
            self.label_name_combo.setCurrentText(current_text)
        elif current_text in unique_names:
            self.label_name_combo.setCurrentText(current_text)
    
    def _save_current_mask(self):
        """保存当前掩码"""
        if self.result_masks is None or len(self.result_masks) == 0 or self.selected_mask_idx >= len(self.result_masks):
            show_warning("没有可用的掩码结果")
            return
        
        # 选择保存目录
        output_dir = self._get_output_directory()
        if not output_dir:
            return
        
        # 获取当前掩码和分数
        current_mask = self.result_masks[self.selected_mask_idx:self.selected_mask_idx+1]
        current_score = self.result_scores[self.selected_mask_idx:self.selected_mask_idx+1]
        
        # 保存掩码
        base_name = self.prefix_edit.text() or "mask"
        image_name = None
        if self.auto_naming_check.isChecked() and self.current_layer:
            image_name = self.current_layer.name
        
        saved_paths = save_masks(
            masks=current_mask,
            scores=current_score,
            output_dir=output_dir,
            image_name=image_name,
            base_name=base_name
        )
        
        show_info(f"已保存掩码到: {saved_paths[0]}")
    
    def _save_all_masks(self):
        """保存所有掩码"""
        if self.result_masks is None or len(self.result_masks) == 0:
            show_warning("没有可用的掩码结果")
            return
        
        # 选择保存目录
        output_dir = self._get_output_directory()
        if not output_dir:
            return
        
        # 保存所有掩码
        base_name = self.prefix_edit.text() or "mask"
        image_name = None
        if self.auto_naming_check.isChecked() and self.current_layer:
            image_name = self.current_layer.name
        
        saved_paths = save_masks(
            masks=self.result_masks,
            scores=self.result_scores,
            output_dir=output_dir,
            image_name=image_name,
            base_name=base_name
        )
        
        show_info(f"已保存 {len(saved_paths)} 个掩码到: {output_dir}")
    
    def _select_output_directory(self):
        """选择输出目录"""
        # 打开目录选择对话框
        dir_dialog = QFileDialog()
        dir_path = dir_dialog.getExistingDirectory(
            self, "选择输出目录", ""
        )
        
        if dir_path:
            # 设置输出目录
            self.output_dir_edit.setText(dir_path)
    
    def _get_output_directory(self):
        """获取输出目录"""
        # 如果批处理标签页已设置目录，则使用该目录
        output_dir = self.output_dir_edit.text()
        
        if not output_dir:
            # 打开目录选择对话框
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(
                self, "选择输出目录", ""
            )
            
            if dir_path:
                # 设置输出目录
                self.output_dir_edit.setText(dir_path)
                output_dir = dir_path
            else:
                return None
        
        return output_dir
    
    # 批处理相关函数
    
    def _add_to_queue(self):
        """添加当前图像到处理队列"""
        # TODO: 实现批处理队列
        # 这里只是示例框架，实际的队列实现可以根据需要扩展
        show_info("批处理功能尚未实现")
        self.process_queue_btn.setEnabled(True)
    
    def _process_queue(self):
        """处理队列中的所有图像"""
        # TODO: 实现批处理逻辑
        show_info("批处理功能尚未实现")
    
    def _import_image_folder(self):
        """导入图片文件夹"""
        # 打开文件夹选择对话框
        folder_dialog = QFileDialog()
        folder_path = folder_dialog.getExistingDirectory(
            self, "选择图片文件夹", ""
        )
        
        if not folder_path:
            return
        
        # 设置文件夹路径
        self.image_folder_path = folder_path
        # 显示文件夹名称，如果路径太长则截断显示
        folder_name = os.path.basename(folder_path)
        # 简化路径显示
        if len(folder_path) > 30:
            displayed_path = f"已选择: {folder_name}"
        else:
            displayed_path = f"已选择: {folder_path}"
        self.folder_path_label.setText(displayed_path)
        self.folder_path_label.setToolTip(folder_path)  # 添加完整路径作为工具提示
        
        # 获取文件夹中的所有图片文件
        self._scan_image_folder()
        
        # 加载第一张图片
        if self.image_files:
            self._load_image_by_index(0)
    
    def _scan_image_folder(self):
        """扫描图片文件夹，获取所有图片文件"""
        self.image_files = []
        
        if not self.image_folder_path or not os.path.exists(self.image_folder_path):
            return
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
        
        # 获取所有图片文件
        for file in sorted(os.listdir(self.image_folder_path)):
            file_path = os.path.join(self.image_folder_path, file)
            if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
                self.image_files.append(file_path)
        
        # 更新计数器
        self._update_image_counter()
    
    def _update_image_counter(self):
        """更新图片计数器"""
        total = len(self.image_files)
        current = self.current_image_index + 1 if self.current_image_index >= 0 else 0
        self.image_counter_label.setText(f"{current}/{total}")
        
        # 更新导航按钮状态
        self.prev_image_btn.setEnabled(self.current_image_index > 0)
        self.next_image_btn.setEnabled(self.current_image_index < total - 1)
    
    def _load_image_by_index(self, index):
        """根据索引加载图片"""
        if index < 0 or index >= len(self.image_files):
            return
        
        # 更新当前索引
        self.current_image_index = index
        
        # 获取图片路径
        image_path = self.image_files[index]
        
        try:
            # 读取图片
            from skimage import io
            image = io.imread(image_path)
            
            # 添加到napari查看器
            if 'imported_image' in self.viewer.layers:
                self.viewer.layers.remove('imported_image')
            
            self.viewer.add_image(image, name='imported_image')
            
            # 自动设置为当前图像
            self.image_combo.setCurrentText('imported_image')
            self._set_current_image()
            
            # 更新计数器
            self._update_image_counter()
            
            # 清除之前的标注
            self._clear_annotations()
            
        except Exception as e:
            show_error(f"加载图片失败: {str(e)}")
    
    def _clear_annotations(self):
        """清除之前的标注"""
        # 清除Shapes图层
        if '标注' in self.viewer.layers:
            shapes_layer = self.viewer.layers['标注']
            shapes_layer.data = []
            # 重置特征数据
            if hasattr(shapes_layer, 'features'):
                shapes_layer.features = {}
                if 'point_type' not in shapes_layer.features:
                    shapes_layer.features['point_type'] = np.array([], dtype=np.int32)
        
        # 清除掩码图层
        if 'MobileSAM掩码' in self.viewer.layers:
            self.viewer.layers.remove('MobileSAM掩码')
        
        # 清除预览图层
        if '掩码预览' in self.viewer.layers:
            self.viewer.layers.remove('掩码预览')
        
        # 重置结果
        self.result_masks = []
        self.result_scores = []
        self.selected_mask_idx = 0
        
        # 清空掩码列表
        self.mask_combo.clear()
        self.mask_combo.setEnabled(False)
        self.add_to_labels_btn.setEnabled(False)
        self.save_current_btn.setEnabled(False)
        self.save_all_btn.setEnabled(False)
        
        # 不清除标签名称列表，以保持标签一致性
        # self.label_name_combo.clear()
        
        # 显示清除完成的消息
        self.viewer.status = "已清除所有标注，可以开始新的标注"
    
    def _load_prev_image(self):
        """加载上一张图片"""
        if self.current_image_index > 0:
            self._load_image_by_index(self.current_image_index - 1)
    
    def _load_next_image(self):
        """加载下一张图片"""
        if self.current_image_index < len(self.image_files) - 1:
            self._load_image_by_index(self.current_image_index + 1)
    
    def _update_point_type(self, checked):
        """更新点类型"""
        if self.positive_point_radio.isChecked():
            self.point_type = 1  # 前景点
        else:
            self.point_type = 0  # 背景点
    
    def _connect_shapes_layer_events(self):
        """连接Shapes图层的事件"""
        # 首次连接时，监听viewer的图层添加事件
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        
        # 连接已存在的Shapes图层
        for layer in self.viewer.layers:
            if isinstance(layer, Shapes):
                self._connect_shapes_events(layer)
    
    def _on_layer_inserted(self, event):
        """当有新图层插入时触发"""
        layer = event.value
        if isinstance(layer, Shapes):
            self._connect_shapes_events(layer)
    
    def _connect_shapes_events(self, shapes_layer):
        """连接特定Shapes图层的事件"""
        # 确保我们不会重复连接同一个图层
        if not hasattr(shapes_layer, '_mobilesam_connected'):
            # 监听数据变化事件
            shapes_layer.events.data.connect(self._on_shapes_data_changed)
            # 添加点事件监听
            shapes_layer.events.data.connect(lambda e: self._update_shape_features(shapes_layer))
            shapes_layer.mouse_drag_callbacks.append(self._shapes_mouse_drag_callback)
            # 标记为已连接
            shapes_layer._mobilesam_connected = True
            # 添加自动计时器，防止数据变化立即触发预测
            shapes_layer._last_changed_time = time.time()
            
            # 初始化特征属性
            if not hasattr(shapes_layer, 'features'):
                shapes_layer.features = {}
            if 'point_type' not in shapes_layer.features:
                shapes_layer.features['point_type'] = np.array([], dtype=np.int32)
    
    def _shapes_mouse_drag_callback(self, layer, event):
        """监听shapes图层的鼠标拖拽事件，用于捕获点添加"""
        # 仅在点标注模式下处理
        if self.prediction_mode != "点标注":
            return
        
        # 鼠标释放时检查是否添加了新点
        if event.type == 'mouse_release' and event.button == 1:  # 左键点击
            # 延迟一点执行，确保shapes数据已完全更新
            from qtpy.QtCore import QTimer
            QTimer.singleShot(50, lambda: self._update_shape_features(layer))
    
    def _update_shape_features(self, shapes_layer):
        """更新shapes图层的特征属性，记录点类型"""
        if self.prediction_mode != "点标注":
            return
        
        # 确保有features属性
        if not hasattr(shapes_layer, 'features'):
            shapes_layer.features = {}
        
        # 获取点形状的数量
        point_shapes = [s for s, t in zip(shapes_layer.data, shapes_layer.shape_type) if t == 'point']
        point_count = len(point_shapes)
        
        # 获取当前点类型特征数组
        current_point_types = shapes_layer.features.get('point_type', np.array([], dtype=np.int32))
        
        # 如果点数量和特征数量不一致，需要更新
        if len(current_point_types) != point_count:
            # 之前记录的点数量
            prev_point_count = len(current_point_types)
            
            # 为新增的点添加类型标记
            if point_count > prev_point_count:
                # 新点的数量
                new_points_count = point_count - prev_point_count
                # 为新点设置当前点类型
                new_point_types = np.full(new_points_count, self.point_type, dtype=np.int32)
                # 合并点类型数组
                updated_point_types = np.concatenate([current_point_types, new_point_types])
                # 更新特征
                shapes_layer.features['point_type'] = updated_point_types
                
                # 更新点的颜色
                self._update_point_colors(shapes_layer)
                
                # 显示当前操作的提示
                point_type_name = "前景点" if self.point_type == 1 else "背景点"
                show_info(f"添加了 {new_points_count} 个{point_type_name}")
    
    def _on_shapes_data_changed(self, event):
        """当Shapes图层数据变化时触发"""
        # 获取触发事件的图层
        shapes_layer = event.source
        
        # 更新shape特征
        self._update_shape_features(shapes_layer)
        
        # 标记图层变化时间
        current_time = time.time()
        shapes_layer._last_changed_time = current_time
        
        # 如果启用了自动预测，则延迟执行预测
        if self.auto_predict_check.isChecked() and self.model_loaded and self.current_image is not None:
            # 延迟执行，确保shapes已经稳定且用户完成操作
            from qtpy.QtCore import QTimer
            
            # 根据模式设置不同的延迟
            if self.prediction_mode == "点标注":
                # 点标注模式下延迟短一些
                delay = 100  # 100毫秒
            else:
                # 框选模式下延迟长一些，给用户调整框的时间
                delay = 1000  # 1000毫秒，即1秒
                
                # 检查是否为矩形工具
                if hasattr(shapes_layer, 'mode') and shapes_layer.mode != 'add_rectangle':
                    # 如果不是矩形工具，显示提示
                    self.viewer.status = "框选标注模式需要使用矩形工具，请切换工具"
                    return
                
                # 检查是否有完整的矩形
                complete_rectangles = [s for s in shapes_layer.data if 
                                       s.shape == (4, 2) and 
                                       shapes_layer.shape_type[shapes_layer.data.index(s)] == 'rectangle']
                
                if not complete_rectangles:
                    # 如果没有完整的矩形，不触发预测
                    return
            
            # 防止重复触发，使用lambda捕获当前时间
            QTimer.singleShot(delay, lambda t=current_time: self._delayed_prediction(t))
    
    def _select_input_directory(self):
        """选择输入目录"""
        # 打开目录选择对话框
        dir_dialog = QFileDialog()
        dir_path = dir_dialog.getExistingDirectory(
            self, "选择输入目录", ""
        )
        
        if dir_path:
            # 设置输入目录
            self.input_dir_edit.setText(dir_path)
            
            # 如果输出目录未设置，默认设置为输入目录下的 "outputs" 文件夹
            if not self.output_dir_edit.text():
                default_output = os.path.join(dir_path, "outputs")
                self.output_dir_edit.setText(default_output)

    def _set_positive_point(self, viewer):
        """设置为前景点模式"""
        self.point_type = 1
        self.point_type_changed_signal.emit(1)

    def _set_negative_point(self, viewer):
        """设置为背景点模式"""
        self.point_type = 0
        self.point_type_changed_signal.emit(0)

    def _update_point_colors(self, shapes_layer):
        """根据点类型更新点的颜色"""
        # 获取点类型特征
        if not hasattr(shapes_layer, 'features') or 'point_type' not in shapes_layer.features:
            return
        
        point_types = shapes_layer.features['point_type']
        
        # 确保有足够的数据来更新
        if len(point_types) == 0:
            return
        
        # 获取点形状的索引
        point_indices = [i for i, t in enumerate(shapes_layer.shape_type) if t == 'point']
        
        if len(point_indices) == 0:
            return
        
        # 准备颜色列表
        face_colors = []
        edge_colors = []
        
        # 设置所有形状的默认颜色
        for i in range(len(shapes_layer.data)):
            if i in point_indices:
                # 点形状，根据点类型设置颜色
                idx = point_indices.index(i)
                if idx < len(point_types):
                    if point_types[idx] == 1:  # 前景点
                        face_colors.append([0, 1, 0, 0.5])  # 绿色半透明
                        edge_colors.append([0, 1, 0, 1])    # 绿色边缘
                    else:  # 背景点
                        face_colors.append([1, 0, 0, 0.5])  # 红色半透明
                        edge_colors.append([1, 0, 0, 1])    # 红色边缘
                else:
                    # 默认颜色
                    face_colors.append([0, 1, 0, 0.5])
                    edge_colors.append([0, 1, 0, 1])
            else:
                # 非点形状使用默认颜色
                face_colors.append([0, 0, 1, 0.3])  # 蓝色半透明
                edge_colors.append([0, 0, 1, 1])    # 蓝色边缘
        
        # 更新shapes图层的颜色
        shapes_layer.face_color = face_colors
        shapes_layer.edge_color = edge_colors
        shapes_layer.refresh()

    def _show_multimask_help(self):
        """显示多掩码生成机制和得分依据的说明"""
        help_dialog = QDialog(self)
        help_dialog.setWindowTitle("关于多掩码生成与得分")
        help_dialog.resize(600, 400)
        help_dialog.setStyleSheet("""
            QDialog {
                background-color: #f2f2f7;
                border-radius: 14px;
            }
            QLabel {
                color: #000000;
                font-weight: 600;
                font-size: 16px;
            }
            QTextBrowser {
                background-color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 8px;
                font-family: 'SF Pro Display', '-apple-system', 'Helvetica Neue', sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 20px;
                font-weight: 600;
                min-height: 36px;
            }
            QPushButton:hover {
                background-color: #0062cc;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        title_label = QLabel("多掩码生成机制与得分依据")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; margin-bottom: 10px;")
        
        help_text = QTextBrowser()
        help_text.setOpenExternalLinks(True)
        help_text.setHtml("""
        <style>
            body {
                font-family: 'SF Pro Display', '-apple-system', 'Helvetica Neue', sans-serif;
                line-height: 1.5;
                color: #000000;
            }
            h3 {
                color: #007aff;
                font-weight: 600;
                margin-top: 16px;
                margin-bottom: 8px;
            }
            p {
                margin: 8px 0;
            }
            ul, ol {
                margin-top: 8px;
                margin-bottom: 8px;
            }
            li {
                margin-bottom: 4px;
            }
            a {
                color: #007aff;
                text-decoration: none;
            }
        </style>
        
        <h3>多掩码生成机制</h3>
        <p>当启用"生成多个掩码候选"选项时，MobileSAM会为每个标注生成多个可能的掩码，这是通过在模型内部使用不同的解码器参数实现的。主要的多掩码生成过程包括：</p>
        <ol>
            <li><b>潜在掩码解码</b>：模型内部会从图像编码和提示编码中生成一组潜在掩码表示</li>
            <li><b>多重阈值解码</b>：使用不同的稳定性阈值对潜在掩码进行解码，产生多个候选掩码</li>
            <li><b>多级IoU预测</b>：针对每个候选掩码，预测其与真实掩码的IoU（交并比）得分</li>
        </ol>
        
        <h3>得分依据</h3>
        <p>掩码得分是模型内部的预测质量指标，主要反映这个掩码与"正确掩码"的匹配程度。得分越高表示模型对这个掩码的预测越有信心。具体而言：</p>
        <ul>
            <li><b>IoU预测值</b>：是模型预测的掩码与真实掩码可能的交并比。范围为0-1，越接近1表示质量越高</li>
            <li><b>稳定性得分</b>：评估掩码在轻微的阈值变化下的稳定性</li>
            <li><b>与提示的一致性</b>：评估掩码与用户输入的点或框的一致程度</li>
        </ul>
        
        <h3>选择掩码的建议</h3>
        <p>虽然系统默认选择得分最高的掩码，但您也可以：</p>
        <ul>
            <li>通过下拉菜单查看所有候选掩码</li>
            <li>选择更符合您预期的掩码，即使其得分不是最高</li>
            <li>如果最高得分的掩码不令人满意，可以添加更多的标注点（特别是负点/背景点）来优化结果</li>
        </ul>
        
        <p>参考：<a href="https://github.com/ChaoningZhang/MobileSAM">MobileSAM 项目</a> | <a href="https://arxiv.org/abs/2304.02643">Segment Anything 论文</a></p>
        """)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(help_dialog.close)
        
        layout.addWidget(title_label)
        layout.addWidget(help_text)
        layout.addWidget(close_btn)
        
        help_dialog.setLayout(layout)
        help_dialog.exec_()

    def _generate_label_color(self, label_id):
        """生成标签的颜色"""
        # 使用黄金分割比例来生成分布均匀的颜色
        golden_ratio_conjugate = 0.618033988749895
        h = (label_id * golden_ratio_conjugate) % 1.0
        # 固定饱和度和明度，只变化色相
        s = 0.8
        v = 0.95
        # 转换为RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # 返回RGBA格式
        return [r, g, b, 1.0]

    def _export_label_info(self):
        """导出标签信息到JSON文件"""
        if not self.label_names:
            show_warning("没有标签信息可以导出")
            return
        
        # 选择保存文件
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "导出标签信息", "", "JSON文件 (*.json)"
        )
        
        if not file_path:
            return
        
        # 准备导出数据
        export_data = {
            "labels": {
                str(label_id): {
                    "name": name,
                    "color": self.label_colors.get(label_id, [0, 0, 0, 1])
                } 
                for label_id, name in self.label_names.items()
            }
        }
        
        # 添加当前图像信息
        if self.current_layer:
            export_data["current_image"] = self.current_layer.name
        
        # 导出到JSON
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            show_info(f"标签信息已导出到: {file_path}")
        except Exception as e:
            show_error(f"导出标签信息失败: {str(e)}")

    def _clear_all_labels(self):
        """清除所有标签"""
        if "分割标签" not in self.viewer.layers:
            show_warning("没有标签图层可清除")
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            "确认清除", 
            "确定要清除所有标签吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 移除Labels图层
            self.viewer.layers.remove("分割标签")
            
            # 重置标签管理
            self.label_names = {}
            self.label_colors = {}
            self.next_label_id = 1
            
            # 更新UI
            self.label_name_combo.clear()
            self.export_labels_btn.setEnabled(False)
            self.clear_labels_btn.setEnabled(False)
            
            show_info("已清除所有标签")

    def _preview_selected_mask(self):
        """在标签图层上预览当前选中的掩码"""
        if self.result_masks is None or len(self.result_masks) == 0 or self.selected_mask_idx >= len(self.result_masks):
            show_warning("没有可用的掩码结果可预览")
            return
        
        # 获取当前掩码
        mask = self.result_masks[self.selected_mask_idx]
        binary_mask = mask_to_binary(mask)
        
        # 检查是否已有预览图层
        preview_layer_name = "掩码预览"
        
        # 如果存在，更新它，否则创建新的
        if preview_layer_name in self.viewer.layers:
            # 更新现有图层
            self.viewer.layers[preview_layer_name].data = binary_mask.astype(np.uint8) * 255
        else:
            # 创建新的图层用于预览
            preview_layer = self.viewer.add_image(
                binary_mask.astype(np.uint8) * 255,
                name=preview_layer_name,
                colormap="magenta",
                opacity=0.5,
                blending="additive"
            )
            
            # 将预览图层置于最顶层 - 修复参数缺失错误
            if preview_layer_name in self.viewer.layers:
                # 获取图层索引
                layer_index = self.viewer.layers.index(preview_layer_name)
                # 正确调用move_selected方法
                self.viewer.layers.move_selected(layer_index, len(self.viewer.layers) - 1)
        
        show_info(f"正在预览掩码 {self.selected_mask_idx + 1}，请确认后添加到标签或调整边界")

    def _adjust_mask_boundary(self, operation_type):
        """调整掩码边界（收缩或扩张）
        
        参数:
            operation_type: 1表示扩张，-1表示收缩
        """
        if self.result_masks is None or len(self.result_masks) == 0 or self.selected_mask_idx >= len(self.result_masks):
            show_warning("没有可用的掩码结果可调整")
            return
        
        # 获取当前掩码
        mask = self.result_masks[self.selected_mask_idx]
        binary_mask = mask_to_binary(mask)
        
        # 创建结构元素
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 根据操作类型执行形态学操作
        if operation_type > 0:
            # 扩张
            adjusted_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1)
            operation_name = "扩张"
        else:
            # 收缩
            adjusted_mask = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
            operation_name = "收缩"
        
        # 更新掩码
        self.result_masks[self.selected_mask_idx] = adjusted_mask.astype(np.float32)
        
        # 更新显示
        self._display_mask(self.result_masks[self.selected_mask_idx])
        
        # 如果有预览图层，也更新它
        if "掩码预览" in self.viewer.layers:
            self.viewer.layers["掩码预览"].data = adjusted_mask * 255
        
        show_info(f"已{operation_name}掩码边界") 

    def _delayed_prediction(self, changed_time):
        """延迟执行预测，确保用户完成操作"""
        # 查找Shapes图层
        shapes_layers = [layer for layer in self.viewer.layers if isinstance(layer, Shapes)]
        if not shapes_layers:
            return
            
        shapes_layer = shapes_layers[0]
        
        # 检查时间戳是否匹配
        if not hasattr(shapes_layer, '_last_changed_time') or shapes_layer._last_changed_time != changed_time:
            # 时间戳不匹配，说明在延迟期间又发生了变化，不执行预测
            return
            
        # 在框选模式下，进行额外检查
        if self.prediction_mode == "框选标注":
            # 检查是否有完整的矩形
            complete_rectangles = [s for s in shapes_layer.data if 
                                  s.shape == (4, 2) and 
                                  shapes_layer.shape_type[shapes_layer.data.index(s)] == 'rectangle']
            
            if not complete_rectangles:
                # 如果没有完整的矩形，不触发预测
                return
                
            # 显示提示，让用户知道延迟预测即将开始
            self.viewer.status = "框选完成，正在执行预测..."
        
        # 执行预测
        self._run_prediction()

# 创建可折叠的分组框(可选实现)
class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("text-align:left;")
        self.toggle_button.clicked.connect(self.on_clicked)
        
        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        
        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)
        
    def on_clicked(self, checked):
        if checked:
            self.content_area.setMaximumHeight(1000)
        else:
            self.content_area.setMaximumHeight(0)