import numpy as np
from napari_mobilesam import MobileSamWidget


def test_widget_creation():
    """测试小部件创建"""
    viewer = None  # 这里应该是一个mock的viewer对象
    widget = MobileSamWidget(viewer)
    assert widget is not None


# 模拟数据测试
def test_shapes_to_points():
    """测试shapes_to_points函数"""
    from napari_mobilesam.utils import shapes_to_points
    
    # 创建模拟的shapes数据
    shapes = [
        {'data': np.array([[10, 20]]), 'shape_type': 'point'},
        {'data': np.array([[30, 40]]), 'shape_type': 'point'},
        {'data': np.array([[1, 2], [3, 4]]), 'shape_type': 'rectangle'},  # 应该被忽略
    ]
    
    # 测试转换
    points, labels = shapes_to_points(shapes, [1, 1])
    
    # 验证结果
    assert points.shape == (2, 2)
    assert np.array_equal(points[0], np.array([10, 20]))
    assert np.array_equal(points[1], np.array([30, 40]))
    assert np.array_equal(labels, np.array([1, 1]))


def test_shapes_to_box():
    """测试shapes_to_box函数"""
    from napari_mobilesam.utils import shapes_to_box
    
    # 创建模拟的shapes数据
    shapes = [
        {'data': np.array([[10, 10], [10, 20], [20, 20], [20, 10]]), 'shape_type': 'rectangle'},
        {'data': np.array([[30, 30]]), 'shape_type': 'point'},  # 应该被忽略
    ]
    
    # 测试转换
    box = shapes_to_box(shapes)
    
    # 验证结果
    assert box.shape == (4,)
    assert np.array_equal(box, np.array([10, 10, 20, 20]))


def test_mask_to_binary():
    """测试mask_to_binary函数"""
    from napari_mobilesam.utils import mask_to_binary
    
    # 创建模拟的掩码数据
    mask = np.array([[0.1, 0.6], [0.8, 0.3]])
    
    # 测试转换
    binary = mask_to_binary(mask, threshold=0.5)
    
    # 验证结果
    assert binary.shape == (2, 2)
    assert np.array_equal(binary, np.array([[0, 1], [1, 0]]))


def test_generate_unique_name():
    """测试generate_unique_name函数"""
    from napari_mobilesam.utils import generate_unique_name
    
    # 测试生成唯一名称
    name1 = generate_unique_name()
    name2 = generate_unique_name()
    
    # 验证结果
    assert name1 != name2
    assert name1.startswith("mask_")
    
    # 测试自定义前缀
    custom_name = generate_unique_name("test")
    assert custom_name.startswith("test_") 