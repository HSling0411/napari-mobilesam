#!/usr/bin/env python
"""
测试 napari-mobilesam 是否已正确安装的脚本。
此脚本检查所有必要的依赖项是否都可以正确导入，并验证 MPS 后端（对于 Mac M 系列芯片）是否可用。
"""

import sys
import importlib
import platform

def check_import(module_name, package_name=None):
    """尝试导入模块并返回成功状态"""
    if package_name is None:
        package_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✅ 成功导入 {module_name}")
        return True
    except ImportError as e:
        print(f"❌ 无法导入 {module_name}: {e}")
        print(f"   请尝试安装: pip install {package_name}")
        return False

def main():
    """运行所有检查"""
    print("\n=== napari-mobilesam 安装测试 ===\n")
    
    # 检查 Python 版本
    py_version = sys.version.split()[0]
    print(f"Python 版本: {py_version}")
    
    if int(py_version.split('.')[0]) < 3 or int(py_version.split('.')[1]) < 8:
        print("❌ Python 版本应为 3.8 或更高")
    else:
        print("✅ Python 版本兼容")
    
    # 检查系统
    system = platform.system()
    print(f"操作系统: {system}")
    
    if system == "Darwin":
        print("✅ 在 macOS 上运行")
        # 检查 Apple 芯片
        machine = platform.machine()
        if machine == "arm64":
            print("✅ 检测到 Apple M 系列芯片")
        else:
            print(f"ℹ️ 检测到 {machine} 架构（非 Apple M 系列芯片）")
    else:
        print(f"ℹ️ 在 {system} 上运行（非 macOS）")
    
    # 检查基本依赖
    required_modules = [
        ("numpy", "numpy"),
        ("napari", "napari[all]"),
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("skimage", "scikit-image"),
        ("qtpy", "qtpy"),
    ]
    
    all_deps_ok = True
    for module, package in required_modules:
        if not check_import(module, package):
            all_deps_ok = False
    
    # 检查 PyTorch MPS 可用性
    if check_import("torch"):
        import torch
        print("\n--- PyTorch 后端检查 ---")
        print(f"PyTorch 版本: {torch.__version__}")
        
        # 检查 MPS
        if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
            mps_available = torch.backends.mps.is_available()
            if mps_available:
                print("✅ MPS 后端可用")
            else:
                print("❌ MPS 后端不可用")
                print("   可能原因: macOS 版本过低 (需要 12.3+) 或 PyTorch 版本不支持")
        else:
            print("❌ 您的 PyTorch 版本不支持 MPS 后端")
            print("   请升级到 PyTorch 1.13 或更高版本")
    
    # 检查插件
    try:
        from napari_mobilesam import MobileSamWidget
        print("\n✅ napari-mobilesam 插件已成功安装!")
    except ImportError:
        all_deps_ok = False
        print("\n❌ 无法导入 napari-mobilesam 插件")
        print("   请检查安装: pip install -e .")
    
    # 总结
    print("\n=== 检查摘要 ===")
    if all_deps_ok:
        print("✅ 所有依赖项都已安装")
        print("✅ 可以运行: python examples/demo.py")
    else:
        print("❌ 有些依赖项缺失，请解决上述问题")
    
    print("\n如果您遇到问题，请参考 INSTALL.md 和 README.md 获取更多信息。")

if __name__ == "__main__":
    main() 