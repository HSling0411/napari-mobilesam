from setuptools import setup, find_packages

setup(
    name="napari-mobilesam",
    version="0.1.0",
    description="MobileSAM segmentation plugin for napari",
    author="napari-mobilesam developers",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/napari-mobilesam",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "napari>=0.4.16",
        "numpy",
        "torch>=1.13.0",
        "opencv-python",
        "scikit-image",
        "qtpy",
    ],
    entry_points={
        "napari.manifest": [
            "napari-mobilesam = napari_mobilesam:napari.yaml",
        ],
    },
) 