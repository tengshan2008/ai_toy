#!/usr/bin/env python3
"""
安装代码标准化所需的依赖项。

该脚本检查环境并安装所需的依赖项。
"""

import sys
import subprocess
import logging
import platform
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_pip():
    """检查 pip 是否可用。"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def install_package(package_name, version=None):
    """
    安装指定的包。
    
    Args:
        package_name: 包名
        version: 版本（如果为 None，则安装最新版本）
    
    Returns:
        是否安装成功
    """
    package_spec = f"{package_name}=={version}" if version else package_name
    
    logger.info(f"安装 {package_spec}...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package_spec],
            check=True,
            capture_output=True
        )
        logger.info(f"✓ {package_spec} 安装成功")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"✗ 安装 {package_spec} 失败: {e}")
        return False

def install_dependencies():
    """安装所有依赖项。"""
    # 检查 pip
    if not check_pip():
        logger.error("✗ pip 不可用，无法安装依赖项")
        return False
    
    # 获取系统信息
    system_info = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }
    
    logger.info(f"系统信息: Python {system_info['python_version']} on {system_info['system']} {system_info['machine']}")
    
    # 安装核心依赖项
    core_dependencies = [
        ("torch", None),
        ("torchvision", None),
        ("sentence-transformers", None),
        ("transformers", None),
        ("fastapi", None),
        ("uvicorn", None),
        ("pydantic", None)
    ]
    
    # 安装可选依赖项
    optional_dependencies = [
        ("numpy", None),
        ("nltk", None),
        ("scikit-learn", None),
        ("pandas", None)
    ]
    
    # 安装核心依赖项
    logger.info("\n=== 安装核心依赖项 ===")
    core_success = True
    for package, version in core_dependencies:
        if not install_package(package, version):
            core_success = False
    
    # 安装可选依赖项
    logger.info("\n=== 安装可选依赖项 ===")
    optional_success = True
    for package, version in optional_dependencies:
        if not install_package(package, version):
            optional_success = False
    
    # 总结
    logger.info("\n=== 安装结果 ===")
    if core_success:
        logger.info("✓ 所有核心依赖项安装成功！")
    else:
        logger.error("✗ 部分核心依赖项安装失败")
    
    if optional_success:
        logger.info("✓ 所有可选依赖项安装成功！")
    else:
        logger.warning("⚠ 部分可选依赖项安装失败，但这不会影响核心功能")
    
    return core_success

def main():
    """主函数。"""
    logger.info("=== 代码标准化依赖项安装 ===\n")
    
    # 安装依赖项
    if install_dependencies():
        logger.info("\n✓ 依赖项安装完成！现在可以运行代码标准化 API 了。")
        logger.info("  运行命令: python run_api.py")
        return 0
    else:
        logger.error("\n✗ 依赖项安装失败。请手动安装所需的依赖项。")
        logger.info("  手动安装命令: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
