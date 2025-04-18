#!/usr/bin/env python3
"""
测试安装和依赖项。

该脚本测试所需的依赖项是否已正确安装。
"""

import sys
import importlib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 要测试的依赖项
DEPENDENCIES = [
    # 核心依赖项
    ('torch', None, '用于张量计算'),
    ('torchvision', None, '与 PyTorch 配合使用'),
    ('sentence_transformers', None, '用于语义相似度计算'),
    ('transformers', None, '用于 NLP 模型'),
    ('fastapi', None, '用于 API 服务'),
    ('uvicorn', None, '用于运行 FastAPI 应用'),
    ('pydantic', None, '用于数据验证'),

    # 可选依赖项
    ('numpy', None, '用于数值计算'),
    ('nltk', None, '用于文本处理'),
    ('sklearn', None, '用于机器学习算法'),
    ('pandas', None, '用于数据处理')
]

def check_dependency(name, version=None, description=None):
    """
    检查依赖项是否已安装。

    Args:
        name: 依赖项名称
        version: 所需版本（如果为 None，则只检查是否已安装）
        description: 依赖项描述

    Returns:
        是否满足要求
    """
    try:
        module = importlib.import_module(name)
        if hasattr(module, '__version__'):
            installed_version = module.__version__
        elif hasattr(module, 'VERSION'):
            installed_version = module.VERSION
        else:
            installed_version = "未知"

        if version is None:
            logger.info(f"✓ {name} 已安装 (版本: {installed_version}) - {description}")
            return True
        else:
            if installed_version == version:
                logger.info(f"✓ {name} 已安装 (版本: {installed_version}) - {description}")
                return True
            else:
                logger.warning(f"✗ {name} 版本不匹配 (已安装: {installed_version}, 需要: {version}) - {description}")
                return False
    except ImportError:
        logger.error(f"✗ {name} 未安装 - {description}")
        return False

def check_all_dependencies():
    """检查所有依赖项。"""
    logger.info("开始检查依赖项...")

    all_passed = True
    core_passed = True

    # 检查核心依赖项
    logger.info("\n=== 核心依赖项 ===")
    for i, (name, version, description) in enumerate(DEPENDENCIES[:7]):
        if not check_dependency(name, version, description):
            all_passed = False
            core_passed = False

    # 检查可选依赖项
    logger.info("\n=== 可选依赖项 ===")
    for i, (name, version, description) in enumerate(DEPENDENCIES[7:]):
        if not check_dependency(name, version, description):
            all_passed = False

    # 检查 Python 版本
    python_version = sys.version
    logger.info(f"\nPython 版本: {python_version}")

    # 总结
    logger.info("\n=== 检查结果 ===")
    if all_passed:
        logger.info("✓ 所有依赖项检查通过！")
    elif core_passed:
        logger.info("⚠ 核心依赖项检查通过，但部分可选依赖项缺失。")
    else:
        logger.error("✗ 核心依赖项检查失败，请安装所需的依赖项。")

    return all_passed

def test_imports():
    """测试导入自定义模块。"""
    logger.info("\n=== 测试导入自定义模块 ===")

    try:
        from code_standardizer.models import StandardCode, UserCode, StandardCodeLibrary
        logger.info("✓ 成功导入 models 模块")

        from code_standardizer.similarity import normalize_text
        logger.info("✓ 成功导入 similarity 模块")

        from code_standardizer.matcher import MatchResult
        logger.info("✓ 成功导入 matcher 模块")

        from code_standardizer.vector_store import VectorStore
        logger.info("✓ 成功导入 vector_store 模块")

        from code_standardizer.enhanced_matcher import EnhancedCodeMatcher
        logger.info("✓ 成功导入 enhanced_matcher 模块")

        from code_standardizer.api import app
        logger.info("✓ 成功导入 api 模块")

        return True
    except Exception as e:
        logger.error(f"导入自定义模块时出错: {e}")
        return False

def main():
    """主函数。"""
    logger.info("=== 代码标准化依赖项测试 ===\n")

    # 检查依赖项
    dependencies_ok = check_all_dependencies()

    # 测试导入
    imports_ok = test_imports()

    # 总结
    logger.info("\n=== 最终结果 ===")
    if dependencies_ok and imports_ok:
        logger.info("✓ 所有测试通过！可以运行 API 服务。")
        logger.info("  运行命令: python run_api.py")
        return 0
    else:
        if not dependencies_ok:
            logger.error("✗ 依赖项检查失败。请安装所需的依赖项。")
            logger.info("  安装命令: pip install -r requirements.txt")
        if not imports_ok:
            logger.error("✗ 模块导入测试失败。请检查代码是否完整。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
