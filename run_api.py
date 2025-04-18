#!/usr/bin/env python3
"""
启动代码标准化 API 服务。

该脚本启动 FastAPI 应用程序，提供代码标准化 API 服务。
"""

import os
import sys
import argparse
import uvicorn
import logging
import traceback
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="启动代码标准化 API 服务")

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务器主机 (默认: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="数据目录 (默认: data)"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载 (开发模式)"
    )

    return parser.parse_args()

def main():
    """主函数。"""
    try:
        args = parse_args()

        # 设置环境变量
        os.environ["DATA_DIR"] = args.data_dir

        # 确保数据目录存在
        try:
            os.makedirs(args.data_dir, exist_ok=True)
            logger.info(f"数据目录: {os.path.abspath(args.data_dir)}")
        except Exception as e:
            logger.error(f"创建数据目录时出错: {e}")
            raise

        # 检查数据目录是否可写
        test_file = os.path.join(args.data_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info("数据目录可写")
        except Exception as e:
            logger.error(f"数据目录不可写: {e}")
            raise

        logger.info(f"启动服务器: {args.host}:{args.port}")

        # 启动服务器
        uvicorn.run(
            "code_standardizer.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"启动服务器时出错: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
