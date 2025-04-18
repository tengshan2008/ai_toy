"""
向量存储模块，用于管理标准代码的预计算向量。

该模块提供了存储和检索标准代码向量的功能，支持增量更新。
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path

# 尝试导入 torch
try:
    import torch
    # 检查 PyTorch 是否可用
    torch.randn(1, 1)  # 简单测试
    TORCH_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TORCH_AVAILABLE = False
    logging.warning(f"无法使用 PyTorch，将使用备选方案: {e}")

# 尝试导入 numpy 作为备选方案
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("无法导入 NumPy，将使用基本字典存储")

from .models import StandardCode
from .similarity import get_sentence_bert_model

# 配置日志
logger = logging.getLogger(__name__)

class VectorStore:
    """
    标准代码向量存储。

    该类管理标准代码的预计算向量，支持增量更新。
    """

    def __init__(self, store_path: Optional[str] = None):
        """
        初始化向量存储。

        Args:
            store_path: 向量存储文件路径。如果为 None，则不会加载或保存向量。
        """
        self.vectors: Dict[str, Any] = {}  # code_value -> 向量（PyTorch 张量、NumPy 数组或字典）
        self.metadata: Dict[str, Dict] = {}  # code_value -> 元数据
        self.store_path = store_path
        self.last_updated = datetime.now()

        # 如果存储路径存在，则加载向量
        if store_path and os.path.exists(store_path):
            self.load_vectors(store_path)

    def compute_vector(self, code: StandardCode):
        """
        计算标准代码的向量表示。

        Args:
            code: 要计算向量的标准代码

        Returns:
            代码的向量表示（PyTorch 张量、NumPy 数组或字典）
        """
        # 组合名称和描述以获得更好的表示
        text = f"{code.name} {code.description}"

        # 尝试使用 sentence-BERT 模型
        try:
            if TORCH_AVAILABLE:
                model = get_sentence_bert_model()
                if model is not None:
                    # 计算向量
                    return model.encode(text, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"使用 sentence-BERT 计算向量时出错: {e}")

        # 备选方案 1: 使用 NumPy 和简单的词袋模型
        if NUMPY_AVAILABLE:
            try:
                # 简单的词袋表示
                words = set(text.lower().split())
                # 创建一个简单的哈希向量
                vector = np.zeros(100)  # 使用 100 维向量
                for word in words:
                    # 使用单词的哈希值来设置向量元素
                    hash_val = hash(word) % 100
                    vector[hash_val] = 1.0
                return vector
            except Exception as e:
                logger.error(f"使用 NumPy 备选方案计算向量时出错: {e}")

        # 备选方案 2: 使用字典
        # 创建一个简单的特征字典
        feature_dict = {
            'code_value': code.code_value,
            'name_length': len(code.name),
            'desc_length': len(code.description),
            'name_words': len(code.name.split()),
            'desc_words': len(code.description.split()),
            'name_first_char': code.name[0].lower() if code.name else '',
            'category': code.category or ''
        }

        logger.warning("使用基本特征字典作为向量表示")
        return feature_dict

    def add_code(self, code: StandardCode) -> None:
        """
        添加一个标准代码到向量存储。

        Args:
            code: 要添加的标准代码
        """
        # 计算向量
        vector = self.compute_vector(code)

        # 存储向量和元数据
        self.vectors[code.code_value] = vector
        self.metadata[code.code_value] = {
            'name': code.name,
            'description': code.description,
            'category': code.category,
            'updated_at': datetime.now().isoformat()
        }

        self.last_updated = datetime.now()

    def update_codes(self, codes: List[StandardCode]) -> Tuple[int, int, int]:
        """
        更新多个标准代码，仅计算新的或已更改的代码的向量。

        Args:
            codes: 要更新的标准代码列表

        Returns:
            元组 (添加的代码数, 更新的代码数, 跳过的代码数)
        """
        added = 0
        updated = 0
        skipped = 0

        for code in codes:
            # 检查代码是否已存在
            if code.code_value in self.metadata:
                # 检查代码是否已更改
                current_metadata = self.metadata[code.code_value]
                if (current_metadata['name'] == code.name and
                    current_metadata['description'] == code.description and
                    current_metadata['category'] == code.category):
                    # 代码未更改，跳过
                    skipped += 1
                    continue

                # 代码已更改，更新向量
                self.add_code(code)
                updated += 1
            else:
                # 新代码，添加向量
                self.add_code(code)
                added += 1

        # 如果有更改，保存向量
        if added > 0 or updated > 0:
            if self.store_path:
                self.save_vectors(self.store_path)

        return added, updated, skipped

    def remove_codes(self, code_values: List[str]) -> int:
        """
        从向量存储中移除代码。

        Args:
            code_values: 要移除的代码值列表

        Returns:
            移除的代码数
        """
        removed = 0

        for code_value in code_values:
            if code_value in self.vectors:
                del self.vectors[code_value]
                del self.metadata[code_value]
                removed += 1

        # 如果有移除，保存向量
        if removed > 0 and self.store_path:
            self.save_vectors(self.store_path)

        return removed

    def get_all_vectors(self) -> Dict[str, Any]:
        """
        获取所有代码向量。

        Returns:
            代码值到向量的映射（可能是 PyTorch 张量、NumPy 数组或字典）
        """
        return self.vectors

    def get_vector(self, code_value: str) -> Optional[Any]:
        """
        获取特定代码的向量。

        Args:
            code_value: 代码值

        Returns:
            代码的向量（可能是 PyTorch 张量、NumPy 数组或字典），如果不存在则返回 None
        """
        return self.vectors.get(code_value)

    def get_code_values(self) -> Set[str]:
        """
        获取所有代码值。

        Returns:
            代码值集合
        """
        return set(self.vectors.keys())

    def save_vectors(self, file_path: str) -> None:
        """
        保存向量到文件。

        Args:
            file_path: 保存向量的文件路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # 处理不同类型的向量
        processed_vectors = {}
        for k, v in self.vectors.items():
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                # 将 PyTorch 张量转换为 CPU 张量
                processed_vectors[k] = v.cpu()
            elif NUMPY_AVAILABLE and isinstance(v, np.ndarray):
                # NumPy 数组可以直接序列化
                processed_vectors[k] = v
            else:
                # 其他类型（如字典）直接保存
                processed_vectors[k] = v

        data = {
            'vectors': processed_vectors,
            'metadata': self.metadata,
            'last_updated': self.last_updated.isoformat()
        }

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"向量已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存向量时出错: {e}")

    def load_vectors(self, file_path: str) -> None:
        """
        从文件加载向量。

        Args:
            file_path: 向量文件路径
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # 加载向量和元数据
            loaded_vectors = data['vectors']
            self.metadata = data['metadata']
            self.last_updated = datetime.fromisoformat(data['last_updated'])

            # 处理不同类型的向量
            self.vectors = {}
            for k, v in loaded_vectors.items():
                # 将向量移动到适当的设备
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    # 如果可能，使用 GPU
                    if torch.cuda.is_available():
                        self.vectors[k] = v.cuda()
                    else:
                        self.vectors[k] = v
                else:
                    # 其他类型直接保存
                    self.vectors[k] = v

            logger.info(f"从 {file_path} 加载了 {len(self.vectors)} 个向量")
        except (FileNotFoundError, pickle.UnpicklingError, KeyError) as e:
            logger.error(f"加载向量时出错: {e}")
            # 初始化为空
            self.vectors = {}
            self.metadata = {}
            self.last_updated = datetime.now()
        except Exception as e:
            logger.error(f"加载向量时发生未知错误: {e}")
            # 初始化为空
            self.vectors = {}
            self.metadata = {}
            self.last_updated = datetime.now()

    def __len__(self) -> int:
        """返回存储中的向量数量。"""
        return len(self.vectors)
