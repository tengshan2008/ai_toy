"""
增强的代码匹配器，使用预计算的向量加速匹配过程。

该模块提供了一个增强版的 CodeMatcher 类，它使用预计算的向量来加速匹配过程。
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import math

# 尝试导入 torch 和 sentence_transformers
try:
    import torch
    # 检查 PyTorch 是否可用
    torch.randn(1, 1)  # 简单测试

    try:
        from sentence_transformers import util
        TORCH_AVAILABLE = True
    except ImportError as e:
        TORCH_AVAILABLE = False
        logging.warning(f"无法导入 sentence_transformers: {e}")
except (ImportError, RuntimeError) as e:
    TORCH_AVAILABLE = False
    logging.warning(f"无法使用 PyTorch: {e}")

# 尝试导入 numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .models import StandardCode, UserCode, StandardCodeLibrary
from .matcher import MatchResult
from .similarity import get_sentence_bert_model, MAX_FIELDS_PER_BATCH
from .vector_store import VectorStore

# 配置日志
logger = logging.getLogger(__name__)

class EnhancedCodeMatcher:
    """
    增强的代码匹配器，使用预计算的向量加速匹配过程。

    该类提供了匹配单个代码或批量代码的功能，并使用预计算的向量来加速匹配过程。
    """

    def __init__(
        self,
        standard_library: StandardCodeLibrary,
        vector_store: Optional[VectorStore] = None,
        vector_store_path: Optional[str] = None,
        threshold: float = 0.7,
        max_fields: int = MAX_FIELDS_PER_BATCH
    ):
        """
        初始化增强的代码匹配器。

        Args:
            standard_library: 要匹配的标准代码库
            vector_store: 预计算的向量存储，如果为 None 则创建新的
            vector_store_path: 向量存储文件路径，如果 vector_store 为 None 则使用
            threshold: 考虑匹配的最低相似度分数
            max_fields: 单批处理的最大字段数
        """
        self.standard_library = standard_library
        self.threshold = threshold
        self.max_fields = max_fields

        # 初始化向量存储
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(vector_store_path)

        # 确保所有标准代码都有向量
        self._update_vectors()

    def _update_vectors(self) -> Tuple[int, int, int]:
        """
        更新向量存储，确保所有标准代码都有向量。

        Returns:
            元组 (添加的代码数, 更新的代码数, 跳过的代码数)
        """
        return self.vector_store.update_codes(self.standard_library.codes)

    def _calculate_similarity(self, vector1, vector2) -> float:
        """
        根据向量类型计算相似度。

        Args:
            vector1: 第一个向量
            vector2: 第二个向量

        Returns:
            相似度分数，范围从 0 到 1
        """
        # 尝试进行类型转换
        if TORCH_AVAILABLE and NUMPY_AVAILABLE:
            # 将 NumPy 数组转换为 PyTorch 张量
            if isinstance(vector1, np.ndarray) and isinstance(vector2, torch.Tensor):
                try:
                    vector1 = torch.from_numpy(vector1.astype(np.float32))
                    logger.info("将 NumPy 数组转换为 PyTorch 张量")
                except Exception as e:
                    logger.error(f"类型转换时出错: {e}")

            # 将 PyTorch 张量转换为 NumPy 数组
            elif isinstance(vector1, torch.Tensor) and isinstance(vector2, np.ndarray):
                try:
                    vector2 = torch.from_numpy(vector2.astype(np.float32))
                    logger.info("将 NumPy 数组转换为 PyTorch 张量")
                except Exception as e:
                    logger.error(f"类型转换时出错: {e}")

        # 处理 PyTorch 张量
        if TORCH_AVAILABLE and isinstance(vector1, torch.Tensor) and isinstance(vector2, torch.Tensor):
            try:
                # 检查向量维度
                if vector1.shape[-1] != vector2.shape[-1]:
                    logger.warning(f"向量维度不匹配: {vector1.shape} 和 {vector2.shape}。使用备选方案。")
                    # 如果维度不匹配，使用字典相似度
                    return 0.1

                # 使用 PyTorch 的余弦相似度
                return util.pytorch_cos_sim(vector1, vector2).item()
            except Exception as e:
                logger.error(f"计算 PyTorch 张量相似度时出错: {e}")

        # 处理 NumPy 数组
        if NUMPY_AVAILABLE and isinstance(vector1, np.ndarray) and isinstance(vector2, np.ndarray):
            try:
                # 计算余弦相似度
                dot_product = np.dot(vector1, vector2)
                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)

                if norm1 == 0 or norm2 == 0:
                    return 0.0

                return dot_product / (norm1 * norm2)
            except Exception as e:
                logger.error(f"计算 NumPy 数组相似度时出错: {e}")

        # 处理字典
        if isinstance(vector1, dict) and isinstance(vector2, dict):
            try:
                # 计算字典的相似度
                # 使用共同键的值进行比较
                common_keys = set(vector1.keys()) & set(vector2.keys())

                if not common_keys:
                    return 0.0

                # 计算字符串字段的相似度
                similarity_sum = 0.0
                count = 0

                for key in common_keys:
                    v1 = vector1[key]
                    v2 = vector2[key]

                    # 字符串字段使用简单的相等比较
                    if isinstance(v1, str) and isinstance(v2, str):
                        if v1 == v2:
                            similarity_sum += 1.0
                        else:
                            # 计算字符串的相似度
                            common_len = 0
                            for c1, c2 in zip(v1, v2):
                                if c1 == c2:
                                    common_len += 1
                            max_len = max(len(v1), len(v2))
                            if max_len > 0:
                                similarity_sum += common_len / max_len
                            else:
                                similarity_sum += 1.0  # 两个空字符串视为相等
                    # 数值字段使用差异比较
                    elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        # 计算数值的相似度
                        max_val = max(abs(v1), abs(v2))
                        if max_val > 0:
                            diff = abs(v1 - v2) / max_val
                            similarity_sum += 1.0 - min(diff, 1.0)  # 差异越小，相似度越高
                        else:
                            similarity_sum += 1.0  # 两个零视为相等

                    count += 1

                if count > 0:
                    return similarity_sum / count
                return 0.0
            except Exception as e:
                logger.error(f"计算字典相似度时出错: {e}")

        # 尝试将 PyTorch 张量转换为 NumPy 数组
        if TORCH_AVAILABLE and NUMPY_AVAILABLE:
            if isinstance(vector1, torch.Tensor) and not isinstance(vector2, torch.Tensor):
                try:
                    # 将 PyTorch 张量转换为 NumPy 数组
                    vector1_np = vector1.detach().cpu().numpy()

                    # 如果 vector2 是 NumPy 数组，使用 NumPy 计算相似度
                    if isinstance(vector2, np.ndarray):
                        dot_product = np.dot(vector1_np, vector2)
                        norm1 = np.linalg.norm(vector1_np)
                        norm2 = np.linalg.norm(vector2)

                        if norm1 == 0 or norm2 == 0:
                            return 0.0

                        return dot_product / (norm1 * norm2)
                except Exception as e:
                    logger.error(f"将 PyTorch 张量转换为 NumPy 数组时出错: {e}")

            elif isinstance(vector2, torch.Tensor) and not isinstance(vector1, torch.Tensor):
                try:
                    # 将 PyTorch 张量转换为 NumPy 数组
                    vector2_np = vector2.detach().cpu().numpy()

                    # 如果 vector1 是 NumPy 数组，使用 NumPy 计算相似度
                    if isinstance(vector1, np.ndarray):
                        dot_product = np.dot(vector1, vector2_np)
                        norm1 = np.linalg.norm(vector1)
                        norm2 = np.linalg.norm(vector2_np)

                        if norm1 == 0 or norm2 == 0:
                            return 0.0

                        return dot_product / (norm1 * norm2)
                except Exception as e:
                    logger.error(f"将 PyTorch 张量转换为 NumPy 数组时出错: {e}")

        # 如果向量类型不匹配或计算出错，返回低相似度
        logger.warning(f"无法计算不同类型向量的相似度: {type(vector1)} 和 {type(vector2)}")
        return 0.1  # 返回一个小的非零值，以便于调试

    def _compute_user_code_vector(self, user_code: UserCode):
        """
        计算用户代码的向量表示。

        Args:
            user_code: 用户代码

        Returns:
            用户代码的向量表示（PyTorch 张量、NumPy 数组或字典）
        """
        # 组合名称和描述以获得更好的表示
        text = f"{user_code.name} {user_code.description or ''}"

        # 尝试使用 sentence-BERT 模型
        if TORCH_AVAILABLE:
            try:
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
            'code_value': user_code.code_value,
            'name_length': len(user_code.name),
            'desc_length': len(user_code.description or ''),
            'name_words': len(user_code.name.split()),
            'desc_words': len(user_code.description.split()) if user_code.description else 0,
            'name_first_char': user_code.name[0].lower() if user_code.name else '',
        }

        logger.warning("使用基本特征字典作为向量表示")
        return feature_dict

    def match_code(self, user_code: UserCode) -> MatchResult:
        """
        将单个用户代码与标准库进行匹配。

        Args:
            user_code: 要匹配的用户代码

        Returns:
            包含匹配信息的 MatchResult
        """
        try:
            # 计算用户代码的向量
            user_vector = self._compute_user_code_vector(user_code)

            # 获取所有标准代码向量
            standard_vectors = self.vector_store.get_all_vectors()

            # 限制要处理的字段数量
            code_values = list(standard_vectors.keys())
            if len(code_values) > self.max_fields:
                logger.warning(f"标准代码数量 ({len(code_values)}) 超过最大限制 ({self.max_fields})。"
                              f"只处理前 {self.max_fields} 个代码。")
                code_values = code_values[:self.max_fields]

            best_match = None
            best_score = 0.0

            # 计算与所有标准代码的相似度
            for code_value in code_values:
                std_vector = standard_vectors[code_value]

                # 根据向量类型计算相似度
                similarity = self._calculate_similarity(user_vector, std_vector)

                # 更新最佳匹配
                if similarity > best_score:
                    best_score = similarity
                    # 获取对应的标准代码
                    best_match = self.standard_library.get_code_by_value(code_value)

            # 检查是否超过阈值
            is_match = best_score >= self.threshold

        except Exception as e:
            logger.error(f"匹配代码时出错: {e}")
            # 出错时返回无匹配结果
            best_match = None
            best_score = 0.0
            is_match = False

        return MatchResult(
            user_code=user_code,
            matched_code=best_match,
            similarity_score=best_score,
            is_match=is_match,
            threshold=self.threshold
        )

    def match_codes_batch(self, user_codes: List[UserCode]) -> List[MatchResult]:
        """
        顺序将多个用户代码与标准库进行匹配。

        Args:
            user_codes: 要匹配的用户代码列表

        Returns:
            MatchResult 对象列表
        """
        if not user_codes:
            return []

        # 限制要处理的用户代码数量
        if len(user_codes) > self.max_fields:
            logger.warning(f"用户代码数量 ({len(user_codes)}) 超过最大限制 ({self.max_fields})。"
                          f"只处理前 {self.max_fields} 个代码。")
            user_codes = user_codes[:self.max_fields]

        # 顺序处理每个用户代码
        results = []
        for code in user_codes:
            results.append(self.match_code(code))

        return results

    def update_library(self, new_library: StandardCodeLibrary) -> Tuple[int, int, int]:
        """
        更新用于匹配的标准代码库。

        Args:
            new_library: 新的标准代码库

        Returns:
            元组 (添加的代码数, 更新的代码数, 跳过的代码数)
        """
        self.standard_library = new_library

        # 更新向量存储
        added, updated, skipped = self._update_vectors()

        logger.info(f"已更新标准库，包含 {len(new_library)} 个代码 "
                   f"(添加: {added}, 更新: {updated}, 跳过: {skipped})")

        return added, updated, skipped

    def set_threshold(self, threshold: float) -> None:
        """
        设置匹配的相似度阈值。

        Args:
            threshold: 新的阈值，范围在 0 到 1 之间
        """
        if not 0 <= threshold <= 1:
            raise ValueError("阈值必须在 0 和 1 之间")

        self.threshold = threshold
        logger.info(f"已更新相似度阈值为 {threshold}")
