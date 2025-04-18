"""
代码标准化包的匹配引擎。

本模块提供了将用户代码与标准代码库进行匹配的功能。
本实现使用顺序处理（无并行处理）并限制字段数量。
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
from .models import StandardCode, UserCode, StandardCodeLibrary
from .similarity import get_best_match, MAX_FIELDS_PER_BATCH

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MatchResult:
    """
    表示用户代码与标准库匹配的结果。

    Attributes:
        user_code: 原始用户代码
        matched_code: 匹配的标准代码（如果有）
        similarity_score: 用户代码与匹配代码之间的相似度分数
        is_match: 相似度分数是否超过阈值
        threshold: 用于匹配的阈值
    """

    def __init__(
        self,
        user_code: UserCode,
        matched_code: Optional[StandardCode],
        similarity_score: float,
        is_match: bool,
        threshold: float
    ):
        self.user_code = user_code
        self.matched_code = matched_code
        self.similarity_score = similarity_score
        self.is_match = is_match
        self.threshold = threshold

    def to_dict(self) -> Dict[str, Any]:
        """将匹配结果转换为字典。"""
        result = {
            'user_code': self.user_code.to_dict(),
            'similarity_score': self.similarity_score,
            'is_match': self.is_match,
            'threshold': self.threshold
        }

        if self.matched_code:
            result['matched_code'] = self.matched_code.to_dict()

        return result

    def __str__(self) -> str:
        """匹配结果的字符串表示。"""
        if self.is_match:
            return (f"找到匹配: 用户代码 '{self.user_code.name}' 匹配标准代码 "
                   f"'{self.matched_code.name}' 相似度为 {self.similarity_score:.2f}")
        else:
            return (f"未找到匹配: 用户代码 '{self.user_code.name}' 没有超过阈值的匹配 "
                   f"阈值 {self.threshold} (最佳: {self.similarity_score:.2f})")


class CodeMatcher:
    """
    将用户代码与标准代码库进行匹配。

    该类提供了匹配单个代码或批量代码的功能。
    本实现使用顺序处理并限制字段数量。
    """

    def __init__(
        self,
        standard_library: StandardCodeLibrary,
        threshold: float = 0.7,
        max_fields: int = MAX_FIELDS_PER_BATCH
    ):
        """
        初始化代码匹配器。

        Args:
            standard_library: 要匹配的标准代码库
            threshold: 考虑匹配的最低相似度分数
            max_fields: 单批处理的最大字段数
        """
        self.standard_library = standard_library
        self.threshold = threshold
        self.max_fields = max_fields

    def match_code(self, user_code: UserCode) -> MatchResult:
        """
        将单个用户代码与标准库进行匹配。

        Args:
            user_code: 要匹配的用户代码

        Returns:
            包含匹配信息的 MatchResult
        """
        matched_code, similarity_score, is_match = get_best_match(
            user_code,
            self.standard_library.codes,
            self.threshold,
            self.max_fields
        )

        return MatchResult(
            user_code=user_code,
            matched_code=matched_code,
            similarity_score=similarity_score,
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

        # 顺序处理每个用户代码（无并行处理）
        results = []
        for code in user_codes:
            results.append(self.match_code(code))

        return results

    def update_library(self, new_library: StandardCodeLibrary) -> None:
        """
        更新用于匹配的标准代码库。

        Args:
            new_library: 新的标准代码库
        """
        self.standard_library = new_library
        logger.info(f"已更新标准库，包含 {len(new_library)} 个代码")

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
