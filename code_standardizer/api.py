"""
代码标准化 API 服务。

该模块提供了一个 FastAPI 应用程序，用于标准代码匹配和标准库管理。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import traceback

from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .models import StandardCode, UserCode, StandardCodeLibrary
from .enhanced_matcher import EnhancedCodeMatcher
from .vector_store import VectorStore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建应用
app = FastAPI(
    title="代码标准化 API",
    description="提供标准代码匹配和标准库管理的 API",
    version="1.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常。"""
    error_id = datetime.now().strftime("%Y%m%d%H%M%S")
    error_msg = f"发生错误 (ID: {error_id}): {str(exc)}"

    # 记录详细错误信息
    logger.error(f"{error_msg}\n{traceback.format_exc()}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "message": error_msg,
            "error_id": error_id
        }
    )

# 数据目录
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 标准库文件路径
LIBRARY_PATH = os.path.join(DATA_DIR, "standard_library.json")

# 向量存储文件路径
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")

# 全局变量
matcher = None
library = None
vector_store = None

# 数据模型
class StandardCodeModel(BaseModel):
    """标准代码模型。"""
    code_value: str
    name: str
    description: str
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserCodeModel(BaseModel):
    """用户代码模型。"""
    code_value: str
    name: str
    description: Optional[str] = None
    additional_fields: Optional[Dict[str, Any]] = None

class MatchRequest(BaseModel):
    """匹配请求模型。"""
    user_codes: List[UserCodeModel]
    threshold: Optional[float] = 0.7

class MatchResult(BaseModel):
    """匹配结果模型。"""
    user_code: UserCodeModel
    matched_code: Optional[StandardCodeModel] = None
    similarity_score: float
    is_match: bool
    threshold: float

class MatchResponse(BaseModel):
    """匹配响应模型。"""
    results: List[MatchResult]
    total: int
    matched: int

class LibraryUpdateRequest(BaseModel):
    """库更新请求模型。"""
    codes: List[StandardCodeModel]

class LibraryUpdateResponse(BaseModel):
    """库更新响应模型。"""
    total: int
    added: int
    updated: int
    skipped: int
    message: str

class LibraryStatsResponse(BaseModel):
    """库统计响应模型。"""
    total_codes: int
    last_updated: datetime
    vector_store_size: int
    categories: Dict[str, int]

# 辅助函数
def initialize():
    """初始化全局变量。"""
    global matcher, library, vector_store

    try:
        # 加载标准库
        library = StandardCodeLibrary(LIBRARY_PATH if os.path.exists(LIBRARY_PATH) else None)

        # 如果标准库为空，创建示例库
        if len(library.codes) == 0:
            try:
                from .utils import create_sample_standard_library
                library = create_sample_standard_library(LIBRARY_PATH)
                logger.info("创建了示例标准库")
            except Exception as e:
                logger.error(f"创建示例标准库时出错: {e}")
                # 如果创建示例库失败，使用空库
                library = StandardCodeLibrary()

        # 初始化向量存储
        try:
            vector_store = VectorStore(VECTOR_STORE_PATH)
        except Exception as e:
            logger.error(f"初始化向量存储时出错: {e}")
            # 如果加载向量存储失败，创建新的
            vector_store = VectorStore()

        # 初始化匹配器
        try:
            matcher = EnhancedCodeMatcher(
                standard_library=library,
                vector_store=vector_store,
                threshold=0.7
            )
        except Exception as e:
            logger.error(f"初始化匹配器时出错: {e}")
            # 如果初始化匹配器失败，创建一个基本的匹配器
            from .matcher import CodeMatcher
            matcher = CodeMatcher(library, threshold=0.7)
            logger.warning("使用基本匹配器代替增强匹配器")

        logger.info(f"初始化完成，标准库包含 {len(library.codes)} 个代码，"
                   f"向量存储包含 {len(vector_store)} 个向量")

    except Exception as e:
        logger.error(f"初始化时发生未预期的错误: {e}\n{traceback.format_exc()}")
        # 创建空对象，以避免 None 引用
        library = StandardCodeLibrary()
        vector_store = VectorStore()
        from .matcher import CodeMatcher
        matcher = CodeMatcher(library, threshold=0.7)

def convert_standard_code(code: StandardCode) -> StandardCodeModel:
    """将 StandardCode 转换为 StandardCodeModel。"""
    return StandardCodeModel(
        code_value=code.code_value,
        name=code.name,
        description=code.description,
        category=code.category,
        created_at=code.created_at,
        updated_at=code.updated_at
    )

def convert_user_code(code: UserCode) -> UserCodeModel:
    """将 UserCode 转换为 UserCodeModel。"""
    return UserCodeModel(
        code_value=code.code_value,
        name=code.name,
        description=code.description,
        additional_fields=code.additional_fields
    )

def convert_match_result(result) -> MatchResult:
    """将匹配结果转换为 MatchResult 模型。"""
    return MatchResult(
        user_code=convert_user_code(result.user_code),
        matched_code=convert_standard_code(result.matched_code) if result.matched_code else None,
        similarity_score=result.similarity_score,
        is_match=result.is_match,
        threshold=result.threshold
    )

def update_library_in_background(codes: List[StandardCodeModel]):
    """在后台更新标准库。"""
    global matcher, library

    # 转换为 StandardCode 对象
    standard_codes = []
    for code_model in codes:
        code = StandardCode(
            code_value=code_model.code_value,
            name=code_model.name,
            description=code_model.description,
            category=code_model.category,
            created_at=code_model.created_at or datetime.now(),
            updated_at=datetime.now()
        )
        standard_codes.append(code)

    # 更新库
    for code in standard_codes:
        library.add_code(code)

    # 保存库
    library.save_to_file(LIBRARY_PATH)

    # 更新匹配器
    added, updated, skipped = matcher.update_library(library)

    logger.info(f"后台更新完成，添加: {added}, 更新: {updated}, 跳过: {skipped}")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行。"""
    initialize()

# API 路由
@app.get("/")
async def root():
    """API 根路径。"""
    return {"message": "代码标准化 API 服务正在运行"}

@app.post("/match", response_model=MatchResponse)
async def match_codes(request: MatchRequest):
    """
    匹配用户代码与标准库。

    Args:
        request: 包含用户代码和阈值的请求

    Returns:
        匹配结果
    """
    global matcher

    if matcher is None:
        initialize()

    # 设置阈值
    if request.threshold is not None:
        matcher.set_threshold(request.threshold)

    # 转换用户代码
    user_codes = []
    for code_model in request.user_codes:
        code = UserCode(
            code_value=code_model.code_value,
            name=code_model.name,
            description=code_model.description,
            additional_fields=code_model.additional_fields or {}
        )
        user_codes.append(code)

    # 匹配代码
    results = matcher.match_codes_batch(user_codes)

    # 转换结果
    match_results = [convert_match_result(result) for result in results]

    # 统计匹配数量
    matched_count = sum(1 for result in match_results if result.is_match)

    return MatchResponse(
        results=match_results,
        total=len(match_results),
        matched=matched_count
    )

@app.post("/library/update", response_model=LibraryUpdateResponse)
async def update_library(
    request: LibraryUpdateRequest,
    background_tasks: BackgroundTasks
):
    """
    更新标准库。

    Args:
        request: 包含要更新的标准代码的请求
        background_tasks: 后台任务

    Returns:
        更新结果
    """
    global matcher, library

    if matcher is None or library is None:
        initialize()

    # 在后台更新库
    background_tasks.add_task(update_library_in_background, request.codes)

    return LibraryUpdateResponse(
        total=len(request.codes),
        added=0,  # 这些值将在后台更新
        updated=0,
        skipped=0,
        message="标准库更新已开始，将在后台处理"
    )

@app.get("/library/stats", response_model=LibraryStatsResponse)
async def get_library_stats():
    """
    获取标准库统计信息。

    Returns:
        标准库统计信息
    """
    global library, vector_store

    if library is None or vector_store is None:
        initialize()

    # 统计类别
    categories = {}
    for code in library.codes:
        category = code.category or "未分类"
        categories[category] = categories.get(category, 0) + 1

    return LibraryStatsResponse(
        total_codes=len(library.codes),
        last_updated=vector_store.last_updated,
        vector_store_size=len(vector_store),
        categories=categories
    )

@app.get("/library/codes", response_model=List[StandardCodeModel])
async def get_library_codes(
    category: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    获取标准库中的代码。

    Args:
        category: 可选的类别过滤器
        limit: 返回的最大代码数
        offset: 分页偏移量

    Returns:
        标准代码列表
    """
    global library

    if library is None:
        initialize()

    # 过滤代码
    filtered_codes = library.codes
    if category:
        filtered_codes = [code for code in filtered_codes if code.category == category]

    # 分页
    paginated_codes = filtered_codes[offset:offset + limit]

    # 转换为模型
    return [convert_standard_code(code) for code in paginated_codes]
