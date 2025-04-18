# 代码标准化 API

本文档介绍如何使用代码标准化 API 服务，该服务提供了字段匹配推荐和标准码库更新功能。

## 功能特点

- 使用 sentence-BERT 进行语义匹配
- 支持标准码库的增量更新
- 预计算向量以提高匹配效率
- RESTful API 接口
- 支持批量处理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动 API 服务

```bash
python run_api.py --host 0.0.0.0 --port 8000 --data-dir data
```

参数说明：
- `--host`: 服务器主机地址（默认：0.0.0.0）
- `--port`: 服务器端口（默认：8000）
- `--data-dir`: 数据目录（默认：data）
- `--reload`: 启用热重载（开发模式）

## API 接口

### 1. 匹配用户代码

**请求**：
```
POST /match
```

**请求体**：
```json
{
  "user_codes": [
    {
      "code_value": "200",
      "name": "成功",
      "description": "请求成功"
    },
    {
      "code_value": "404",
      "name": "资源未找到",
      "description": "请求的资源不存在"
    }
  ],
  "threshold": 0.7
}
```

**响应**：
```json
{
  "results": [
    {
      "user_code": {
        "code_value": "200",
        "name": "成功",
        "description": "请求成功"
      },
      "matched_code": {
        "code_value": "200",
        "name": "OK",
        "description": "Request was successful",
        "category": "Success"
      },
      "similarity_score": 0.85,
      "is_match": true,
      "threshold": 0.7
    },
    {
      "user_code": {
        "code_value": "404",
        "name": "资源未找到",
        "description": "请求的资源不存在"
      },
      "matched_code": {
        "code_value": "404",
        "name": "Not Found",
        "description": "The requested resource could not be found",
        "category": "Client Error"
      },
      "similarity_score": 0.78,
      "is_match": true,
      "threshold": 0.7
    }
  ],
  "total": 2,
  "matched": 2
}
```

### 2. 更新标准库

**请求**：
```
POST /library/update
```

**请求体**：
```json
{
  "codes": [
    {
      "code_value": "1001",
      "name": "数据库连接错误",
      "description": "无法连接到数据库",
      "category": "系统错误"
    },
    {
      "code_value": "1002",
      "name": "认证失败",
      "description": "用户认证失败",
      "category": "安全错误"
    }
  ]
}
```

**响应**：
```json
{
  "total": 2,
  "added": 0,
  "updated": 0,
  "skipped": 0,
  "message": "标准库更新已开始，将在后台处理"
}
```

### 3. 获取标准库统计信息

**请求**：
```
GET /library/stats
```

**响应**：
```json
{
  "total_codes": 16,
  "last_updated": "2023-05-20T15:30:45.123456",
  "vector_store_size": 16,
  "categories": {
    "Success": 4,
    "Client Error": 6,
    "Server Error": 4,
    "系统错误": 1,
    "安全错误": 1
  }
}
```

### 4. 获取标准库中的代码

**请求**：
```
GET /library/codes?category=Success&limit=10&offset=0
```

**响应**：
```json
[
  {
    "code_value": "200",
    "name": "OK",
    "description": "Request was successful",
    "category": "Success"
  },
  {
    "code_value": "201",
    "name": "Created",
    "description": "Resource was successfully created",
    "category": "Success"
  }
]
```

## 示例代码

查看 `examples/api_example.py` 文件，了解如何使用 API：

```bash
python examples/api_example.py
```

## 增量更新说明

标准码库的增量更新是通过以下方式实现的：

1. 使用 `VectorStore` 类存储预计算的向量
2. 当添加新的标准代码时，只计算新代码的向量
3. 当更新现有代码时，只重新计算已更改的代码的向量
4. 未更改的代码保持原有向量，不需要重新计算

这种方式大大提高了更新效率，特别是对于大型标准码库。

## 性能优化

1. 使用预计算的向量加速匹配过程
2. 限制每批处理的字段数量
3. 使用较小的 BERT 模型（paraphrase-MiniLM-L6-v2）以提高速度
4. 延迟加载 BERT 模型，仅在需要时加载

## 故障排除

如果遇到问题，请检查：

1. 确保已安装所有依赖
   ```bash
   python test_installation.py  # 测试环境
   python install_dependencies.py  # 安装依赖项
   ```

2. 确保 API 服务正在运行
   ```bash
   python run_api.py
   ```

3. 检查日志输出以获取详细错误信息

4. 确保数据目录可写入

### 常见问题

#### 1. PyTorch 或 torchvision 错误

如果遇到 `operator torchvision::nms does not exist` 错误，这通常是由于 PyTorch 和 torchvision 版本不匹配导致的。代码已经进行了优化，可以处理这种情况，但如果仍然出现问题，请尝试：

```bash
# 卸载现有版本
pip uninstall -y torch torchvision

# 安装兼容的版本
pip install torch==2.0.0 torchvision==0.15.0
```

#### 2. sentence-transformers 错误

如果遇到与 sentence-transformers 相关的错误，请尝试：

```bash
pip install -U sentence-transformers
```

如果仍然出现问题，代码将自动回退到基本的相似度计算方法。

#### 3. 内存错误

如果遇到内存不足错误，请尝试减小 `MAX_FIELDS_PER_BATCH` 值：

```bash
# 在启动 API 时指定较小的批处理大小
python run_api.py --max-fields 50
```

或者编辑 `code_standardizer/similarity.py` 文件，将 `MAX_FIELDS_PER_BATCH` 值从 100 减小到 50 或更小。
