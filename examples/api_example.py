#!/usr/bin/env python3
"""
代码标准化 API 使用示例。

该脚本演示如何使用代码标准化 API。
"""

import os
import sys
import json
import requests
from pathlib import Path

# API 基础 URL
API_BASE_URL = "http://localhost:8000"

def print_section(title):
    """打印带有分隔线的标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def get_library_stats():
    """获取标准库统计信息。"""
    print_section("获取标准库统计信息")
    
    response = requests.get(f"{API_BASE_URL}/library/stats")
    data = response.json()
    
    print(f"标准库包含 {data['total_codes']} 个代码")
    print(f"最后更新时间: {data['last_updated']}")
    print(f"向量存储大小: {data['vector_store_size']}")
    print("\n类别统计:")
    for category, count in data['categories'].items():
        print(f"  - {category}: {count} 个代码")

def get_library_codes():
    """获取标准库中的代码。"""
    print_section("获取标准库中的代码")
    
    response = requests.get(f"{API_BASE_URL}/library/codes", params={"limit": 5})
    codes = response.json()
    
    print(f"获取到 {len(codes)} 个代码:")
    for code in codes:
        print(f"  - {code['code_value']}: {code['name']} ({code['category'] or '未分类'})")

def match_codes():
    """匹配用户代码与标准库。"""
    print_section("匹配用户代码与标准库")
    
    # 创建一些示例用户代码
    user_codes = [
        {
            "code_value": "200",
            "name": "成功",
            "description": "请求成功"
        },
        {
            "code_value": "404",
            "name": "资源未找到",
            "description": "请求的资源不存在"
        },
        {
            "code_value": "999",
            "name": "自定义错误",
            "description": "发生了自定义错误"
        }
    ]
    
    # 发送匹配请求
    response = requests.post(
        f"{API_BASE_URL}/match",
        json={"user_codes": user_codes, "threshold": 0.6}
    )
    data = response.json()
    
    print(f"匹配结果: 总计 {data['total']} 个代码，匹配 {data['matched']} 个")
    
    for result in data['results']:
        user_code = result['user_code']
        if result['is_match']:
            matched_code = result['matched_code']
            print(f"  - 用户代码 '{user_code['name']}' 匹配标准代码 '{matched_code['name']}' "
                  f"(相似度: {result['similarity_score']:.2f})")
        else:
            print(f"  - 用户代码 '{user_code['name']}' 没有匹配 "
                  f"(最佳相似度: {result['similarity_score']:.2f})")

def update_library():
    """更新标准库。"""
    print_section("更新标准库")
    
    # 创建一些新的标准代码
    new_codes = [
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
    
    # 发送更新请求
    response = requests.post(
        f"{API_BASE_URL}/library/update",
        json={"codes": new_codes}
    )
    data = response.json()
    
    print(f"更新结果: {data['message']}")
    print(f"总计: {data['total']} 个代码")

def main():
    """主函数。"""
    print("代码标准化 API 使用示例")
    
    # 检查 API 是否运行
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code != 200:
            print(f"错误: API 服务返回状态码 {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到 API 服务 ({API_BASE_URL})")
        print("请确保 API 服务正在运行 (python run_api.py)")
        return
    
    # 执行示例操作
    get_library_stats()
    get_library_codes()
    match_codes()
    update_library()
    
    # 等待一段时间后再次获取统计信息，以查看更新效果
    print("\n等待 5 秒钟，让后台更新完成...")
    import time
    time.sleep(5)
    
    get_library_stats()

if __name__ == "__main__":
    main()
