"""
代码标准化包的数据模型。

本模块定义了标准代码和用户代码的数据结构。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime


@dataclass
class StandardCode:
    """
    表示标准库中的标准返回码。

    Attributes:
        code_value: 代码的数字或字符串值
        name: 代码的标准化名称
        description: 关于代码的附加信息
        category: 可选的分组类别
        created_at: 此代码添加到标准库的时间
        updated_at: 此代码最后更新的时间
    """
    code_value: str
    name: str
    description: str
    category: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """将标准代码转换为字典。"""
        return {
            'code_value': self.code_value,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardCode':
        """从字典创建 StandardCode 实例。"""
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else None
        updated_at = datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None

        return cls(
            code_value=data['code_value'],
            name=data['name'],
            description=data['description'],
            category=data.get('category'),
            created_at=created_at,
            updated_at=updated_at
        )


@dataclass
class UserCode:
    """
    表示用户上传的返回码。

    Attributes:
        code_value: 代码的数字或字符串值
        name: 用户提供的名称
        description: 用户提供的附加信息
        additional_fields: 用户提供的任何其他字段
    """
    code_value: str
    name: str
    description: Optional[str] = None
    additional_fields: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_fields is None:
            self.additional_fields = {}

    def to_dict(self) -> Dict[str, Any]:
        """将用户代码转换为字典。"""
        result = {
            'code_value': self.code_value,
            'name': self.name,
        }

        if self.description:
            result['description'] = self.description

        if self.additional_fields:
            result.update(self.additional_fields)

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserCode':
        """从字典创建 UserCode 实例。"""
        # 提取已知字段
        code_value = data.pop('code_value')
        name = data.pop('name')
        description = data.pop('description', None)

        # 所有剩余字段都进入 additional_fields
        return cls(
            code_value=code_value,
            name=name,
            description=description,
            additional_fields=data
        )


class StandardCodeLibrary:
    """
    管理标准代码集合。

    该类提供了加载、保存和更新标准代码库的功能。
    它还处理从库中搜索和检索代码。
    """

    def __init__(self, library_path: Optional[str] = None):
        """
        初始化标准代码库。

        Args:
            library_path: 包含标准代码的 JSON 文件路径。
                          如果为 None，则创建一个空库。
        """
        self.codes: List[StandardCode] = []
        self.library_path = library_path

        if library_path and os.path.exists(library_path):
            self.load_from_file(library_path)

    def add_code(self, code: StandardCode) -> None:
        """
        向库中添加新的标准代码。

        Args:
            code: 要添加的 StandardCode
        """
        self.codes.append(code)

    def get_code_by_value(self, code_value: str) -> Optional[StandardCode]:
        """
        根据值检索标准代码。

        Args:
            code_value: 要检索的代码值

        Returns:
            如果找到则返回 StandardCode，否则返回 None
        """
        for code in self.codes:
            if code.code_value == code_value:
                return code
        return None

    def get_code_by_name(self, name: str) -> Optional[StandardCode]:
        """
        根据名称检索标准代码。

        Args:
            name: 要检索的代码名称

        Returns:
            如果找到则返回 StandardCode，否则返回 None
        """
        for code in self.codes:
            if code.name == name:
                return code
        return None

    def load_from_file(self, file_path: str) -> None:
        """
        Load standard codes from a JSON file.

        Args:
            file_path: Path to the JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.codes = [StandardCode.from_dict(item) for item in data]
            self.library_path = file_path
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load standard code library: {e}")

    def save_to_file(self, file_path: Optional[str] = None) -> None:
        """
        Save the standard codes to a JSON file.

        Args:
            file_path: Path to save the JSON file. If None, uses the path from initialization.
        """
        save_path = file_path or self.library_path
        if not save_path:
            raise ValueError("No file path specified for saving the library")

        data = [code.to_dict() for code in self.codes]

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def update_from_file(self, file_path: str) -> None:
        """
        Update the standard code library from a file.

        This method replaces the current library with the contents of the file.

        Args:
            file_path: Path to the JSON file with updated codes
        """
        self.load_from_file(file_path)

    def __len__(self) -> int:
        """Return the number of codes in the library."""
        return len(self.codes)

    def __iter__(self):
        """Iterate through the codes in the library."""
        return iter(self.codes)
