"""ファイル検証パッケージ

セキュアなファイル検証機能を提供する。
パストラバーサル防御、MIMEタイプ検証、画像解像度制限を含む。

使用例::

    from src.validators import FileValidator

    validator = FileValidator(
        allowed_base_dir="/uploads",
        max_file_size=10 * 1024 * 1024,
    )
    result = validator.validate("/uploads/image.png")
    if result.is_valid:
        print(result.file_info)
"""

from src.validators.file_validator import (
    FileFormatError,
    FileSizeError,
    FileValidator,
    ImageDimensionsError,
    PathTraversalError,
    ValidationError,
    ValidationResult,
)

__all__ = [
    "FileFormatError",
    "FileSizeError",
    "FileValidator",
    "ImageDimensionsError",
    "PathTraversalError",
    "ValidationError",
    "ValidationResult",
]
