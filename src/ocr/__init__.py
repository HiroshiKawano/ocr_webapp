"""OCR処理パッケージ

PaddleOCR を使用した画像からのテキスト抽出機能を提供する。
FileValidator によるセキュアな入力検証を統合している。

使用例:
    from src.ocr import OCRProcessor, OCRProcessingError

    processor = OCRProcessor()
    text = processor.extract_text("/path/to/image.png")
    detailed = processor.extract_text("/path/to/image.png", detailed=True)

    # カスタムvalidatorを使用する場合:
    from src.validators import FileValidator
    validator = FileValidator(allowed_base_dir="/app/uploads")
    processor = OCRProcessor(validator=validator)
"""

from .processor import OCRProcessingError, OCRProcessor

# validators モジュールから例外クラスを再エクスポート（後方互換性のため）
# 注: 相対インポートを使用してStreamlit Cloud互換性を確保
from ..validators import (
    FileFormatError,
    FileSizeError,
    ImageDimensionsError,
    PathTraversalError,
    ValidationError,
)

__all__ = [
    "OCRProcessor",
    "OCRProcessingError",
    "ValidationError",
    "PathTraversalError",
    "FileSizeError",
    "FileFormatError",
    "ImageDimensionsError",
]

