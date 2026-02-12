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

from src.ocr.processor import OCRProcessingError, OCRProcessor
from src.validators import (
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
