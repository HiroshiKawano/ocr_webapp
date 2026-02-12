"""OCRProcessor + FileValidator 統合テスト

FileValidatorをOCRProcessorに統合した後の動作を検証する。
以下のシナリオをカバー:
  - 検証成功 → OCR処理成功
  - 検証失敗 → 適切な例外スロー（ValidationError系）
  - カスタムvalidator設定
  - 後方互換性（validator=None）
  - 入力形式ごとの統合動作（ファイルパス、バイナリ、PIL Image）
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from src.ocr.processor import OCRProcessor, OCRProcessingError
from src.validators.file_validator import (
    FileFormatError,
    FileSizeError,
    FileValidator,
    ImageDimensionsError,
    PathTraversalError,
    ValidationError,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------


def _create_test_image(
    width: int = 100,
    height: int = 100,
    fmt: str = "PNG",
    mode: str = "RGB",
) -> bytes:
    """テスト用の画像バイナリを生成する"""
    image = Image.new(mode, (width, height), color="red")
    buffer = io.BytesIO()
    save_image = image
    if mode not in ("RGB", "RGBA", "L", "P") and fmt != "BMP":
        save_image = image.convert("RGB")
    if fmt == "BMP" and mode not in ("RGB", "L", "P"):
        save_image = image.convert("RGB")
    save_image.save(buffer, format=fmt)
    return buffer.getvalue()


def _create_temp_image_file(
    tmp_path: Path,
    filename: str = "test.png",
    width: int = 100,
    height: int = 100,
    fmt: str = "PNG",
) -> Path:
    """一時ディレクトリにテスト用画像ファイルを作成する"""
    file_path = tmp_path / filename
    data = _create_test_image(width=width, height=height, fmt=fmt)
    file_path.write_bytes(data)
    return file_path


# ---------------------------------------------------------------------------
# PaddleOCR の典型的な戻り値
# ---------------------------------------------------------------------------

MOCK_OCR_RESULT = [
    [
        [
            [[100, 50], [300, 50], [300, 80], [100, 80]],
            ("テスト文字列", 0.95),
        ],
    ]
]

MOCK_OCR_EMPTY_RESULT = [None]


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """テスト用ベースディレクトリ"""
    base = tmp_path / "uploads"
    base.mkdir()
    return base


@pytest.fixture
def processor_with_validator(base_dir: Path):
    """FileValidator統合済みのOCRProcessorインスタンスを返す"""
    validator = FileValidator(
        allowed_base_dir=base_dir,
        max_file_size=10 * 1024 * 1024,
        max_dimensions=(10000, 10000),
    )
    with patch("src.ocr.processor.PaddleOCR") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        proc = OCRProcessor(lang="japan", validator=validator)
        yield proc


@pytest.fixture
def processor_default():
    """デフォルト設定のOCRProcessorインスタンスを返す（validator=None相当）"""
    with patch("src.ocr.processor.PaddleOCR") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        proc = OCRProcessor(lang="japan")
        yield proc


@pytest.fixture
def processor_strict():
    """厳格な設定のOCRProcessorインスタンスを返す"""
    strict_validator = FileValidator(
        max_file_size=1 * 1024 * 1024,  # 1MB
        allowed_extensions={".png"},  # PNGのみ
        max_dimensions=(500, 500),
        allowed_base_dir=None,
    )
    with patch("src.ocr.processor.PaddleOCR") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        proc = OCRProcessor(lang="japan", validator=strict_validator)
        yield proc


# ===========================================================================
# 検証成功ケース
# ===========================================================================


class TestValidationSuccessIntegration:
    """検証を通過した場合にOCR処理が正常に完了するテスト"""

    def test_正常なファイルパスでOCR処理成功(
        self, processor_with_validator, base_dir
    ):
        """検証を通過した正常なファイルでOCR処理が成功する"""
        img_file = _create_temp_image_file(base_dir, "valid.png")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_正常なファイルパスで詳細モード成功(
        self, processor_with_validator, base_dir
    ):
        """検証を通過した正常なファイルで詳細モードが成功する"""
        img_file = _create_temp_image_file(base_dir, "valid.png")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(
            str(img_file), detailed=True
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "テスト文字列"
        assert result[0]["confidence"] == pytest.approx(0.95)

    def test_JPEGファイルでOCR処理成功(
        self, processor_with_validator, base_dir
    ):
        """JPEG形式のファイルでOCR処理が成功する"""
        img_file = _create_temp_image_file(
            base_dir, "valid.jpg", fmt="JPEG"
        )
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert "テスト文字列" in result

    def test_BMPファイルでOCR処理成功(
        self, processor_with_validator, base_dir
    ):
        """BMP形式のファイルでOCR処理が成功する"""
        img_file = _create_temp_image_file(
            base_dir, "valid.bmp", fmt="BMP"
        )
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert "テスト文字列" in result

    def test_サブディレクトリ内のファイルでOCR処理成功(
        self, processor_with_validator, base_dir
    ):
        """ベースディレクトリ配下のサブディレクトリでOCR処理が成功する"""
        sub_dir = base_dir / "2024" / "01"
        sub_dir.mkdir(parents=True)
        img_file = _create_temp_image_file(sub_dir, "photo.png")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert "テスト文字列" in result


# ===========================================================================
# 検証失敗ケース
# ===========================================================================


class TestValidationFailureIntegration:
    """検証に失敗した場合に適切な例外がスローされるテスト"""

    def test_パストラバーサル攻撃で例外(
        self, processor_with_validator, base_dir
    ):
        """パストラバーサル攻撃を検出してPathTraversalErrorをスロー"""
        # ベースディレクトリ外にファイルを作成
        outside_file = base_dir.parent / "secret.png"
        outside_file.write_bytes(_create_test_image())
        try:
            malicious_path = base_dir / ".." / "secret.png"
            with pytest.raises(PathTraversalError):
                processor_with_validator.extract_text(str(malicious_path))
        finally:
            outside_file.unlink(missing_ok=True)

    def test_絶対パスによるベースディレクトリ外アクセスで例外(
        self, processor_with_validator
    ):
        """ベースディレクトリ外の絶対パスでPathTraversalErrorをスロー"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_create_test_image())
            outside_path = f.name
        try:
            with pytest.raises(PathTraversalError):
                processor_with_validator.extract_text(outside_path)
        finally:
            Path(outside_path).unlink(missing_ok=True)

    def test_ファイルサイズ超過で例外(
        self, processor_with_validator, base_dir
    ):
        """10MB超のファイルでFileSizeErrorをスロー"""
        large_file = base_dir / "large.png"
        data = _create_test_image(width=100, height=100, fmt="PNG")
        large_file.write_bytes(data + b"\x00" * (11 * 1024 * 1024))

        with pytest.raises(FileSizeError, match="ファイルサイズ"):
            processor_with_validator.extract_text(str(large_file))

    def test_不正な拡張子で例外(
        self, processor_with_validator, base_dir
    ):
        """サポートされていない拡張子でFileFormatErrorをスロー"""
        txt_file = base_dir / "document.txt"
        txt_file.write_text("テキストファイル")

        with pytest.raises(FileFormatError, match="許可されていない"):
            processor_with_validator.extract_text(str(txt_file))

    def test_MIME不一致で例外(
        self, processor_with_validator, base_dir
    ):
        """拡張子とMIMEタイプの不一致でFileFormatErrorをスロー"""
        # PNG内容をJPG拡張子で保存
        fake_jpg = base_dir / "actually_png.jpg"
        fake_jpg.write_bytes(_create_test_image(fmt="PNG"))

        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            processor_with_validator.extract_text(str(fake_jpg))

    def test_画像解像度超過で例外(self, base_dir):
        """解像度制限を超える画像でImageDimensionsErrorをスロー"""
        validator = FileValidator(
            allowed_base_dir=base_dir,
            max_dimensions=(50, 50),
        )
        with patch("src.ocr.processor.PaddleOCR") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            proc = OCRProcessor(lang="japan", validator=validator)

            img_file = _create_temp_image_file(
                base_dir, "oversized.png", width=100, height=100
            )

            with pytest.raises(ImageDimensionsError, match="解像度"):
                proc.extract_text(str(img_file))

    def test_存在しないファイルで例外(
        self, processor_with_validator, base_dir
    ):
        """存在しないファイルパスでValidationErrorをスロー"""
        nonexistent = base_dir / "nonexistent.png"
        with pytest.raises(ValidationError, match="ファイルが見つかりません"):
            processor_with_validator.extract_text(str(nonexistent))

    def test_NULLバイトを含むパスで例外(
        self, processor_with_validator, base_dir
    ):
        """NULLバイトを含むパスでValidationErrorをスロー"""
        malicious_path = str(base_dir / "image\x00.png")
        with pytest.raises(ValidationError):
            processor_with_validator.extract_text(malicious_path)


# ===========================================================================
# カスタムvalidatorのテスト
# ===========================================================================


class TestCustomValidatorIntegration:
    """カスタムFileValidatorによる独自ルール適用テスト"""

    def test_カスタムvalidatorで厳格な検証(self, processor_strict):
        """カスタムFileValidatorで1MB以下のPNGのみ許可"""
        # 有効なPNGバイナリデータ（小さいサイズ）
        data = _create_test_image(width=50, height=50, fmt="PNG")
        processor_strict._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_strict.extract_text(data)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_カスタムvalidatorでJPEG拒否(self, processor_strict):
        """PNGのみ許可のvalidatorでJPEGバイナリが拒否される"""
        data = _create_test_image(width=50, height=50, fmt="JPEG")

        with pytest.raises(FileFormatError):
            processor_strict.extract_text(data)

    def test_カスタムvalidatorでサイズ超過(self, processor_strict):
        """1MB制限のvalidatorで大きいバイナリが拒否される"""
        # 1MB超のデータを作成
        base_data = _create_test_image(width=100, height=100, fmt="PNG")
        large_data = base_data + b"\x00" * (1_500_000)

        with pytest.raises(FileSizeError):
            processor_strict.extract_text(large_data)

    def test_カスタムvalidatorで解像度超過(self, processor_strict):
        """500x500制限のvalidatorで大きい画像が拒否される"""
        large_image = Image.new("RGB", (600, 400), color="blue")
        processor_strict._engine.ocr.return_value = MOCK_OCR_RESULT

        with pytest.raises(ImageDimensionsError, match="解像度"):
            processor_strict.extract_text(large_image)


# ===========================================================================
# デフォルトvalidator（後方互換性）のテスト
# ===========================================================================


class TestDefaultValidatorIntegration:
    """validator=None（デフォルト）の場合の後方互換性テスト"""

    def test_validatorなしでデフォルト動作(self, processor_default):
        """validator=Noneでデフォルト設定が使用される"""
        # デフォルトvalidatorはパス制限なし
        assert processor_default._validator is not None
        assert processor_default._validator._allowed_base_dir is None

    def test_デフォルトvalidatorでバイナリデータ処理(
        self, processor_default
    ):
        """デフォルトvalidatorで有効なバイナリデータの処理が成功する"""
        data = _create_test_image(width=50, height=50, fmt="PNG")
        processor_default._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_default.extract_text(data)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_デフォルトvalidatorでPIL_Image処理(
        self, processor_default
    ):
        """デフォルトvalidatorでPIL Imageの処理が成功する"""
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))
        processor_default._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_default.extract_text(image)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_デフォルトvalidatorの許容拡張子(self, processor_default):
        """デフォルトvalidatorが.jpg .jpeg .png .bmp .tiffを許可する"""
        expected = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        actual = processor_default._validator._allowed_extensions
        assert actual == expected


# ===========================================================================
# 統合シナリオ
# ===========================================================================


class TestIntegrationScenarios:
    """複合的な統合シナリオのテスト"""

    def test_検証成功後にOCRエンジンエラー(
        self, processor_with_validator, base_dir
    ):
        """検証は通過したがOCRエンジンでエラー発生 → OCRProcessingError"""
        img_file = _create_temp_image_file(base_dir, "valid.png")
        processor_with_validator._engine.ocr.side_effect = RuntimeError(
            "OCRエンジン内部エラー"
        )

        with pytest.raises(OCRProcessingError, match="OCR処理中にエラー"):
            processor_with_validator.extract_text(str(img_file))

    def test_バイナリデータの検証とOCR処理(
        self, processor_with_validator
    ):
        """バイナリデータを検証してOCR処理"""
        data = _create_test_image(width=50, height=50, fmt="PNG")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(data)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_PIL_Imageの検証とOCR処理(
        self, processor_with_validator
    ):
        """PIL Imageを検証してOCR処理"""
        image = Image.new("RGB", (200, 100), color=(255, 255, 255))
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(image)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_空のOCR結果が空文字を返す(
        self, processor_with_validator, base_dir
    ):
        """検証成功後、OCR結果が空の場合は空文字列を返す"""
        img_file = _create_temp_image_file(base_dir, "blank.png")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_EMPTY_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert result == ""

    def test_空のOCR結果が詳細モードで空リストを返す(
        self, processor_with_validator, base_dir
    ):
        """検証成功後、OCR結果が空の場合は詳細モードで空リストを返す"""
        img_file = _create_temp_image_file(base_dir, "blank.png")
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_EMPTY_RESULT

        result = processor_with_validator.extract_text(
            str(img_file), detailed=True
        )

        assert result == []

    def test_不正なバイナリデータで検証エラー(
        self, processor_with_validator
    ):
        """画像として認識できないバイナリデータでFileFormatErrorをスロー"""
        invalid_bytes = b"This is plain text, not an image."

        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            processor_with_validator.extract_text(invalid_bytes)

    def test_空のバイナリデータで検証エラー(
        self, processor_with_validator
    ):
        """空のバイナリデータでValidationErrorをスロー"""
        with pytest.raises(ValidationError, match="空"):
            processor_with_validator.extract_text(b"")

    def test_不正な入力型で例外(self, processor_with_validator):
        """サポートされていない型で適切なエラーがスローされる"""
        # FileValidatorのvalidateがValidationErrorをスローする
        with pytest.raises((ValidationError, OCRProcessingError)):
            processor_with_validator.extract_text(12345)


# ===========================================================================
# エラー型の分離テスト
# ===========================================================================


class TestErrorTypeSeparation:
    """ValidationError系とOCRProcessingErrorが適切に分離されるテスト"""

    def test_ValidationErrorはOCRProcessingErrorではない(self):
        """ValidationErrorとOCRProcessingErrorは独立した例外クラス"""
        assert not issubclass(ValidationError, OCRProcessingError)
        assert not issubclass(OCRProcessingError, ValidationError)

    def test_PathTraversalErrorはValidationError(self):
        """PathTraversalErrorはValidationErrorのサブクラス"""
        assert issubclass(PathTraversalError, ValidationError)

    def test_FileSizeErrorはValidationError(self):
        """FileSizeErrorはValidationErrorのサブクラス"""
        assert issubclass(FileSizeError, ValidationError)

    def test_FileFormatErrorはValidationError(self):
        """FileFormatErrorはValidationErrorのサブクラス"""
        assert issubclass(FileFormatError, ValidationError)

    def test_ImageDimensionsErrorはValidationError(self):
        """ImageDimensionsErrorはValidationErrorのサブクラス"""
        assert issubclass(ImageDimensionsError, ValidationError)

    def test_検証エラーとOCRエラーを分離してcatchできる(
        self, processor_with_validator, base_dir
    ):
        """try-exceptで検証エラーとOCRエラーを個別にcatchできる"""
        # パストラバーサルエラー → ValidationError系
        outside_file = base_dir.parent / "secret.png"
        outside_file.write_bytes(_create_test_image())

        try:
            malicious_path = base_dir / ".." / "secret.png"
            caught_validation = False
            caught_ocr = False

            try:
                processor_with_validator.extract_text(str(malicious_path))
            except ValidationError:
                caught_validation = True
            except OCRProcessingError:
                caught_ocr = True

            assert caught_validation is True
            assert caught_ocr is False
        finally:
            outside_file.unlink(missing_ok=True)


# ===========================================================================
# __init__.py エクスポートのテスト
# ===========================================================================


class TestModuleExports:
    """src.ocrパッケージのエクスポートが正しいことを確認するテスト"""

    def test_OCRProcessorがエクスポートされている(self):
        """OCRProcessorがsrc.ocrからインポートできる"""
        from src.ocr import OCRProcessor
        assert OCRProcessor is not None

    def test_OCRProcessingErrorがエクスポートされている(self):
        """OCRProcessingErrorがsrc.ocrからインポートできる"""
        from src.ocr import OCRProcessingError
        assert OCRProcessingError is not None

    def test_ValidationErrorがエクスポートされている(self):
        """ValidationErrorがsrc.ocrからインポートできる"""
        from src.ocr import ValidationError
        assert ValidationError is not None

    def test_PathTraversalErrorがエクスポートされている(self):
        """PathTraversalErrorがsrc.ocrからインポートできる"""
        from src.ocr import PathTraversalError
        assert PathTraversalError is not None

    def test_FileSizeErrorがエクスポートされている(self):
        """FileSizeErrorがsrc.ocrからインポートできる"""
        from src.ocr import FileSizeError
        assert FileSizeError is not None

    def test_FileFormatErrorがエクスポートされている(self):
        """FileFormatErrorがsrc.ocrからインポートできる"""
        from src.ocr import FileFormatError
        assert FileFormatError is not None

    def test_ImageDimensionsErrorがエクスポートされている(self):
        """ImageDimensionsErrorがsrc.ocrからインポートできる"""
        from src.ocr import ImageDimensionsError
        assert ImageDimensionsError is not None


# ===========================================================================
# パディング統合テスト
# ===========================================================================


class TestPaddingIntegration:
    """パディング前処理の統合テスト

    画像の端にあるテキストが検出されない問題を防ぐための
    パディング前処理が、各入力形式で正しく動作することを確認する。
    """

    def test_パディング有効時にファイルパスからOCR処理成功(
        self, processor_with_validator, base_dir
    ):
        """パディング有効時にファイルパスからOCR処理が成功する"""
        img_file = _create_temp_image_file(base_dir, "padded.png")
        processor_with_validator._add_padding = True
        processor_with_validator._padding_size = 30
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_パディング有効時にバイナリデータからOCR処理成功(
        self, processor_with_validator
    ):
        """パディング有効時にバイナリデータからOCR処理が成功する"""
        data = _create_test_image(width=50, height=50, fmt="PNG")
        processor_with_validator._add_padding = True
        processor_with_validator._padding_size = 30
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(data)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_パディング有効時にPIL_ImageからOCR処理成功(
        self, processor_with_validator
    ):
        """パディング有効時にPIL ImageからOCR処理が成功する"""
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        processor_with_validator._add_padding = True
        processor_with_validator._padding_size = 30
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(image)

        assert isinstance(result, str)
        assert "テスト文字列" in result

    def test_パディング無効時の後方互換性(
        self, processor_with_validator, base_dir
    ):
        """パディングを無効化しても既存の動作が維持される"""
        img_file = _create_temp_image_file(base_dir, "no_padding.png")
        processor_with_validator._add_padding = False
        processor_with_validator._engine.ocr.return_value = MOCK_OCR_RESULT

        result = processor_with_validator.extract_text(str(img_file))

        assert isinstance(result, str)
        assert "テスト文字列" in result


_TEST1_JPG_PATH = Path(__file__).parent.parent.parent / "test_img" / "test1.jpg"


@pytest.mark.skipif(
    not _TEST1_JPG_PATH.exists(),
    reason="テスト画像 test_img/test1.jpg が存在しません",
)
class TestTest1JpgRegression:
    """test1.jpg の回帰テスト

    test1.jpg は各行に「0123456789」が記載された画像。
    パディングなしでは右端の「9」がバウンディングボックスの
    検出範囲外になり認識できないバグがあった。

    注意: このテストは実際のPaddleOCRエンジンを使用する統合テスト。
    """

    @pytest.fixture
    def real_processor(self):
        """実際のPaddleOCRエンジンを使用するプロセッサ"""
        return OCRProcessor(lang="japan")

    @pytest.mark.slow
    def test_test1_jpgで全行に9が含まれる(self, real_processor):
        """test1.jpg の全行で「9」が認識されることを確認

        バグ再現: パディングなしでは右端の「9」が欠落する。
        パディング有効により、全行で「9」が認識されるはず。
        """
        test_image_path = str(
            Path(__file__).parent.parent.parent / "test_img" / "test1.jpg"
        )
        result = real_processor.extract_text(test_image_path)
        lines = result.strip().split("\n")

        # 7行のテキストが認識されるはず
        assert len(lines) >= 6, (
            f"期待: 6行以上, 実際: {len(lines)}行\n結果: {result}"
        )

        # 全行でスペース除去後に「9」が含まれることを確認
        for i, line in enumerate(lines):
            cleaned = line.replace(" ", "")
            assert "9" in cleaned, (
                f"行{i + 1}で「9」が認識されていません: {repr(line)}\n"
                f"全結果:\n{result}"
            )

    @pytest.mark.slow
    def test_test1_jpgで全行に0から9が含まれる(self, real_processor):
        """test1.jpg の全行で0-9の全数字が認識されることを確認"""
        test_image_path = str(
            Path(__file__).parent.parent.parent / "test_img" / "test1.jpg"
        )
        result = real_processor.extract_text(test_image_path)
        lines = result.strip().split("\n")

        for i, line in enumerate(lines):
            cleaned = line.replace(" ", "")
            for digit in "0123456789":
                assert digit in cleaned, (
                    f"行{i + 1}で「{digit}」が認識されていません: "
                    f"{repr(line)}\n全結果:\n{result}"
                )

    @pytest.mark.slow
    def test_test1_jpgでパディング無効だと9が欠落する(self):
        """パディング無効の場合、test1.jpg で「9」が欠落することを確認

        このテストはバグの根本原因を文書化する。
        パディングを無効にすると、右端の「9」が検出されない行がある。
        """
        processor_no_padding = OCRProcessor(
            lang="japan", add_padding=False
        )
        test_image_path = str(
            Path(__file__).parent.parent.parent / "test_img" / "test1.jpg"
        )
        result = processor_no_padding.extract_text(test_image_path)
        lines = result.strip().split("\n")

        # パディングなしでは少なくとも一部の行で「9」が欠落する
        missing_9_count = sum(
            1 for line in lines if "9" not in line.replace(" ", "")
        )
        assert missing_9_count > 0, (
            "パディングなしでも全行で「9」が認識された"
            "（バグが再現しない環境の可能性あり）"
        )
