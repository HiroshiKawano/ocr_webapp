"""FileValidatorのユニットテスト

TDD RED フェーズ: 包括的なテストケースを先に定義する。
テスト対象:
  - ファイルパス検証（パストラバーサル、拡張子、サイズ、MIME）
  - バイナリデータ検証（サイズ、形式、破損データ）
  - PIL Image 検証（解像度、モード）
  - エッジケース（境界値、複数エラー、設定カスタマイズ）
"""

import io
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

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
    # BMPモードの場合はRGBに変換して保存
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


def _create_fake_png_with_text_content(tmp_path: Path) -> Path:
    """拡張子は.pngだが中身がテキストのファイルを作成する"""
    file_path = tmp_path / "fake.png"
    file_path.write_text("これはテキストファイルです")
    return file_path


def _create_large_bytes(size_mb: float) -> bytes:
    """指定サイズ（MB）のバイナリデータを生成する"""
    size_bytes = int(size_mb * 1024 * 1024)
    # 有効なPNGヘッダー + パディング
    header = _create_test_image(width=1, height=1, fmt="PNG")
    if size_bytes <= len(header):
        return header[:size_bytes]
    return header + b"\x00" * (size_bytes - len(header))


def _create_bmp_bytes(width: int = 10, height: int = 10) -> bytes:
    """テスト用のBMP画像バイナリを生成する"""
    return _create_test_image(width=width, height=height, fmt="BMP")


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
def validator(base_dir: Path) -> FileValidator:
    """デフォルト設定のFileValidator"""
    return FileValidator(allowed_base_dir=base_dir)


@pytest.fixture
def validator_no_base() -> FileValidator:
    """ベースディレクトリ制限なしのFileValidator"""
    return FileValidator()


@pytest.fixture
def valid_png(base_dir: Path) -> Path:
    """有効なPNG画像ファイル"""
    return _create_temp_image_file(base_dir, "valid.png", fmt="PNG")


@pytest.fixture
def valid_jpg(base_dir: Path) -> Path:
    """有効なJPEG画像ファイル"""
    return _create_temp_image_file(base_dir, "valid.jpg", fmt="JPEG")


@pytest.fixture
def valid_bmp(base_dir: Path) -> Path:
    """有効なBMP画像ファイル"""
    return _create_temp_image_file(base_dir, "valid.bmp", fmt="BMP")


# ===========================================================================
# カスタム例外のテスト
# ===========================================================================


class TestExceptions:
    """カスタム例外の継承関係テスト"""

    def test_validation_error_is_exception(self):
        """ValidationErrorはExceptionを継承している"""
        assert issubclass(ValidationError, Exception)

    def test_path_traversal_error_inherits_validation_error(self):
        """PathTraversalErrorはValidationErrorを継承している"""
        assert issubclass(PathTraversalError, ValidationError)

    def test_file_size_error_inherits_validation_error(self):
        """FileSizeErrorはValidationErrorを継承している"""
        assert issubclass(FileSizeError, ValidationError)

    def test_file_format_error_inherits_validation_error(self):
        """FileFormatErrorはValidationErrorを継承している"""
        assert issubclass(FileFormatError, ValidationError)

    def test_image_dimensions_error_inherits_validation_error(self):
        """ImageDimensionsErrorはValidationErrorを継承している"""
        assert issubclass(ImageDimensionsError, ValidationError)


# ===========================================================================
# ValidationResult のテスト
# ===========================================================================


class TestValidationResult:
    """ValidationResultデータクラスのテスト"""

    def test_valid_result_creation(self):
        """有効な検証結果を作成できる"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            file_info={"size": 1024},
        )
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.file_info == {"size": 1024}

    def test_invalid_result_creation(self):
        """無効な検証結果を作成できる"""
        result = ValidationResult(
            is_valid=False,
            errors=["サイズ超過"],
            warnings=["大きな画像です"],
            file_info={"size": 20_000_000},
        )
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


# ===========================================================================
# FileValidator 初期化テスト
# ===========================================================================


class TestFileValidatorInit:
    """FileValidatorの初期化テスト"""

    def test_default_initialization(self):
        """デフォルト値で初期化できる"""
        v = FileValidator()
        assert v._max_file_size == 10 * 1024 * 1024
        assert ".png" in v._allowed_extensions
        assert ".jpg" in v._allowed_extensions
        assert ".jpeg" in v._allowed_extensions
        assert ".bmp" in v._allowed_extensions
        assert v._max_dimensions == (10000, 10000)
        assert v._allowed_base_dir is None

    def test_custom_file_size(self):
        """カスタムファイルサイズ制限で初期化できる"""
        v = FileValidator(max_file_size=5 * 1024 * 1024)
        assert v._max_file_size == 5 * 1024 * 1024

    def test_custom_extensions(self):
        """カスタム拡張子リストで初期化できる"""
        v = FileValidator(allowed_extensions={".gif", ".webp"})
        assert v._allowed_extensions == {".gif", ".webp"}
        assert ".png" not in v._allowed_extensions

    def test_custom_max_dimensions(self):
        """カスタム解像度制限で初期化できる"""
        v = FileValidator(max_dimensions=(5000, 5000))
        assert v._max_dimensions == (5000, 5000)

    def test_custom_base_dir_as_string(self):
        """文字列でベースディレクトリを指定できる"""
        v = FileValidator(allowed_base_dir="/tmp/uploads")
        assert v._allowed_base_dir == Path("/tmp/uploads")

    def test_custom_base_dir_as_path(self):
        """Pathでベースディレクトリを指定できる"""
        v = FileValidator(allowed_base_dir=Path("/tmp/uploads"))
        assert v._allowed_base_dir == Path("/tmp/uploads")


# ===========================================================================
# ファイルパス検証テスト
# ===========================================================================


class TestValidateFilePath:
    """ファイルパス検証のテスト"""

    def test_valid_png_file(self, validator: FileValidator, valid_png: Path):
        """有効なPNGファイルの検証に成功する"""
        result = validator.validate_file_path(valid_png)
        assert result.is_valid is True
        assert result.errors == []
        assert "size" in result.file_info
        assert "format" in result.file_info

    def test_valid_jpg_file(self, validator: FileValidator, valid_jpg: Path):
        """有効なJPEGファイルの検証に成功する"""
        result = validator.validate_file_path(valid_jpg)
        assert result.is_valid is True

    def test_valid_bmp_file(self, validator: FileValidator, valid_bmp: Path):
        """有効なBMPファイルの検証に成功する"""
        result = validator.validate_file_path(valid_bmp)
        assert result.is_valid is True

    def test_string_path_input(self, validator: FileValidator, valid_png: Path):
        """文字列パスを受け付ける"""
        result = validator.validate_file_path(str(valid_png))
        assert result.is_valid is True

    def test_nonexistent_file(self, validator: FileValidator, base_dir: Path):
        """存在しないファイルで ValidationError を送出する"""
        nonexistent = base_dir / "nonexistent.png"
        with pytest.raises(ValidationError, match="ファイルが見つかりません"):
            validator.validate_file_path(nonexistent)

    def test_disallowed_extension_txt(
        self, validator: FileValidator, base_dir: Path
    ):
        """不正な拡張子(.txt)で ValidationError を送出する"""
        txt_file = base_dir / "document.txt"
        txt_file.write_text("テキスト内容")
        with pytest.raises(FileFormatError, match="許可されていない"):
            validator.validate_file_path(txt_file)

    def test_disallowed_extension_pdf(
        self, validator: FileValidator, base_dir: Path
    ):
        """不正な拡張子(.pdf)で ValidationError を送出する"""
        pdf_file = base_dir / "document.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(FileFormatError, match="許可されていない"):
            validator.validate_file_path(pdf_file)

    def test_disallowed_extension_exe(
        self, validator: FileValidator, base_dir: Path
    ):
        """不正な拡張子(.exe)で ValidationError を送出する"""
        exe_file = base_dir / "malware.exe"
        exe_file.write_bytes(b"MZ" + b"\x00" * 100)
        with pytest.raises(FileFormatError, match="許可されていない"):
            validator.validate_file_path(exe_file)

    def test_file_size_exceeds_limit(
        self, validator: FileValidator, base_dir: Path
    ):
        """10MBを超えるファイルで FileSizeError を送出する"""
        large_file = base_dir / "large.png"
        # 11MB のデータ
        data = _create_test_image(width=100, height=100, fmt="PNG")
        large_file.write_bytes(data + b"\x00" * (11 * 1024 * 1024))
        with pytest.raises(FileSizeError, match="ファイルサイズ"):
            validator.validate_file_path(large_file)

    def test_mime_type_mismatch(
        self, validator: FileValidator, base_dir: Path
    ):
        """拡張子とMIMEタイプが一致しない場合 FileFormatError を送出する"""
        fake_png = _create_fake_png_with_text_content(base_dir)
        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            validator.validate_file_path(fake_png)

    def test_file_info_contains_dimensions(
        self, validator: FileValidator, valid_png: Path
    ):
        """file_infoに解像度情報が含まれる"""
        result = validator.validate_file_path(valid_png)
        assert "width" in result.file_info
        assert "height" in result.file_info
        assert result.file_info["width"] == 100
        assert result.file_info["height"] == 100

    def test_file_info_contains_extension(
        self, validator: FileValidator, valid_png: Path
    ):
        """file_infoに拡張子情報が含まれる"""
        result = validator.validate_file_path(valid_png)
        assert result.file_info["extension"] == ".png"


# ===========================================================================
# パストラバーサル防御テスト
# ===========================================================================


class TestPathTraversalDefense:
    """パストラバーサル攻撃の防御テスト"""

    def test_relative_path_traversal(
        self, validator: FileValidator, base_dir: Path
    ):
        """../を使ったパストラバーサルを防ぐ"""
        # ベースディレクトリの外にファイルを作成
        outside_file = base_dir.parent / "secret.png"
        outside_file.write_bytes(_create_test_image())
        try:
            malicious_path = base_dir / ".." / "secret.png"
            with pytest.raises(PathTraversalError):
                validator.validate_file_path(malicious_path)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_absolute_path_escape(
        self, validator: FileValidator, base_dir: Path
    ):
        """絶対パスによるベースディレクトリ外へのアクセスを防ぐ"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_create_test_image())
            outside_path = Path(f.name)
        try:
            with pytest.raises(PathTraversalError):
                validator.validate_file_path(outside_path)
        finally:
            outside_path.unlink(missing_ok=True)

    def test_symlink_escape(
        self, validator: FileValidator, base_dir: Path, tmp_path: Path
    ):
        """シンボリックリンク経由の脱出を防ぐ"""
        # ベースディレクトリ外にファイルを作成
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.png"
        outside_file.write_bytes(_create_test_image())

        # ベースディレクトリ内にシンボリックリンクを作成
        symlink_path = base_dir / "link.png"
        symlink_path.symlink_to(outside_file)

        with pytest.raises(PathTraversalError):
            validator.validate_file_path(symlink_path)

    def test_double_dot_in_middle_of_path(
        self, validator: FileValidator, base_dir: Path
    ):
        """パス中間の..を検出する"""
        sub_dir = base_dir / "subdir"
        sub_dir.mkdir()
        # ベースディレクトリの外にファイルを作成
        outside_file = base_dir.parent / "escape.png"
        outside_file.write_bytes(_create_test_image())
        try:
            malicious = base_dir / "subdir" / ".." / ".." / "escape.png"
            with pytest.raises(PathTraversalError):
                validator.validate_file_path(malicious)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_no_base_dir_allows_any_path(
        self, validator_no_base: FileValidator, tmp_path: Path
    ):
        """ベースディレクトリ未指定の場合はパス制限なし"""
        img_file = _create_temp_image_file(tmp_path, "anywhere.png")
        result = validator_no_base.validate_file_path(img_file)
        assert result.is_valid is True

    def test_path_with_null_bytes(
        self, validator: FileValidator, base_dir: Path
    ):
        """NULLバイトを含むパスを拒否する"""
        malicious_path = str(base_dir / "image\x00.png")
        with pytest.raises(ValidationError):
            validator.validate_file_path(malicious_path)

    def test_subdirectory_within_base_is_allowed(
        self, validator: FileValidator, base_dir: Path
    ):
        """ベースディレクトリ内のサブディレクトリは許可される"""
        sub_dir = base_dir / "2024" / "01"
        sub_dir.mkdir(parents=True)
        img_file = _create_temp_image_file(sub_dir, "photo.png")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True


# ===========================================================================
# バイナリデータ検証テスト
# ===========================================================================


class TestValidateBytes:
    """バイナリデータ検証のテスト"""

    def test_valid_png_bytes(self, validator: FileValidator):
        """有効なPNGバイナリの検証に成功する"""
        data = _create_test_image(fmt="PNG")
        result = validator.validate_bytes(data)
        assert result.is_valid is True
        assert "size" in result.file_info
        assert "format" in result.file_info

    def test_valid_jpeg_bytes(self, validator: FileValidator):
        """有効なJPEGバイナリの検証に成功する"""
        data = _create_test_image(fmt="JPEG")
        result = validator.validate_bytes(data)
        assert result.is_valid is True

    def test_valid_bmp_bytes(self, validator: FileValidator):
        """有効なBMPバイナリの検証に成功する"""
        data = _create_bmp_bytes()
        result = validator.validate_bytes(data)
        assert result.is_valid is True

    def test_empty_bytes(self, validator: FileValidator):
        """空のバイナリデータで ValidationError を送出する"""
        with pytest.raises(ValidationError, match="空"):
            validator.validate_bytes(b"")

    def test_non_image_bytes(self, validator: FileValidator):
        """画像でないバイナリデータで FileFormatError を送出する"""
        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            validator.validate_bytes(b"This is plain text, not an image.")

    def test_corrupted_image_bytes(self, validator: FileValidator):
        """破損した画像データで ValidationError を送出する"""
        # PNGヘッダーのみで本体が壊れている
        corrupted = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        with pytest.raises(ValidationError):
            validator.validate_bytes(corrupted)

    def test_size_exceeds_limit(self, validator: FileValidator):
        """10MBを超えるバイナリデータで FileSizeError を送出する"""
        large_data = _create_large_bytes(11.0)
        with pytest.raises(FileSizeError, match="ファイルサイズ"):
            validator.validate_bytes(large_data)

    def test_size_exactly_at_limit(self):
        """ちょうど10MBのデータは受け入れる"""
        v = FileValidator(max_file_size=10 * 1024 * 1024)
        # ちょうど10MBの有効な画像データを作成
        base_image = _create_test_image(width=100, height=100, fmt="PNG")
        # 10MBちょうどにするためパディング
        # 実際にはバイナリデータのサイズで判定するため、パディングを加算
        target_size = 10 * 1024 * 1024
        padded = base_image + b"\x00" * (target_size - len(base_image))
        # サイズがちょうど10MBであることを確認
        assert len(padded) == target_size
        # 10MBちょうどは許可される（サイズチェックのみテスト）
        # 画像として壊れている可能性があるため、サイズチェック段階で通過すればOK
        # validate_bytesではサイズチェック後に形式チェックがあるので、
        # ここではサイズエラーが出ないことを確認
        try:
            v.validate_bytes(padded)
        except FileSizeError:
            pytest.fail("10MBちょうどのデータで FileSizeError が発生してはならない")
        except ValidationError:
            # 画像形式エラーは許容（パディングで壊れるため）
            pass

    def test_file_info_for_valid_bytes(self, validator: FileValidator):
        """有効なバイナリデータのfile_infoに正しい情報が含まれる"""
        data = _create_test_image(width=200, height=150, fmt="PNG")
        result = validator.validate_bytes(data)
        assert result.file_info["width"] == 200
        assert result.file_info["height"] == 150
        assert result.file_info["size"] == len(data)

    def test_gif_bytes_rejected_by_default(self, validator: FileValidator):
        """GIF形式はデフォルトで拒否される"""
        data = _create_test_image(width=10, height=10, fmt="GIF")
        with pytest.raises(FileFormatError):
            validator.validate_bytes(data)


# ===========================================================================
# PIL Image 検証テスト
# ===========================================================================


class TestValidateImage:
    """PIL Image検証のテスト"""

    def test_valid_rgb_image(self, validator: FileValidator):
        """有効なRGB画像の検証に成功する"""
        image = Image.new("RGB", (100, 100), color="blue")
        result = validator.validate_image(image)
        assert result.is_valid is True
        assert result.file_info["width"] == 100
        assert result.file_info["height"] == 100

    def test_valid_rgba_image(self, validator: FileValidator):
        """有効なRGBA画像の検証に成功する"""
        image = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        result = validator.validate_image(image)
        assert result.is_valid is True

    def test_valid_l_mode_image(self, validator: FileValidator):
        """グレースケール画像の検証に成功する"""
        image = Image.new("L", (100, 100), color=128)
        result = validator.validate_image(image)
        assert result.is_valid is True

    def test_dimensions_exceed_limit(self, validator: FileValidator):
        """解像度制限を超えた画像で ImageDimensionsError を送出する"""
        image = Image.new("RGB", (10001, 100), color="red")
        with pytest.raises(ImageDimensionsError, match="解像度"):
            validator.validate_image(image)

    def test_dimensions_exceed_limit_height(self, validator: FileValidator):
        """高さが解像度制限を超えた場合 ImageDimensionsError を送出する"""
        image = Image.new("RGB", (100, 10001), color="red")
        with pytest.raises(ImageDimensionsError, match="解像度"):
            validator.validate_image(image)

    def test_dimensions_exactly_at_limit(self, validator: FileValidator):
        """ちょうど10000x10000の画像は受け入れる"""
        # 実際にメモリを大量消費するので小さめのカスタムバリデータを使う
        v = FileValidator(max_dimensions=(500, 500))
        image = Image.new("RGB", (500, 500), color="green")
        result = v.validate_image(image)
        assert result.is_valid is True

    def test_dimensions_one_over_limit(self):
        """制限+1ピクセルの画像で ImageDimensionsError を送出する"""
        v = FileValidator(max_dimensions=(500, 500))
        image = Image.new("RGB", (501, 500), color="green")
        with pytest.raises(ImageDimensionsError):
            v.validate_image(image)

    def test_palette_mode_image_warning(self, validator: FileValidator):
        """パレットモード(P)の画像は警告を含む"""
        image = Image.new("P", (100, 100))
        result = validator.validate_image(image)
        # パレットモードは検証自体は成功するが警告を含む
        assert result.is_valid is True
        assert any("P" in w or "パレット" in w for w in result.warnings)

    def test_cmyk_mode_image_warning(self, validator: FileValidator):
        """CMYKモードの画像は警告を含む"""
        image = Image.new("CMYK", (100, 100))
        result = validator.validate_image(image)
        assert result.is_valid is True
        assert any("CMYK" in w for w in result.warnings)

    def test_file_info_includes_mode(self, validator: FileValidator):
        """file_infoにモード情報が含まれる"""
        image = Image.new("RGB", (100, 100))
        result = validator.validate_image(image)
        assert result.file_info["mode"] == "RGB"

    def test_file_info_includes_dimensions(self, validator: FileValidator):
        """file_infoに解像度情報が含まれる"""
        image = Image.new("RGB", (320, 240))
        result = validator.validate_image(image)
        assert result.file_info["width"] == 320
        assert result.file_info["height"] == 240


# ===========================================================================
# 統合的 validate() メソッドテスト
# ===========================================================================


class TestValidateDispatch:
    """validate()メソッドの入力自動判別テスト"""

    def test_dispatch_file_path_string(
        self, validator: FileValidator, valid_png: Path
    ):
        """文字列パスを正しくディスパッチする"""
        result = validator.validate(str(valid_png))
        assert result.is_valid is True

    def test_dispatch_file_path_path_object(
        self, validator: FileValidator, valid_png: Path
    ):
        """Pathオブジェクトを正しくディスパッチする"""
        result = validator.validate(valid_png)
        assert result.is_valid is True

    def test_dispatch_bytes(self, validator: FileValidator):
        """バイナリデータを正しくディスパッチする"""
        data = _create_test_image(fmt="PNG")
        result = validator.validate(data)
        assert result.is_valid is True

    def test_dispatch_pil_image(self, validator: FileValidator):
        """PIL Imageを正しくディスパッチする"""
        image = Image.new("RGB", (100, 100))
        result = validator.validate(image)
        assert result.is_valid is True

    def test_unsupported_type_raises_error(self, validator: FileValidator):
        """サポートされていない型で ValidationError を送出する"""
        with pytest.raises(ValidationError, match="サポートされていない"):
            validator.validate(12345)

    def test_unsupported_type_list(self, validator: FileValidator):
        """リスト型で ValidationError を送出する"""
        with pytest.raises(ValidationError, match="サポートされていない"):
            validator.validate([1, 2, 3])

    def test_none_input_raises_error(self, validator: FileValidator):
        """None入力で ValidationError を送出する"""
        with pytest.raises(ValidationError, match="サポートされていない"):
            validator.validate(None)


# ===========================================================================
# 設定カスタマイズテスト
# ===========================================================================


class TestCustomConfiguration:
    """設定のカスタマイズテスト"""

    def test_custom_file_size_limit(self, base_dir: Path):
        """カスタムサイズ制限（1MB）が適用される"""
        v = FileValidator(
            max_file_size=1 * 1024 * 1024,
            allowed_base_dir=base_dir,
        )
        # 1.5MBのファイルを作成
        img_file = base_dir / "big.png"
        data = _create_test_image(width=100, height=100, fmt="PNG")
        img_file.write_bytes(data + b"\x00" * (1_500_000))
        with pytest.raises(FileSizeError):
            v.validate_file_path(img_file)

    def test_custom_file_size_allows_smaller(self, base_dir: Path):
        """カスタムサイズ制限以下のファイルは受け入れる"""
        v = FileValidator(
            max_file_size=5 * 1024 * 1024,
            allowed_base_dir=base_dir,
        )
        img_file = _create_temp_image_file(base_dir, "small.png")
        result = v.validate_file_path(img_file)
        assert result.is_valid is True

    def test_custom_extensions_gif_allowed(self):
        """カスタム拡張子でGIFを許可する"""
        v = FileValidator(allowed_extensions={".gif", ".png"})
        data = _create_test_image(width=10, height=10, fmt="GIF")
        result = v.validate_bytes(data)
        assert result.is_valid is True

    def test_custom_extensions_png_rejected(self):
        """カスタム拡張子でPNGを拒否する"""
        v = FileValidator(allowed_extensions={".gif"})
        data = _create_test_image(width=10, height=10, fmt="PNG")
        with pytest.raises(FileFormatError):
            v.validate_bytes(data)

    def test_custom_max_dimensions(self):
        """カスタム解像度制限が適用される"""
        v = FileValidator(max_dimensions=(200, 200))
        image = Image.new("RGB", (201, 100))
        with pytest.raises(ImageDimensionsError):
            v.validate_image(image)

    def test_custom_base_dir(self, tmp_path: Path):
        """カスタムベースディレクトリが適用される"""
        custom_base = tmp_path / "custom_uploads"
        custom_base.mkdir()
        v = FileValidator(allowed_base_dir=custom_base)

        img_file = _create_temp_image_file(custom_base, "test.png")
        result = v.validate_file_path(img_file)
        assert result.is_valid is True

    def test_custom_base_dir_blocks_outside(self, tmp_path: Path):
        """カスタムベースディレクトリ外のアクセスをブロックする"""
        custom_base = tmp_path / "custom_uploads"
        custom_base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        v = FileValidator(allowed_base_dir=custom_base)

        img_file = _create_temp_image_file(outside, "test.png")
        with pytest.raises(PathTraversalError):
            v.validate_file_path(img_file)


# ===========================================================================
# エッジケーステスト
# ===========================================================================


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_very_small_image(self, validator: FileValidator):
        """1x1ピクセルの画像を受け入れる"""
        image = Image.new("RGB", (1, 1))
        result = validator.validate_image(image)
        assert result.is_valid is True
        assert result.file_info["width"] == 1
        assert result.file_info["height"] == 1

    def test_unicode_filename(
        self, validator: FileValidator, base_dir: Path
    ):
        """Unicode文字を含むファイル名を処理できる"""
        img_file = _create_temp_image_file(base_dir, "日本語ファイル名.png")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True

    def test_emoji_filename(
        self, validator: FileValidator, base_dir: Path
    ):
        """絵文字を含むファイル名を処理できる"""
        img_file = _create_temp_image_file(base_dir, "photo_camera.png")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True

    def test_case_insensitive_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """大文字の拡張子(.PNG)も受け入れる"""
        img_file = _create_temp_image_file(base_dir, "image.PNG", fmt="PNG")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True

    def test_mixed_case_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """混合ケースの拡張子(.Jpg)も受け入れる"""
        img_file = _create_temp_image_file(base_dir, "image.Jpg", fmt="JPEG")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True

    def test_double_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """二重拡張子(.txt.png)のファイル（中身が画像）を処理する"""
        img_file = _create_temp_image_file(base_dir, "file.txt.png", fmt="PNG")
        result = validator.validate_file_path(img_file)
        # 最後の拡張子が.pngで中身もPNGなので検証に成功する
        assert result.is_valid is True

    def test_no_extension_file(
        self, validator: FileValidator, base_dir: Path
    ):
        """拡張子なしのファイルを拒否する"""
        no_ext = base_dir / "noextension"
        no_ext.write_bytes(_create_test_image())
        with pytest.raises(FileFormatError):
            validator.validate_file_path(no_ext)

    def test_hidden_file(
        self, validator: FileValidator, base_dir: Path
    ):
        """隠しファイル(.image.png)の検証が正常に動作する"""
        hidden_file = _create_temp_image_file(base_dir, ".hidden.png")
        result = validator.validate_file_path(hidden_file)
        assert result.is_valid is True

    def test_one_byte_data(self, validator: FileValidator):
        """1バイトのデータで ValidationError を送出する"""
        with pytest.raises(ValidationError):
            validator.validate_bytes(b"\x00")

    def test_max_dimension_exactly_width(self):
        """最大幅ちょうどの画像は受け入れる"""
        v = FileValidator(max_dimensions=(100, 200))
        image = Image.new("RGB", (100, 100))
        result = v.validate_image(image)
        assert result.is_valid is True

    def test_max_dimension_exactly_height(self):
        """最大高さちょうどの画像は受け入れる"""
        v = FileValidator(max_dimensions=(200, 100))
        image = Image.new("RGB", (100, 100))
        result = v.validate_image(image)
        assert result.is_valid is True

    def test_warnings_only_no_errors(self, validator: FileValidator):
        """警告のみでエラーなしの結果を返す"""
        # CMYKモードの画像 → 警告のみ
        image = Image.new("CMYK", (100, 100))
        result = validator.validate_image(image)
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert len(result.errors) == 0

    def test_multiple_validation_issues_bytes(self):
        """複数の検証問題がある場合すべてを報告する（バイナリ用）"""
        # サイズが非常に小さい制限を設定
        v = FileValidator(
            max_file_size=10,  # 10バイト
            allowed_extensions={".gif"},  # GIFのみ許可
        )
        # PNG画像（サイズ超過 + 拡張子不一致）
        data = _create_test_image(width=10, height=10, fmt="PNG")
        # 少なくとも1つのエラーが出ること
        with pytest.raises(ValidationError):
            v.validate_bytes(data)


# ===========================================================================
# imghdr/MIMEタイプ検証の詳細テスト
# ===========================================================================


class TestMimeTypeValidation:
    """MIMEタイプ検証の詳細テスト"""

    def test_jpg_with_correct_mime(
        self, validator: FileValidator, base_dir: Path
    ):
        """正しいMIMEタイプのJPEGファイルは検証に成功する"""
        img_file = _create_temp_image_file(base_dir, "photo.jpg", fmt="JPEG")
        result = validator.validate_file_path(img_file)
        assert result.is_valid is True

    def test_png_header_but_jpg_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """PNG内容をJPG拡張子で保存した場合 FileFormatError を送出する"""
        fake_jpg = base_dir / "actually_png.jpg"
        fake_jpg.write_bytes(_create_test_image(fmt="PNG"))
        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            validator.validate_file_path(fake_jpg)

    def test_jpeg_header_but_png_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """JPEG内容をPNG拡張子で保存した場合 FileFormatError を送出する"""
        fake_png = base_dir / "actually_jpeg.png"
        fake_png.write_bytes(_create_test_image(fmt="JPEG"))
        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            validator.validate_file_path(fake_png)

    def test_gif_content_with_png_extension(
        self, validator: FileValidator, base_dir: Path
    ):
        """GIF内容をPNG拡張子で保存した場合 FileFormatError を送出する"""
        fake_png = base_dir / "actually_gif.png"
        fake_png.write_bytes(_create_test_image(fmt="GIF"))
        with pytest.raises(FileFormatError, match="不正なファイル形式"):
            validator.validate_file_path(fake_png)


# ===========================================================================
# ファイルパス経由の解像度超過テスト
# ===========================================================================


class TestFilePathDimensionsExceed:
    """ファイルパス検証経由での解像度超過テスト"""

    def test_oversized_image_file(self, base_dir: Path):
        """解像度制限を超える画像ファイルで ImageDimensionsError を送出する"""
        v = FileValidator(
            max_dimensions=(50, 50),
            allowed_base_dir=base_dir,
        )
        # 100x100の画像を作成（制限は50x50）
        img_file = _create_temp_image_file(
            base_dir, "oversized.png", width=100, height=100, fmt="PNG"
        )
        with pytest.raises(ImageDimensionsError, match="解像度"):
            v.validate_file_path(img_file)

    def test_oversized_image_file_height_only(self, base_dir: Path):
        """高さのみ解像度制限を超える画像ファイルで ImageDimensionsError を送出する"""
        v = FileValidator(
            max_dimensions=(200, 50),
            allowed_base_dir=base_dir,
        )
        img_file = _create_temp_image_file(
            base_dir, "tall.png", width=100, height=100, fmt="PNG"
        )
        with pytest.raises(ImageDimensionsError, match="解像度"):
            v.validate_file_path(img_file)


# ===========================================================================
# バイナリデータ経由の解像度超過テスト
# ===========================================================================


class TestBytesDimensionsExceed:
    """バイナリデータ検証経由での解像度超過テスト"""

    def test_oversized_bytes_image(self):
        """解像度制限を超えるバイナリ画像で ImageDimensionsError を送出する"""
        v = FileValidator(max_dimensions=(50, 50))
        data = _create_test_image(width=100, height=100, fmt="PNG")
        with pytest.raises(ImageDimensionsError, match="解像度"):
            v.validate_bytes(data)

    def test_oversized_bytes_image_width_only(self):
        """幅のみ解像度制限を超えるバイナリ画像で ImageDimensionsError を送出する"""
        v = FileValidator(max_dimensions=(50, 200))
        data = _create_test_image(width=100, height=100, fmt="PNG")
        with pytest.raises(ImageDimensionsError, match="解像度"):
            v.validate_bytes(data)


# ===========================================================================
# Image.open失敗テスト（ファイルパス経由）
# ===========================================================================


class TestImageOpenFailure:
    """Image.openが失敗する場合のテスト"""

    def test_corrupted_file_raises_validation_error(self, base_dir: Path):
        """壊れた画像ファイルを開けない場合 ValidationError を送出する"""
        v = FileValidator(allowed_base_dir=base_dir)
        # BMPヘッダーを持つが中身が壊れたファイル
        corrupted_file = base_dir / "corrupted.bmp"
        # BMPのマジックバイト（BM）+ 壊れたヘッダー
        corrupted_data = b"BM" + b"\x00" * 10 + b"\xff" * 20
        corrupted_file.write_bytes(corrupted_data)
        with pytest.raises(ValidationError):
            v.validate_file_path(corrupted_file)

    def test_image_open_with_mock_exception(self, base_dir: Path):
        """Image.openが例外を送出する場合 ValidationError を送出する

        PIL ベースの画像タイプ検出でも Image.open を使用するため、
        モック時は型検出が先に失敗し FileFormatError が発生する。
        """
        v = FileValidator(allowed_base_dir=base_dir)
        img_file = _create_temp_image_file(base_dir, "mock_fail.png")
        with patch("src.validators.file_validator.Image.open") as mock_open:
            mock_open.side_effect = OSError("ファイルを読み込めません")
            with pytest.raises(ValidationError, match="不正なファイル形式です"):
                v.validate_file_path(img_file)
