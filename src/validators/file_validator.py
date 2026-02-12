"""ファイル検証モジュール

セキュアなファイル検証機能を提供する。
パストラバーサル防御、MIMEタイプ検証、画像解像度制限を含む。

セキュリティ要件:
  - パストラバーサル攻撃の防御（シンボリックリンク解決含む）
  - MIMEタイプとファイル拡張子の一致検証
  - ファイルサイズ制限（DoS防止）
  - 画像解像度制限（メモリ枯渇防止）
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 拡張子とMIMEタイプのマッピング
# ---------------------------------------------------------------------------

# imghdrが返す形式名と拡張子の対応
_EXTENSION_TO_IMGHDR: dict[str, set[str]] = {
    ".jpg": {"jpeg"},
    ".jpeg": {"jpeg"},
    ".png": {"png"},
    ".bmp": {"bmp"},
    ".gif": {"gif"},
    ".webp": {"webp"},
    ".tiff": {"tiff"},
    ".tif": {"tiff"},
}

def _detect_image_type_from_path(file_path: Union[str, Path]) -> str | None:
    """ファイルパスから画像タイプを検出する（imghdrの代替）

    Python 3.13で削除された imghdr.what() の代替として、
    PIL を使用して画像形式を判定する。

    Returns:
        検出された形式名（小文字）。検出できない場合は None。
    """
    try:
        with Image.open(file_path) as img:
            fmt = img.format
            return fmt.lower() if fmt else None
    except Exception:
        return None


def _detect_image_type_from_bytes(data: bytes) -> str | None:
    """バイナリデータから画像タイプを検出する（imghdrの代替）

    Python 3.13で削除された imghdr.what(None, h=data) の代替として、
    PIL を使用して画像形式を判定する。

    Returns:
        検出された形式名（小文字）。検出できない場合は None。
    """
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.close()
        return fmt.lower() if fmt else None
    except Exception:
        return None


# imghdrの形式名から拡張子への逆マッピング
_IMGHDR_TO_EXTENSIONS: dict[str, set[str]] = {
    "jpeg": {".jpg", ".jpeg"},
    "png": {".png"},
    "bmp": {".bmp"},
    "gif": {".gif"},
    "webp": {".webp"},
    "tiff": {".tiff", ".tif"},
}

# 推奨モード（RGB、RGBA、L以外は警告）
_RECOMMENDED_MODES: frozenset[str] = frozenset({"RGB", "RGBA", "L"})


# ---------------------------------------------------------------------------
# カスタム例外
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """検証エラーの基底クラス"""


class PathTraversalError(ValidationError):
    """パストラバーサル攻撃が検出された場合のエラー"""


class FileSizeError(ValidationError):
    """ファイルサイズが制限を超えた場合のエラー"""


class FileFormatError(ValidationError):
    """ファイル形式が不正な場合のエラー"""


class ImageDimensionsError(ValidationError):
    """画像解像度が制限を超えた場合のエラー"""


# ---------------------------------------------------------------------------
# ValidationResult データクラス
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """検証結果を表すイミュータブルなデータクラス

    Attributes:
        is_valid: 検証が成功したかどうか
        errors: 検証エラーのリスト
        warnings: 検証警告のリスト
        file_info: ファイル情報の辞書（サイズ、形式、解像度等）
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FileValidator クラス
# ---------------------------------------------------------------------------


class FileValidator:
    """セキュアなファイル検証クラス

    ファイルパス、バイナリデータ、PIL Image のすべての入力形式に対応し、
    パストラバーサル防御、MIMEタイプ検証、サイズ・解像度制限を行う。

    Attributes:
        _max_file_size: 最大ファイルサイズ（バイト）
        _allowed_extensions: 許可する拡張子のセット
        _max_dimensions: 最大解像度（幅, 高さ）のタプル
        _allowed_base_dir: 許可するベースディレクトリ（Noneの場合は制限なし）
    """

    def __init__(
        self,
        *,
        max_file_size: int = 10 * 1024 * 1024,
        allowed_extensions: set[str] | None = None,
        max_dimensions: tuple[int, int] = (10000, 10000),
        allowed_base_dir: str | Path | None = None,
    ) -> None:
        """FileValidatorを初期化する

        Args:
            max_file_size: 最大ファイルサイズ（バイト）。デフォルトは10MB。
            allowed_extensions: 許可する拡張子のセット。
                デフォルトは {".jpg", ".jpeg", ".png", ".bmp"}。
            max_dimensions: 最大解像度 (幅, 高さ)。デフォルトは (10000, 10000)。
            allowed_base_dir: 許可するベースディレクトリ。
                Noneの場合はパス制限を行わない。
        """
        self._max_file_size = max_file_size
        self._allowed_extensions = (
            allowed_extensions
            if allowed_extensions is not None
            else {".jpg", ".jpeg", ".png", ".bmp"}
        )
        self._max_dimensions = max_dimensions
        self._allowed_base_dir = (
            Path(allowed_base_dir) if allowed_base_dir is not None else None
        )

    # -------------------------------------------------------------------
    # 公開メソッド
    # -------------------------------------------------------------------

    def validate(
        self,
        source: str | Path | bytes | Image.Image,
    ) -> ValidationResult:
        """ファイルを検証する（入力形式を自動判別）

        Args:
            source: 検証対象。以下の形式に対応:
                - str: ファイルパス
                - Path: ファイルパス
                - bytes: バイナリデータ
                - PIL.Image.Image: PIL イメージオブジェクト

        Returns:
            ValidationResult: 検証結果

        Raises:
            ValidationError: 検証に失敗した場合
        """
        if isinstance(source, (str, Path)):
            return self.validate_file_path(source)

        if isinstance(source, bytes):
            return self.validate_bytes(source)

        if isinstance(source, Image.Image):
            return self.validate_image(source)

        raise ValidationError(
            f"サポートされていない入力形式です: {type(source).__name__}。"
            "str, Path, bytes, PIL.Image.Image のいずれかを指定してください。"
        )

    def validate_file_path(
        self, file_path: str | Path
    ) -> ValidationResult:
        """ファイルパスを検証する

        以下の順序で検証を行う:
          1. NULLバイトチェック
          2. パストラバーサルチェック
          3. ファイル存在チェック
          4. 拡張子チェック
          5. ファイルサイズチェック
          6. MIMEタイプチェック
          7. 画像解像度チェック

        Args:
            file_path: 検証対象のファイルパス

        Returns:
            ValidationResult: 検証結果

        Raises:
            ValidationError: 検証に失敗した場合
        """
        # NULLバイトチェック
        path_str = str(file_path)
        if "\x00" in path_str:
            raise ValidationError(
                "ファイルパスにNULLバイトが含まれています"
            )

        path = Path(file_path)

        # パストラバーサルチェック
        self._validate_path_traversal(path)

        # ファイル存在チェック
        if not path.exists():
            raise ValidationError(
                f"ファイルが見つかりません: {file_path}"
            )

        # 拡張子チェック
        extension = path.suffix.lower()
        if extension not in self._allowed_extensions:
            raise FileFormatError(
                f"許可されていないファイル拡張子です: {extension}。"
                f"対応拡張子: {', '.join(sorted(self._allowed_extensions))}"
            )

        # ファイルサイズチェック
        file_size = path.stat().st_size
        if file_size > self._max_file_size:
            raise FileSizeError(
                f"ファイルサイズが制限を超えています: "
                f"{file_size:,}バイト（上限: {self._max_file_size:,}バイト）"
            )

        # MIMEタイプチェック
        self._validate_mime_type_for_path(path, extension)

        # 画像を開いて解像度チェック + file_info 収集
        return self._validate_and_collect_image_info_from_path(
            path, extension, file_size
        )

    def validate_bytes(self, data: bytes) -> ValidationResult:
        """バイナリデータを検証する

        以下の順序で検証を行う:
          1. 空データチェック
          2. サイズチェック
          3. 画像形式チェック（MIMEタイプ）
          4. 拡張子マッチング（許可された形式か）
          5. 画像解像度チェック

        Args:
            data: 検証対象のバイナリデータ

        Returns:
            ValidationResult: 検証結果

        Raises:
            ValidationError: 検証に失敗した場合
        """
        # 空データチェック
        if not data:
            raise ValidationError("空のデータが渡されました")

        # サイズチェック
        if len(data) > self._max_file_size:
            raise FileSizeError(
                f"ファイルサイズが制限を超えています: "
                f"{len(data):,}バイト（上限: {self._max_file_size:,}バイト）"
            )

        # 画像形式チェック
        detected_type = _detect_image_type_from_bytes(data)
        if detected_type is None:
            raise FileFormatError(
                "不正なファイル形式です: 画像として認識できないデータです"
            )

        # 許可された形式か確認
        allowed_imghdr_types = self._get_allowed_imghdr_types()
        if detected_type not in allowed_imghdr_types:
            raise FileFormatError(
                f"不正なファイル形式です: 検出={detected_type}、"
                f"許可={', '.join(sorted(allowed_imghdr_types))}"
            )

        # 画像を開いて解像度チェック + file_info 収集
        return self._validate_and_collect_image_info_from_bytes(
            data, detected_type
        )

    def validate_image(self, image: Image.Image) -> ValidationResult:
        """PIL Imageを検証する

        以下の順序で検証を行う:
          1. 解像度チェック
          2. モードチェック（警告のみ）
          3. file_info 収集

        Args:
            image: 検証対象の PIL Image

        Returns:
            ValidationResult: 検証結果

        Raises:
            ImageDimensionsError: 解像度が制限を超えた場合
        """
        width, height = image.size
        max_w, max_h = self._max_dimensions

        # 解像度チェック
        if width > max_w or height > max_h:
            raise ImageDimensionsError(
                f"画像の解像度が制限を超えています: "
                f"{width}x{height}（上限: {max_w}x{max_h}）"
            )

        # モードチェック（警告のみ）
        warnings = self._check_image_mode(image.mode)

        file_info = {
            "width": width,
            "height": height,
            "mode": image.mode,
        }

        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            file_info=file_info,
        )

    # -------------------------------------------------------------------
    # プライベートメソッド: パストラバーサル防御
    # -------------------------------------------------------------------

    def _validate_path_traversal(self, file_path: Path) -> None:
        """パストラバーサル攻撃を防ぐ

        シンボリックリンクを解決し、ベースディレクトリ内であることを確認する。

        Args:
            file_path: 検証対象のパス

        Raises:
            PathTraversalError: ベースディレクトリ外へのアクセスが検出された場合
        """
        if self._allowed_base_dir is None:
            return

        # シンボリックリンクを解決して実際のパスを取得
        resolved_path = file_path.resolve()
        base_dir = self._allowed_base_dir.resolve()

        # ベースディレクトリ内であることを確認
        try:
            resolved_path.relative_to(base_dir)
        except ValueError:
            raise PathTraversalError(
                f"許可されていないパスへのアクセス: {file_path}"
            ) from None

    # -------------------------------------------------------------------
    # プライベートメソッド: MIMEタイプ検証
    # -------------------------------------------------------------------

    def _validate_mime_type_for_path(
        self, file_path: Path, extension: str
    ) -> None:
        """ファイルの実際のMIMEタイプと拡張子の一致を検証する

        Args:
            file_path: 検証対象のファイルパス
            extension: ファイルの拡張子（小文字）

        Raises:
            FileFormatError: MIMEタイプと拡張子が一致しない場合
        """
        detected_type = _detect_image_type_from_path(file_path)

        if detected_type is None:
            raise FileFormatError(
                f"不正なファイル形式です: "
                f"画像として認識できないデータです（拡張子: {extension}）"
            )

        # 拡張子に対応するimghdrの形式名を取得
        expected_types = _EXTENSION_TO_IMGHDR.get(extension, set())

        if detected_type not in expected_types:
            raise FileFormatError(
                f"不正なファイル形式です: "
                f"拡張子={extension}、検出された形式={detected_type}"
            )

    # -------------------------------------------------------------------
    # プライベートメソッド: 画像情報収集
    # -------------------------------------------------------------------

    def _validate_and_collect_image_info_from_path(
        self,
        file_path: Path,
        extension: str,
        file_size: int,
    ) -> ValidationResult:
        """ファイルパスから画像を開き、解像度チェックとfile_info収集を行う

        Args:
            file_path: 画像ファイルのパス
            extension: ファイル拡張子（小文字）
            file_size: ファイルサイズ（バイト）

        Returns:
            ValidationResult: 検証結果

        Raises:
            ImageDimensionsError: 解像度が制限を超えた場合
            ValidationError: 画像を開けない場合
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                img_format = img.format
        except Exception as err:
            raise ValidationError(
                f"画像ファイルを開けません: {err}"
            ) from err

        max_w, max_h = self._max_dimensions
        if width > max_w or height > max_h:
            raise ImageDimensionsError(
                f"画像の解像度が制限を超えています: "
                f"{width}x{height}（上限: {max_w}x{max_h}）"
            )

        warnings = self._check_image_mode(mode)

        file_info = {
            "size": file_size,
            "format": img_format,
            "extension": extension,
            "width": width,
            "height": height,
            "mode": mode,
        }

        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            file_info=file_info,
        )

    def _validate_and_collect_image_info_from_bytes(
        self,
        data: bytes,
        detected_type: str,
    ) -> ValidationResult:
        """バイナリデータから画像を開き、解像度チェックとfile_info収集を行う

        Args:
            data: 画像のバイナリデータ
            detected_type: imghdrが検出した形式名

        Returns:
            ValidationResult: 検証結果

        Raises:
            ImageDimensionsError: 解像度が制限を超えた場合
            ValidationError: 画像を開けない場合
        """
        try:
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            mode = image.mode
            img_format = image.format
        except Exception as err:
            raise ValidationError(
                f"画像データを開けません: {err}"
            ) from err

        max_w, max_h = self._max_dimensions
        if width > max_w or height > max_h:
            raise ImageDimensionsError(
                f"画像の解像度が制限を超えています: "
                f"{width}x{height}（上限: {max_w}x{max_h}）"
            )

        warnings = self._check_image_mode(mode)

        # 検出された形式に対応する拡張子を取得
        extensions = _IMGHDR_TO_EXTENSIONS.get(detected_type, set())
        primary_extension = sorted(extensions)[0] if extensions else ""

        file_info = {
            "size": len(data),
            "format": img_format,
            "extension": primary_extension,
            "width": width,
            "height": height,
            "mode": mode,
            "detected_type": detected_type,
        }

        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            file_info=file_info,
        )

    # -------------------------------------------------------------------
    # プライベートメソッド: ユーティリティ
    # -------------------------------------------------------------------

    def _get_allowed_imghdr_types(self) -> set[str]:
        """許可された拡張子に対応するimghdr形式名のセットを返す

        Returns:
            imghdr形式名のセット
        """
        result: set[str] = set()
        for ext in self._allowed_extensions:
            types = _EXTENSION_TO_IMGHDR.get(ext, set())
            result.update(types)
        return result

    @staticmethod
    def _check_image_mode(mode: str) -> list[str]:
        """画像モードを検査し、非推奨モードの場合は警告を返す

        Args:
            mode: 画像のカラーモード

        Returns:
            警告メッセージのリスト
        """
        warnings: list[str] = []
        if mode not in _RECOMMENDED_MODES:
            warnings.append(
                f"非推奨の画像モードです: {mode}。"
                f"推奨モード: {', '.join(sorted(_RECOMMENDED_MODES))}"
            )
        return warnings
