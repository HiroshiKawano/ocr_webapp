"""OCR処理モジュール

PaddleOCR を使用した画像からのテキスト抽出機能を提供する。
対応言語: 日本語（デフォルト）
対応入力: ファイルパス（str）、バイナリデータ（bytes）、PIL Image

セキュリティ:
    FileValidator による入力検証を統合し、パストラバーサル防御、
    MIMEタイプ検証、ファイルサイズ制限、画像解像度制限を提供する。
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

# PaddleOCR は OCRProcessor.__init__ 内で遅延インポートする
# （モジュールインポート時のモデル初期化を防ぎ、Streamlit Cloudの起動を高速化）

# 相対インポートを使用（Streamlit Cloud互換性のため）
from ..validators import FileValidator, ValidationError

logger = logging.getLogger(__name__)

# サポートする画像形式
SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# デフォルトのパディングサイズ（ピクセル）
# 画像の端にあるテキストがバウンディングボックスの検出範囲外になる問題を防ぐ
DEFAULT_PADDING_SIZE = 80


class OCRProcessingError(Exception):
    """OCR処理中に発生したエラーを表すカスタム例外"""


class OCRProcessor:
    """PaddleOCR を使用した OCR 処理クラス

    画像からテキストを抽出し、シンプルモード（プレーンテキスト）
    または詳細モード（構造化データ）で結果を返す。
    FileValidator による入力検証をサポートし、セキュリティを強化する。

    画像の端にあるテキストの検出精度を向上させるため、
    OCR処理前に画像にパディング（余白）を追加する前処理をサポートする。

    Attributes:
        _engine: PaddleOCR エンジンインスタンス
        _validator: ファイル検証器（FileValidator）
        _add_padding: パディング前処理の有効/無効フラグ
        _padding_size: パディングサイズ（ピクセル）
    """

    def __init__(
        self,
        *,
        lang: str = "japan",
        validator: FileValidator | None = None,
        add_padding: bool = True,
        padding_size: int = DEFAULT_PADDING_SIZE,
    ) -> None:
        """OCRProcessor を初期化する

        Args:
            lang: OCR で使用する言語。デフォルトは 'japan'（日本語）
            validator: ファイル検証器（オプション）。
                Noneの場合はデフォルト設定のFileValidatorを生成する。
                デフォルト設定は後方互換性を保持するため制限が緩い。
            add_padding: 画像にパディング（余白）を追加するかどうか。
                デフォルトは True。画像の端にあるテキストの検出精度を
                向上させる。
            padding_size: パディングのサイズ（ピクセル）。
                デフォルトは DEFAULT_PADDING_SIZE。各辺に指定サイズの
                白い余白を追加する。
        """
        # PaddleOCR を遅延インポート（モジュール読み込み時のモデル初期化を回避）
        from paddleocr import PaddleOCR

        try:
            # PaddleOCR 3.x（PP-OCRv5）:
            # - use_textline_orientation=False: テキスト行方向モデルを省略（メモリ節約）
            # - use_doc_orientation_classify=False: 文書方向分類モデルを無効化
            # - use_doc_unwarping=False: 文書歪み補正モデルを無効化
            # - enable_mkldnn=False: oneDNN無効化（PIR互換性問題を回避）
            # これにより読み込むモデル数を5→2に削減し、メモリ消費を最小化
            # （Streamlit Cloud無料枠の~1GBメモリ制限に対応）
            self._engine = PaddleOCR(
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                lang=lang,
                enable_mkldnn=False,
            )
        except TypeError:
            # PaddleOCR 2.x: 新パラメータ未対応
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                enable_mkldnn=False,
            )

        self._add_padding = add_padding
        self._padding_size = padding_size

        # validatorが指定されていない場合はデフォルト設定を使用
        # デフォルトは既存動作を壊さないよう制限が緩い
        self._validator = validator or FileValidator(
            max_file_size=10 * 1024 * 1024,
            allowed_extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"},
            max_dimensions=(10000, 10000),
            allowed_base_dir=None,  # パス制限なし
        )

    def extract_text(
        self,
        source: Union[str, bytes, Image.Image],
        *,
        detailed: bool = False,
    ) -> Union[str, list[dict]]:
        """画像からテキストを抽出する

        検証 -> 入力準備 -> OCR実行 -> 結果フォーマット の順に処理する。

        Args:
            source: 画像ソース。以下の形式に対応:
                - str: ファイルパス
                - bytes: 画像のバイナリデータ
                - PIL.Image.Image: PIL イメージオブジェクト
            detailed: True の場合、構造化データ（テキスト、座標、信頼度）を返す

        Returns:
            detailed=False: 改行区切りのプレーンテキスト（str）
            detailed=True: 構造化データのリスト（list[dict]）
                各要素は以下のキーを持つ:
                - text: 認識されたテキスト
                - bounding_box: バウンディングボックス座標
                - confidence: 信頼度スコア（0.0 ~ 1.0）

        Raises:
            ValidationError: 入力検証に失敗した場合（およびそのサブクラス）
            OCRProcessingError: OCR処理中にエラーが発生した場合
        """
        # ステップ1: 検証（ValidationError系はそのままスロー）
        self._validator.validate(source)

        # ステップ2: 入力準備
        image_input = self._prepare_input(source)

        # ステップ3: OCR実行
        ocr_result = self._run_ocr(image_input)

        # ステップ4: 結果フォーマット
        return self._format_result(ocr_result, detailed=detailed)

    def _prepare_input(
        self, source: Union[str, bytes, Image.Image]
    ) -> Union[str, np.ndarray]:
        """入力ソースを PaddleOCR が受け付ける形式に変換する

        パディングが有効な場合、すべての入力形式をPIL Imageに変換し、
        パディングを適用してからnumpy配列に変換する。

        パディングが無効な場合は、ファイルパスはそのまま文字列で返し、
        バイナリデータとPIL Imageはnumpy配列に変換して返す。

        Args:
            source: 画像ソース

        Returns:
            PaddleOCR に渡すための画像データ（ファイルパス文字列または numpy 配列）

        Raises:
            OCRProcessingError: 入力が不正な場合
        """
        if isinstance(source, str):
            if self._add_padding:
                image = self._load_image_from_path(source)
                image = self._add_padding_to_image(image)
                return self._convert_pil_to_numpy(image)
            return self._validate_file_path(source)

        if isinstance(source, bytes):
            image = self._load_image_from_bytes(source)
        elif isinstance(source, Image.Image):
            image = source
        else:
            raise OCRProcessingError(
                f"サポートされていない入力形式です: {type(source).__name__}。"
                "str, bytes, PIL.Image.Image のいずれかを指定してください。"
            )

        # パディング適用
        if self._add_padding:
            image = self._add_padding_to_image(image)

        return self._convert_pil_to_numpy(image)

    @staticmethod
    def _validate_file_path(file_path: str) -> str:
        """ファイルパスの存在と形式を検証して返す

        Args:
            file_path: 画像ファイルのパス

        Returns:
            検証済みのファイルパス文字列

        Raises:
            OCRProcessingError: ファイルが存在しない、または形式が不正な場合
        """
        path = Path(file_path)

        if not path.exists():
            raise OCRProcessingError(
                f"ファイルが見つかりません: {file_path}"
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise OCRProcessingError(
                f"サポートされていない画像形式です: {path.suffix}。"
                f"対応形式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        return file_path

    def _load_image_from_path(self, file_path: str) -> Image.Image:
        """ファイルパスを検証し、PIL Image として読み込む

        内部で _validate_file_path を使用してパスを検証した後、
        PIL Image として読み込む。

        Args:
            file_path: 画像ファイルのパス

        Returns:
            PIL Image オブジェクト

        Raises:
            OCRProcessingError: ファイルが存在しない、形式が不正、
                または読み込みに失敗した場合
        """
        self._validate_file_path(file_path)

        try:
            return Image.open(file_path)
        except Exception as err:
            raise OCRProcessingError(
                f"画像ファイルの読み込みに失敗しました: {err}"
            ) from err

    @staticmethod
    def _load_image_from_bytes(data: bytes) -> Image.Image:
        """バイナリデータを PIL Image として読み込む

        Args:
            data: 画像のバイナリデータ

        Returns:
            PIL Image オブジェクト

        Raises:
            OCRProcessingError: 画像として読み込めない場合
        """
        try:
            return Image.open(io.BytesIO(data))
        except Exception as err:
            raise OCRProcessingError(
                f"画像データの読み込みに失敗しました: {err}"
            ) from err

    def _add_padding_to_image(self, image: Image.Image) -> Image.Image:
        """画像に白い余白（パディング）を追加する

        画像の端にあるテキストがバウンディングボックスの検出範囲外に
        なる問題を防ぐため、画像の周囲に余白を追加する。

        Args:
            image: 元の PIL Image オブジェクト

        Returns:
            パディングが追加された新しい PIL Image オブジェクト
        """
        # 画像モードに応じた白色の値を決定
        fill_color = 255 if image.mode == "L" else (255, 255, 255)

        return ImageOps.expand(
            image,
            border=self._padding_size,
            fill=fill_color,
        )

    @staticmethod
    def _convert_pil_to_numpy(image: Image.Image) -> np.ndarray:
        """PIL Image を numpy 配列に変換する

        Args:
            image: PIL イメージオブジェクト

        Returns:
            RGB形式の numpy 配列
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def _run_ocr(self, image_input: Union[str, np.ndarray]) -> list:
        """PaddleOCR エンジンで OCR を実行する

        Args:
            image_input: PaddleOCR に渡す画像データ

        Returns:
            PaddleOCR の生の結果

        Raises:
            OCRProcessingError: OCR処理中にエラーが発生した場合
        """
        try:
            return self._engine.ocr(image_input)
        except Exception as err:
            raise OCRProcessingError(
                f"OCR処理中にエラーが発生しました: {err}"
            ) from err

    @staticmethod
    def _is_new_format(result_item: object) -> bool:
        """PaddleOCR の結果が新フォーマット（辞書ライク）かどうかを判定する

        新フォーマット（v3+）: 辞書ライクなオブジェクト（rec_texts, rec_scores, rec_polys）
        旧フォーマット: リスト（[[bbox, (text, confidence)], ...]）

        Args:
            result_item: ocr_result[0] の値

        Returns:
            新フォーマットの場合は True
        """
        return isinstance(result_item, dict) or (
            hasattr(result_item, "__getitem__")
            and hasattr(result_item, "get")
            and not isinstance(result_item, (list, tuple))
        )

    @staticmethod
    def _format_new_result(
        result_item: dict,
        *,
        detailed: bool,
    ) -> Union[str, list[dict]]:
        """新しいPaddleOCR（v3+）の結果をフォーマットする

        Args:
            result_item: PaddleOCR v3+ の OCRResult（辞書ライクオブジェクト）
            detailed: 詳細モードフラグ

        Returns:
            フォーマット済みの結果
        """
        texts = result_item.get("rec_texts", [])
        scores = result_item.get("rec_scores", [])
        polys = result_item.get("rec_polys", [])

        if not texts:
            return [] if detailed else ""

        if detailed:
            return [
                {
                    "text": text,
                    "bounding_box": bbox,
                    "confidence": score,
                }
                for text, score, bbox in zip(texts, scores, polys)
            ]

        return "\n".join(texts)

    @staticmethod
    def _format_legacy_result(
        detections: list,
        *,
        detailed: bool,
    ) -> Union[str, list[dict]]:
        """旧PaddleOCRの結果をフォーマットする

        Args:
            detections: 旧フォーマットの検出結果リスト [[bbox, (text, confidence)], ...]
            detailed: 詳細モードフラグ

        Returns:
            フォーマット済みの結果
        """
        if detailed:
            return [
                {
                    "text": text,
                    "bounding_box": bbox,
                    "confidence": confidence,
                }
                for bbox, (text, confidence) in detections
            ]

        lines = [text for _, (text, _) in detections]
        return "\n".join(lines)

    @staticmethod
    def _format_result(
        ocr_result: list,
        *,
        detailed: bool,
    ) -> Union[str, list[dict]]:
        """OCR結果をフォーマットする

        PaddleOCR の旧フォーマット（リスト形式）と新フォーマット（辞書ライク）の
        両方に対応する。フォーマットは最初の要素の型で自動判定する。

        旧フォーマット: [[[bbox, (text, confidence)], ...]]
        新フォーマット: [OCRResult{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}]

        Args:
            ocr_result: PaddleOCR の生の結果
            detailed: 詳細モードフラグ

        Returns:
            フォーマット済みの結果
        """
        # OCR結果が空の場合
        if not ocr_result or ocr_result[0] is None:
            return [] if detailed else ""

        first_item = ocr_result[0]

        # 新しいPaddleOCR形式（辞書ライクオブジェクト）の判定
        if OCRProcessor._is_new_format(first_item):
            return OCRProcessor._format_new_result(
                first_item, detailed=detailed
            )

        # 旧フォーマット（リスト形式）
        return OCRProcessor._format_legacy_result(
            first_item, detailed=detailed
        )
