"""OCRProcessor のユニットテスト

TDD Red-Green-Refactor サイクルに従い、まずテストから記述する。
PaddleOCR はモックを使用し、外部依存を排除する。

注意:
    このテストファイルはOCRProcessor単体のユニットテストであるため、
    FileValidatorの検証はモックでバイパスしている。
    FileValidator統合の検証は tests/integration/test_ocr_with_validation.py で行う。
"""

import io
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

from src.ocr.processor import OCRProcessor, OCRProcessingError
from src.validators.file_validator import ValidationResult


# --- テスト用フィクスチャ ---


# FileValidatorのvalidateが返す「常に成功」の結果
_VALID_RESULT = ValidationResult(
    is_valid=True,
    errors=[],
    warnings=[],
    file_info={},
)


@pytest.fixture
def mock_paddleocr_result():
    """PaddleOCR の典型的な戻り値をモックする"""
    return [
        [
            [
                [[100, 50], [300, 50], [300, 80], [100, 80]],
                ("こんにちは世界", 0.95),
            ],
            [
                [[100, 100], [400, 100], [400, 130], [100, 130]],
                ("テスト文字列です", 0.88),
            ],
        ]
    ]


@pytest.fixture
def mock_paddleocr_empty_result():
    """OCR結果が空の場合の戻り値"""
    return [None]


@pytest.fixture
def sample_pil_image():
    """テスト用のPIL Imageを生成する"""
    return Image.new("RGB", (200, 100), color=(255, 255, 255))


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """テスト用の画像バイナリデータを生成する"""
    buffer = io.BytesIO()
    sample_pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def processor():
    """モック済みの OCRProcessor インスタンスを返す

    FileValidatorのvalidateメソッドをモックし、検証をバイパスする。
    OCRProcessor単体の動作テストに集中するため。
    パディングはデフォルト無効（パディングのテストは専用クラスで行う）。
    """
    with patch("src.ocr.processor.PaddleOCR") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        proc = OCRProcessor(lang="japan", add_padding=False)
        # FileValidatorの検証をバイパス（常に成功を返す）
        proc._validator = MagicMock()
        proc._validator.validate.return_value = _VALID_RESULT
        yield proc


# --- ユニットテスト ---


class TestOCRProcessorInit:
    """OCRProcessor の初期化テスト"""

    def test_デフォルトで日本語モデルを使用する(self):
        """PaddleOCR が日本語設定で初期化されることを確認"""
        with patch("src.ocr.processor.PaddleOCR") as mock_cls:
            OCRProcessor()
            mock_cls.assert_called_once()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["lang"] == "japan"

    def test_言語を指定して初期化できる(self):
        """明示的に言語を指定できることを確認"""
        with patch("src.ocr.processor.PaddleOCR") as mock_cls:
            OCRProcessor(lang="en")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["lang"] == "en"


class TestOCRProcessorSimpleMode:
    """シンプルモード（プレーンテキスト出力）のテスト"""

    def test_ファイルパスからOCR処理_シンプルモード(
        self, processor, mock_paddleocr_result
    ):
        """ファイルパスを渡してプレーンテキストを取得できる"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png")

        assert isinstance(result, str)
        assert "こんにちは世界" in result
        assert "テスト文字列です" in result

    def test_バイナリデータからOCR処理(
        self, processor, mock_paddleocr_result, sample_image_bytes
    ):
        """バイナリデータを渡してテキストを取得できる"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        result = processor.extract_text(sample_image_bytes)

        assert isinstance(result, str)
        assert "こんにちは世界" in result

    def test_PIL_ImageからOCR処理(
        self, processor, mock_paddleocr_result, sample_pil_image
    ):
        """PIL Image を渡してテキストを取得できる"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        result = processor.extract_text(sample_pil_image)

        assert isinstance(result, str)
        assert "こんにちは世界" in result

    def test_テキストは改行で結合される(
        self, processor, mock_paddleocr_result
    ):
        """複数行のテキストが改行で結合されることを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png")

        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "こんにちは世界"
        assert lines[1] == "テスト文字列です"


class TestOCRProcessorDetailedMode:
    """詳細モード（構造化データ出力）のテスト"""

    def test_ファイルパスからOCR処理_詳細モード(
        self, processor, mock_paddleocr_result
    ):
        """詳細モードで構造化データを取得できる"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png", detailed=True)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_詳細モードの構造化データにテキストが含まれる(
        self, processor, mock_paddleocr_result
    ):
        """各要素にテキスト情報が含まれることを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png", detailed=True)

        first_item = result[0]
        assert "text" in first_item
        assert first_item["text"] == "こんにちは世界"

    def test_詳細モードの構造化データにバウンディングボックスが含まれる(
        self, processor, mock_paddleocr_result
    ):
        """各要素にバウンディングボックス座標が含まれることを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png", detailed=True)

        first_item = result[0]
        assert "bounding_box" in first_item
        assert first_item["bounding_box"] == [
            [100, 50],
            [300, 50],
            [300, 80],
            [100, 80],
        ]

    def test_詳細モードの構造化データに信頼度スコアが含まれる(
        self, processor, mock_paddleocr_result
    ):
        """各要素に信頼度スコアが含まれることを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png", detailed=True)

        first_item = result[0]
        assert "confidence" in first_item
        assert first_item["confidence"] == pytest.approx(0.95)


class TestOCRProcessorErrorHandling:
    """エラーハンドリングのテスト"""

    def test_不正なファイルパスでエラーが発生する(self, processor):
        """存在しないファイルパスを渡すとエラーが発生する"""
        with pytest.raises(OCRProcessingError, match="ファイルが見つかりません"):
            processor.extract_text("/nonexistent/path/image.png")

    def test_サポートされていない形式でエラーが発生する(self, processor):
        """サポートされていないファイル形式でエラーが発生する"""
        with patch("src.ocr.processor.Path.exists", return_value=True):
            with pytest.raises(
                OCRProcessingError, match="サポートされていない画像形式"
            ):
                processor.extract_text("/path/to/document.pdf")

    def test_不正な入力型でエラーが発生する(self, processor):
        """サポートされていない型を渡すとエラーが発生する"""
        with pytest.raises(OCRProcessingError, match="サポートされていない入力形式"):
            processor.extract_text(12345)

    def test_空の画像の処理(self, processor, mock_paddleocr_empty_result):
        """空の画像（OCR結果なし）を処理した場合、空文字を返す"""
        processor._engine.ocr.return_value = mock_paddleocr_empty_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/empty.png")

        assert result == ""

    def test_空の画像の詳細モード処理(self, processor, mock_paddleocr_empty_result):
        """空の画像を詳細モードで処理した場合、空リストを返す"""
        processor._engine.ocr.return_value = mock_paddleocr_empty_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/empty.png", detailed=True)

        assert result == []

    def test_OCRエンジンのエラーが適切にラップされる(self, processor):
        """PaddleOCR 内部のエラーが OCRProcessingError に変換される"""
        processor._engine.ocr.side_effect = RuntimeError("内部エラー")

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            with pytest.raises(OCRProcessingError, match="OCR処理中にエラー"):
                processor.extract_text("/path/to/image.png")

    def test_不正なバイナリデータでエラーが発生する(self, processor):
        """画像として解読できないバイナリデータでエラーが発生する"""
        invalid_bytes = b"this is not an image"

        with pytest.raises(OCRProcessingError, match="画像データの読み込みに失敗"):
            processor.extract_text(invalid_bytes)


class TestOCRProcessorJapanese:
    """日本語テキスト認識のテスト"""

    def test_日本語テキストの正確な認識(self, processor):
        """日本語テキストが正確に認識されることを確認"""
        japanese_result = [
            [
                [
                    [[10, 10], [200, 10], [200, 40], [10, 40]],
                    ("請求書", 0.97),
                ],
                [
                    [[10, 50], [300, 50], [300, 80], [10, 80]],
                    ("合計金額: ¥12,345", 0.93),
                ],
                [
                    [[10, 90], [250, 90], [250, 120], [10, 120]],
                    ("お支払い期限: 2025年3月31日", 0.91),
                ],
            ]
        ]
        processor._engine.ocr.return_value = japanese_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/invoice.png")

        assert "請求書" in result
        assert "合計金額: ¥12,345" in result
        assert "お支払い期限: 2025年3月31日" in result

    def test_日本語テキストの詳細モード認識(self, processor):
        """日本語テキストが詳細モードで正確に構造化されることを確認"""
        japanese_result = [
            [
                [
                    [[10, 10], [200, 10], [200, 40], [10, 40]],
                    ("東京都渋谷区", 0.96),
                ],
            ]
        ]
        processor._engine.ocr.return_value = japanese_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/address.png", detailed=True)

        assert len(result) == 1
        assert result[0]["text"] == "東京都渋谷区"
        assert result[0]["confidence"] == pytest.approx(0.96)


class TestOCRProcessorNewPaddleOCRFormat:
    """新しいPaddleOCR（v3+）の結果フォーマット対応テスト

    PaddleOCR v3+ では ocr() メソッドが OCRResult オブジェクト
    （辞書ライク）のリストを返す。各 OCRResult は以下のキーを持つ:
      - rec_texts: テキストのリスト
      - rec_scores: スコアのリスト
      - rec_polys: バウンディングボックスのリスト

    旧フォーマット: [[[bbox, (text, confidence)], ...]]
    新フォーマット: [OCRResult{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}]
    """

    @pytest.fixture
    def new_format_result(self):
        """新しいPaddleOCR形式の結果を返す（辞書ライクなオブジェクト）"""
        return [
            {
                "rec_texts": ["12345", "67890"],
                "rec_scores": [0.98, 0.95],
                "rec_polys": [
                    [[10, 10], [100, 10], [100, 40], [10, 40]],
                    [[10, 50], [100, 50], [100, 80], [10, 80]],
                ],
            }
        ]

    @pytest.fixture
    def new_format_single_line(self):
        """新しいPaddleOCR形式の単一行結果"""
        return [
            {
                "rec_texts": ["テスト"],
                "rec_scores": [0.99],
                "rec_polys": [
                    [[10, 10], [100, 10], [100, 40], [10, 40]],
                ],
            }
        ]

    @pytest.fixture
    def new_format_empty(self):
        """新しいPaddleOCR形式の空結果"""
        return [
            {
                "rec_texts": [],
                "rec_scores": [],
                "rec_polys": [],
            }
        ]

    @pytest.fixture
    def new_format_multiline_numbers(self):
        """新しいPaddleOCR形式の複数行数字画像（バグ再現用）

        このケースが「too many values to unpack」エラーの原因。
        複数行の数字を含むJPG画像で発生する。
        """
        return [
            {
                "rec_texts": ["100", "200", "300", "400", "500"],
                "rec_scores": [0.97, 0.96, 0.95, 0.94, 0.93],
                "rec_polys": [
                    [[10, 10], [80, 10], [80, 30], [10, 30]],
                    [[10, 40], [80, 40], [80, 60], [10, 60]],
                    [[10, 70], [80, 70], [80, 90], [10, 90]],
                    [[10, 100], [80, 100], [80, 120], [10, 120]],
                    [[10, 130], [80, 130], [80, 150], [10, 150]],
                ],
            }
        ]

    def test_新フォーマットの複数行数字でシンプルモード(
        self, processor, new_format_multiline_numbers
    ):
        """新しいPaddleOCR形式の複数行数字画像でプレーンテキストを取得できる

        このテストが「too many values to unpack」バグを再現する。
        """
        processor._engine.ocr.return_value = new_format_multiline_numbers

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".jpg",
             ):
            result = processor.extract_text("/path/to/numbers.jpg")

        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) == 5
        assert lines[0] == "100"
        assert lines[1] == "200"
        assert lines[2] == "300"
        assert lines[3] == "400"
        assert lines[4] == "500"

    def test_新フォーマットの複数行数字で詳細モード(
        self, processor, new_format_multiline_numbers
    ):
        """新しいPaddleOCR形式の複数行数字画像で構造化データを取得できる"""
        processor._engine.ocr.return_value = new_format_multiline_numbers

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".jpg",
             ):
            result = processor.extract_text(
                "/path/to/numbers.jpg", detailed=True
            )

        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0]["text"] == "100"
        assert result[0]["confidence"] == pytest.approx(0.97)
        assert result[0]["bounding_box"] == [
            [10, 10], [80, 10], [80, 30], [10, 30]
        ]
        assert result[4]["text"] == "500"
        assert result[4]["confidence"] == pytest.approx(0.93)

    def test_新フォーマットのシンプルモード(
        self, processor, new_format_result
    ):
        """新しいPaddleOCR形式でプレーンテキストを取得できる"""
        processor._engine.ocr.return_value = new_format_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png")

        assert isinstance(result, str)
        assert "12345" in result
        assert "67890" in result

    def test_新フォーマットの詳細モード(
        self, processor, new_format_result
    ):
        """新しいPaddleOCR形式で構造化データを取得できる"""
        processor._engine.ocr.return_value = new_format_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text(
                "/path/to/image.png", detailed=True
            )

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["text"] == "12345"
        assert result[0]["confidence"] == pytest.approx(0.98)
        assert result[0]["bounding_box"] == [
            [10, 10], [100, 10], [100, 40], [10, 40]
        ]
        assert result[1]["text"] == "67890"
        assert result[1]["confidence"] == pytest.approx(0.95)

    def test_新フォーマットの単一行(
        self, processor, new_format_single_line
    ):
        """新しいPaddleOCR形式の単一行結果を処理できる"""
        processor._engine.ocr.return_value = new_format_single_line

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png")

        assert result == "テスト"

    def test_新フォーマットの空結果_シンプルモード(
        self, processor, new_format_empty
    ):
        """新しいPaddleOCR形式の空結果でシンプルモードが空文字を返す"""
        processor._engine.ocr.return_value = new_format_empty

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/empty.png")

        assert result == ""

    def test_新フォーマットの空結果_詳細モード(
        self, processor, new_format_empty
    ):
        """新しいPaddleOCR形式の空結果で詳細モードが空リストを返す"""
        processor._engine.ocr.return_value = new_format_empty

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text(
                "/path/to/empty.png", detailed=True
            )

        assert result == []

    def test_新フォーマットでバイナリデータから処理(
        self, processor, new_format_result, sample_image_bytes
    ):
        """新しいPaddleOCR形式でバイナリデータからテキストを取得できる"""
        processor._engine.ocr.return_value = new_format_result

        result = processor.extract_text(sample_image_bytes)

        assert isinstance(result, str)
        assert "12345" in result

    def test_新フォーマットでPIL_Imageから処理(
        self, processor, new_format_result, sample_pil_image
    ):
        """新しいPaddleOCR形式でPIL Imageからテキストを取得できる"""
        processor._engine.ocr.return_value = new_format_result

        result = processor.extract_text(sample_pil_image)

        assert isinstance(result, str)
        assert "12345" in result

    def test_旧フォーマットも引き続き動作する(
        self, processor, mock_paddleocr_result
    ):
        """旧フォーマット（後方互換性）が引き続き動作することを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ):
            result = processor.extract_text("/path/to/image.png")

        assert isinstance(result, str)
        assert "こんにちは世界" in result
        assert "テスト文字列です" in result


class TestOCRProcessorSupportedFormats:
    """対応画像形式のテスト"""

    @pytest.mark.parametrize(
        "extension",
        [".jpg", ".jpeg", ".png", ".bmp"],
    )
    def test_サポートされている形式が受け入れられる(
        self, processor, mock_paddleocr_result, extension
    ):
        """jpg, jpeg, png, bmp 形式が受け入れられることを確認"""
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=extension,
             ):
            result = processor.extract_text(f"/path/to/image{extension}")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "extension",
        [".pdf", ".gif", ".svg", ".tiff", ".webp"],
    )
    def test_サポートされていない形式が拒否される(self, processor, extension):
        """pdf, gif, svg, tiff, webp 形式が拒否されることを確認"""
        with patch("src.ocr.processor.Path.exists", return_value=True):
            with pytest.raises(
                OCRProcessingError, match="サポートされていない画像形式"
            ):
                processor.extract_text(f"/path/to/image{extension}")


class TestOCRProcessorPadding:
    """画像パディング前処理のテスト

    画像の端にあるテキストがバウンディングボックスの検出範囲外に
    なる問題を防ぐため、画像に余白を追加する前処理をテストする。
    """

    def test_デフォルトでパディングが有効(self):
        """OCRProcessor のデフォルト設定でパディングが有効であることを確認"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor()
            assert proc._add_padding is True

    def test_パディングを無効化できる(self):
        """add_padding=False でパディングを無効化できることを確認"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(add_padding=False)
            assert proc._add_padding is False

    def test_パディングサイズを指定できる(self):
        """padding_size でパディングサイズを指定できることを確認"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(padding_size=50)
            assert proc._padding_size == 50

    def test_デフォルトパディングサイズ(self):
        """デフォルトのパディングサイズが適切に設定されていることを確認"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor()
            assert proc._padding_size > 0

    def test_PIL_Imageにパディングが適用される(
        self, processor, mock_paddleocr_result, sample_pil_image
    ):
        """PIL Image 入力時にパディングが適用されることを確認"""
        processor._add_padding = True
        processor._padding_size = 30
        processor._engine.ocr.return_value = mock_paddleocr_result

        # _add_padding_to_image が呼ばれることを確認
        with patch.object(
            processor, "_add_padding_to_image", wraps=processor._add_padding_to_image
        ) as mock_pad:
            processor.extract_text(sample_pil_image)
            mock_pad.assert_called_once()

    def test_バイナリデータにパディングが適用される(
        self, processor, mock_paddleocr_result, sample_image_bytes
    ):
        """バイナリデータ入力時にパディングが適用されることを確認"""
        processor._add_padding = True
        processor._padding_size = 30
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch.object(
            processor, "_add_padding_to_image", wraps=processor._add_padding_to_image
        ) as mock_pad:
            processor.extract_text(sample_image_bytes)
            mock_pad.assert_called_once()

    def test_ファイルパスにパディングが適用される(
        self, processor, mock_paddleocr_result, sample_pil_image
    ):
        """ファイルパス入力時にパディングが適用されることを確認"""
        processor._add_padding = True
        processor._padding_size = 30
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch("src.ocr.processor.Path.exists", return_value=True), \
             patch(
                 "src.ocr.processor.Path.suffix",
                 new_callable=PropertyMock,
                 return_value=".png",
             ), \
             patch("src.ocr.processor.Image.open", return_value=sample_pil_image), \
             patch.object(
                 processor,
                 "_add_padding_to_image",
                 wraps=processor._add_padding_to_image,
             ) as mock_pad:
            processor.extract_text("/path/to/image.png")
            mock_pad.assert_called_once()

    def test_パディング無効時にパディングが適用されない(
        self, processor, mock_paddleocr_result, sample_pil_image
    ):
        """add_padding=False の場合パディングが適用されないことを確認"""
        processor._add_padding = False
        processor._engine.ocr.return_value = mock_paddleocr_result

        with patch.object(
            processor, "_add_padding_to_image"
        ) as mock_pad:
            processor.extract_text(sample_pil_image)
            mock_pad.assert_not_called()

    def test_パディング後の画像サイズが正しい(self):
        """パディング後の画像サイズが元サイズ + 2 * padding_size になる"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(padding_size=30)
            proc._validator = MagicMock()
            proc._validator.validate.return_value = _VALID_RESULT

            original = Image.new("RGB", (200, 100), color=(255, 255, 255))
            padded = proc._add_padding_to_image(original)

            assert padded.size == (260, 160)

    def test_パディングは白色で埋められる(self):
        """パディング領域が白色（255, 255, 255）で埋められることを確認"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(padding_size=10)
            proc._validator = MagicMock()
            proc._validator.validate.return_value = _VALID_RESULT

            original = Image.new("RGB", (50, 50), color=(0, 0, 0))
            padded = proc._add_padding_to_image(original)

            # 左上隅のピクセルはパディング領域なので白色
            assert padded.getpixel((0, 0)) == (255, 255, 255)
            # 右下隅のピクセルもパディング領域なので白色
            assert padded.getpixel((69, 69)) == (255, 255, 255)
            # 元画像の領域は黒色
            assert padded.getpixel((10, 10)) == (0, 0, 0)

    def test_グレースケール画像にもパディングが適用される(self):
        """グレースケール（L モード）画像にもパディングが正しく適用される"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(padding_size=10)
            proc._validator = MagicMock()
            proc._validator.validate.return_value = _VALID_RESULT

            original = Image.new("L", (50, 50), color=0)
            padded = proc._add_padding_to_image(original)

            assert padded.size == (70, 70)
            # パディング領域は白色（255）
            assert padded.getpixel((0, 0)) == 255

    def test_RGBA画像にもパディングが適用される(self):
        """RGBA 画像にもパディングが正しく適用される"""
        with patch("src.ocr.processor.PaddleOCR"):
            proc = OCRProcessor(padding_size=10)
            proc._validator = MagicMock()
            proc._validator.validate.return_value = _VALID_RESULT

            original = Image.new("RGBA", (50, 50), color=(0, 0, 0, 255))
            padded = proc._add_padding_to_image(original)

            assert padded.size == (70, 70)
