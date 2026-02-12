"""Streamlit UI (app.py) のユニットテスト

TDD Red-Green-Refactor サイクルに従い、まずテストから記述する。

テスト戦略:
  - 純粋関数（create_history_entry）: 直接呼び出しでテスト
  - Streamlit依存関数（process_single_file等）: st をモックして直接テスト
  - UI描画関数: AppTest.from_function を使用してStreamlitコンポーネントをテスト
  - OCRProcessor, FileValidator: モックで外部依存を排除

注意:
  Streamlit の AppTest は file_uploader をサポートしていないため、
  ファイルアップロード部分はヘルパー関数の直接テストで補完する。
  AppTest.from_function はクロージャ変数を参照できないため、
  モックオブジェクトを使うテストには直接呼び出し + st.mock を使用する。
"""

from __future__ import annotations

import io
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.ocr.processor import OCRProcessingError
from src.validators.file_validator import (
    FileFormatError,
    FileSizeError,
    PathTraversalError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _create_test_image_bytes(
    fmt: str = "PNG",
    size: tuple[int, int] = (100, 50),
) -> bytes:
    """テスト用画像のバイナリデータを生成する"""
    img = Image.new("RGB", size, color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_uploaded_file(name: str, data: bytes) -> MagicMock:
    """アップロードファイルのモックオブジェクトを作成する"""
    mock_file = MagicMock()
    mock_file.name = name
    mock_file.getvalue.return_value = data
    return mock_file


# ---------------------------------------------------------------------------
# create_history_entry のユニットテスト
# ---------------------------------------------------------------------------


class TestCreateHistoryEntry:
    """処理履歴エントリ作成関数のテスト"""

    def test_成功エントリの作成(self):
        """成功ステータスのエントリが正しく作成される"""
        from src.ui.app import create_history_entry

        entry = create_history_entry(
            filename="test.png",
            status="success",
        )

        assert entry["filename"] == "test.png"
        assert entry["status"] == "success"
        assert "timestamp" in entry
        assert "error_message" not in entry

    def test_エラーエントリの作成(self):
        """エラーステータスのエントリにエラーメッセージが含まれる"""
        from src.ui.app import create_history_entry

        entry = create_history_entry(
            filename="bad.png",
            status="error",
            error_message="ファイルが不正です",
        )

        assert entry["filename"] == "bad.png"
        assert entry["status"] == "error"
        assert entry["error_message"] == "ファイルが不正です"
        assert "timestamp" in entry

    def test_タイムスタンプの形式(self):
        """タイムスタンプが YYYY-MM-DD HH:MM:SS 形式である"""
        from src.ui.app import create_history_entry

        entry = create_history_entry(
            filename="test.png",
            status="success",
        )

        parsed = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
        assert parsed is not None

    def test_エラーメッセージ空文字ではerror_messageキーが含まれない(self):
        """error_messageが空文字の場合はキーが含まれない"""
        from src.ui.app import create_history_entry

        entry = create_history_entry(
            filename="test.png",
            status="success",
            error_message="",
        )

        assert "error_message" not in entry

    def test_エントリはイミュータブルに作成される(self):
        """create_history_entryは毎回新しい辞書を返す"""
        from src.ui.app import create_history_entry

        entry1 = create_history_entry(filename="a.png", status="success")
        entry2 = create_history_entry(filename="b.png", status="success")

        assert entry1 is not entry2
        assert entry1["filename"] != entry2["filename"]


# ---------------------------------------------------------------------------
# process_single_file のユニットテスト
# ---------------------------------------------------------------------------


class TestProcessSingleFile:
    """単一ファイルOCR処理関数のテスト

    process_single_file は st.error() を呼ぶため、
    st.error をモックして直接テストする。
    """

    def test_成功時にテキストと成功エントリを返す(self):
        """OCR処理が成功した場合、テキストと成功ステータスのエントリを返す"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "抽出されたテキスト"

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"fake_image_data", "test.png"
            )

        assert text == "抽出されたテキスト"
        assert entry["status"] == "success"
        assert entry["filename"] == "test.png"
        # エラーは呼ばれない
        mock_st.error.assert_not_called()

    def test_ValidationError時に空テキストとエラーエントリを返す(self):
        """ValidationError発生時、空テキストとエラーエントリを返す"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = ValidationError(
            "検証エラーメッセージ"
        )

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "bad.png"
            )

        assert text == ""
        assert entry["status"] == "error"
        assert "検証エラーメッセージ" in entry["error_message"]
        mock_st.error.assert_called_once()

    def test_OCRProcessingError時に空テキストとエラーエントリを返す(self):
        """OCRProcessingError発生時、空テキストとエラーエントリを返す"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = OCRProcessingError(
            "OCR処理失敗"
        )

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "fail.png"
            )

        assert text == ""
        assert entry["status"] == "error"
        mock_st.error.assert_called_once()

    def test_FileSizeError時にエラーメッセージが表示される(self):
        """FileSizeError発生時にst.errorが呼ばれる"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = FileSizeError(
            "ファイルサイズが制限を超えています"
        )

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "large.png"
            )

        assert text == ""
        assert entry["status"] == "error"
        # st.errorの引数にファイルサイズメッセージが含まれる
        error_arg = mock_st.error.call_args[0][0]
        assert "ファイルサイズ" in error_arg

    def test_FileFormatError時にエラーメッセージが表示される(self):
        """FileFormatError発生時にst.errorが呼ばれる"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = FileFormatError(
            "許可されていないファイル拡張子です"
        )

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "test.gif"
            )

        assert text == ""
        error_arg = mock_st.error.call_args[0][0]
        assert "ファイル拡張子" in error_arg

    def test_PathTraversalError時にエラーメッセージが表示される(self):
        """PathTraversalError発生時にst.errorが呼ばれる"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = PathTraversalError(
            "許可されていないパスへのアクセス"
        )

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "../../etc/passwd"
            )

        assert text == ""
        assert entry["status"] == "error"
        error_arg = mock_st.error.call_args[0][0]
        assert "パス" in error_arg

    def test_予期しない例外時にも適切にハンドリングされる(self):
        """RuntimeError等の予期しない例外でもクラッシュせずエラーを返す"""
        from src.ui.app import process_single_file

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = RuntimeError("予期しないエラー")

        with patch("src.ui.app.st") as mock_st:
            text, entry = process_single_file(
                mock_processor, b"data", "test.png"
            )

        assert text == ""
        assert entry["status"] == "error"
        error_arg = mock_st.error.call_args[0][0]
        assert "予期しない" in error_arg


# ---------------------------------------------------------------------------
# process_uploaded_files のユニットテスト
# ---------------------------------------------------------------------------


class TestProcessUploadedFiles:
    """複数ファイル一括処理関数のテスト"""

    def test_単一ファイルの処理(self):
        """1ファイルのOCR処理が正しく実行される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "テスト結果"

        mock_file = _make_uploaded_file("single.png", b"img_data")

        # session_stateのモック
        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [mock_file])

        assert "テスト結果" in mock_session["extracted_text"]
        assert "single.png" in mock_session["extracted_text"]
        assert len(mock_session["processing_history"]) == 1
        assert mock_session["processing_history"][0]["status"] == "success"

    def test_複数ファイルの処理(self):
        """複数ファイルが全て処理され結果が結合される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = [
            "ファイル1のテキスト",
            "ファイル2のテキスト",
        ]

        files = [
            _make_uploaded_file("file1.png", b"data1"),
            _make_uploaded_file("file2.png", b"data2"),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        text = mock_session["extracted_text"]
        assert "ファイル1のテキスト" in text
        assert "ファイル2のテキスト" in text
        assert "file1.png" in text
        assert "file2.png" in text
        assert len(mock_session["processing_history"]) == 2

    def test_複数ファイルで一部エラーでも残りは処理される(self):
        """一部ファイルでエラーが発生しても他のファイルは処理される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = [
            "成功テキスト",
            FileFormatError("不正な形式"),
        ]

        files = [
            _make_uploaded_file("good.png", b"data1"),
            _make_uploaded_file("bad.gif", b"data2"),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        text = mock_session["extracted_text"]
        assert "成功テキスト" in text

        history = mock_session["processing_history"]
        statuses = [h["status"] for h in history]
        assert "success" in statuses
        assert "error" in statuses

    def test_全ファイルエラー時はテキストが空(self):
        """全ファイルでエラーの場合、extracted_textは空"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = [
            OCRProcessingError("エラー1"),
            OCRProcessingError("エラー2"),
        ]

        files = [
            _make_uploaded_file("err1.png", b"data1"),
            _make_uploaded_file("err2.png", b"data2"),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        assert mock_session["extracted_text"] == ""
        history = mock_session["processing_history"]
        assert all(h["status"] == "error" for h in history)

    def test_バイナリデータがOCRProcessorに渡される(self):
        """アップロードファイルのバイナリデータがextract_textに渡される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "結果"

        test_data = b"test_binary_data"
        mock_file = _make_uploaded_file("test.png", test_data)

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [mock_file])

        mock_processor.extract_text.assert_called_once_with(test_data)

    def test_結果がファイル名付きセパレータで区切られる(self):
        """各ファイルの結果が --- filename --- 形式で区切られる"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = ["テキストA", "テキストB"]

        files = [
            _make_uploaded_file("a.png", b"d1"),
            _make_uploaded_file("b.png", b"d2"),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        text = mock_session["extracted_text"]
        assert "--- a.png ---" in text
        assert "--- b.png ---" in text


# ---------------------------------------------------------------------------
# UIコンポーネント描画のテスト（AppTest.from_function 使用）
# ---------------------------------------------------------------------------


class TestUIRendering:
    """UI描画関数のテスト"""

    def test_ヘッダーが表示される(self):
        """ページタイトルにOCRが含まれる"""
        def _run():
            from src.ui.app import render_header
            render_header()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.title) > 0
        assert "OCR" in at.title[0].value

    def test_OCR実行ボタンが表示される(self):
        """OCR実行ボタンが表示される"""
        def _run():
            from src.ui.app import render_ocr_button
            render_ocr_button()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.button) > 0
        button_labels = [b.label for b in at.button]
        assert any("OCR" in label for label in button_labels)

    def test_テキストエリアが表示される(self):
        """抽出テキスト表示用テキストエリアが描画される"""
        def _run():
            import streamlit as st
            st.session_state["extracted_text"] = "サンプルテキスト"
            from src.ui.app import render_extracted_text
            render_extracted_text()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.text_area) > 0
        assert at.text_area[0].value == "サンプルテキスト"

    def test_テキストエリアが空の初期状態(self):
        """初期状態ではテキストエリアが空"""
        def _run():
            import streamlit as st
            st.session_state["extracted_text"] = ""
            from src.ui.app import render_extracted_text
            render_extracted_text()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.text_area) > 0
        assert at.text_area[0].value == ""

    def test_ファイルアップローダーが描画される(self):
        """render_file_uploaderがエラーなく描画される"""
        def _run():
            from src.ui.app import render_file_uploader
            render_file_uploader()

        at = AppTest.from_function(_run)
        at.run()

        # file_uploader は AppTest では UnknownElement として扱われる
        # エラーなく描画されることを確認
        assert not at.exception


class TestDownloadButton:
    """ダウンロード機能のテスト"""

    def test_テキストが空の場合ダウンロードボタンは非表示(self):
        """extracted_textが空の場合、download_buttonが描画されない"""
        def _run():
            import streamlit as st
            st.session_state["extracted_text"] = ""
            from src.ui.app import render_download_button
            render_download_button()

        at = AppTest.from_function(_run)
        at.run()

        # ダウンロードボタンは表示されない
        assert len(at.button) == 0

    def test_テキストがある場合ダウンロードボタンが表示される(self):
        """extracted_textが存在する場合、download_buttonが描画される"""
        def _run():
            import streamlit as st
            st.session_state["extracted_text"] = "OCR結果テキスト"
            from src.ui.app import render_download_button
            render_download_button()

        at = AppTest.from_function(_run)
        at.run()

        # エラーなく描画されることを確認
        assert not at.exception


class TestProcessingHistoryRendering:
    """処理履歴描画のテスト"""

    def test_履歴が空の場合インフォメッセージが表示される(self):
        """処理履歴が空の場合、「まだ処理履歴がありません」が表示される"""
        def _run():
            import streamlit as st
            st.session_state["processing_history"] = []
            from src.ui.app import render_processing_history
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.info) > 0
        assert any("処理履歴" in i.value for i in at.info)

    def test_成功履歴の表示(self):
        """成功した処理の履歴が正しく表示される"""
        def _run():
            import streamlit as st
            st.session_state["processing_history"] = [
                {
                    "filename": "test.png",
                    "status": "success",
                    "timestamp": "2025-01-01 12:00:00",
                },
            ]
            from src.ui.app import render_processing_history
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        assert len(at.subheader) > 0
        # st.write は markdown として出力される
        all_text = " ".join(m.value for m in at.markdown)
        assert "test.png" in all_text
        assert "成功" in all_text

    def test_エラー履歴の表示(self):
        """エラーの処理履歴にエラーメッセージが含まれる"""
        def _run():
            import streamlit as st
            st.session_state["processing_history"] = [
                {
                    "filename": "error.png",
                    "status": "error",
                    "timestamp": "2025-01-01 12:00:00",
                    "error_message": "テスト用エラーメッセージ",
                },
            ]
            from src.ui.app import render_processing_history
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        all_text = " ".join(m.value for m in at.markdown)
        assert "error.png" in all_text
        assert "エラー" in all_text

    def test_複数履歴が新しい順に表示される(self):
        """複数の処理履歴が新しいものから順に表示される"""
        def _run():
            import streamlit as st
            st.session_state["processing_history"] = [
                {
                    "filename": "first.png",
                    "status": "success",
                    "timestamp": "2025-01-01 12:00:00",
                },
                {
                    "filename": "second.png",
                    "status": "success",
                    "timestamp": "2025-01-01 12:01:00",
                },
            ]
            from src.ui.app import render_processing_history
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        all_texts = [m.value for m in at.markdown]
        # 新しいものが先に表示される（reversed）
        second_idx = None
        first_idx = None
        for i, t in enumerate(all_texts):
            if "second.png" in t:
                second_idx = i
            if "first.png" in t:
                first_idx = i

        assert second_idx is not None
        assert first_idx is not None
        assert second_idx < first_idx


class TestSessionStateInitialization:
    """セッション状態の初期化テスト"""

    def test_セッション状態が正しく初期化される(self):
        """初回起動時にsession_stateが正しいキーと初期値で初期化される"""
        def _run():
            from src.ui.app import initialize_session_state
            initialize_session_state()

        at = AppTest.from_function(_run)
        at.run()

        assert "extracted_text" in at.session_state
        assert "processing_history" in at.session_state
        assert at.session_state["extracted_text"] == ""
        assert at.session_state["processing_history"] == []

    def test_セッション状態が再実行で上書きされない(self):
        """既にセッション状態が設定されている場合、再実行で上書きされない"""
        def _run():
            from src.ui.app import initialize_session_state
            initialize_session_state()

        at = AppTest.from_function(_run)
        at.run()

        # 値を変更
        at.session_state["extracted_text"] = "テスト結果"
        at.run()

        # 上書きされていない
        assert at.session_state["extracted_text"] == "テスト結果"


class TestMainFlowWithButton:
    """ボタンクリックを含むメインフローのテスト（AppTest使用）"""

    def test_OCR実行ボタン押下時にファイル未選択でエラー表示(self):
        """ファイルなしでOCR実行ボタンを押すとエラーメッセージが表示される"""
        def _run():
            import streamlit as st
            from src.ui.app import initialize_session_state, render_header
            from src.ui.app import render_ocr_button, render_extracted_text
            from src.ui.app import render_download_button
            from src.ui.app import render_processing_history

            initialize_session_state()
            render_header()

            # file_uploaderの代わりに空リスト
            uploaded_files = []

            if render_ocr_button():
                if not uploaded_files:
                    st.error("画像ファイルをアップロードしてください。")

            render_extracted_text()
            render_download_button()
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        # ボタンをクリック
        ocr_buttons = [b for b in at.button if "OCR" in b.label]
        assert len(ocr_buttons) > 0
        ocr_buttons[0].click().run()

        # エラーメッセージ表示を確認
        error_messages = [e.value for e in at.error]
        assert any("アップロード" in msg for msg in error_messages)

    def test_初期表示でエラーがない(self):
        """アプリ初期表示時にエラーが発生しないことを確認"""
        def _run():
            import streamlit as st
            from src.ui.app import initialize_session_state, render_header
            from src.ui.app import render_ocr_button, render_extracted_text
            from src.ui.app import render_download_button
            from src.ui.app import render_processing_history

            initialize_session_state()
            render_header()
            render_ocr_button()
            render_extracted_text()
            render_download_button()
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        # エラーがない
        assert len(at.error) == 0
        assert not at.exception
        # タイトルが表示される
        assert len(at.title) > 0
        # テキストエリアが表示される
        assert len(at.text_area) > 0
