"""Streamlit UI と OCRProcessor の統合テスト

UIコンポーネントとOCR処理が正しく連携することを検証する。
OCRエンジン（PaddleOCR）自体はモックするが、
app.py のヘルパー関数とUI描画関数の統合はテストする。

テスト戦略:
  - OCR処理フローの統合: st をモックして process_uploaded_files + session_state を検証
  - UI描画の統合: AppTest.from_function でエラーフリーな描画を検証
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.ocr.processor import OCRProcessingError
from src.validators.file_validator import (
    FileFormatError,
    FileSizeError,
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
    img = Image.new("RGB", size, color=(200, 200, 200))
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
# 統合テスト: OCR処理フローの統合
# ---------------------------------------------------------------------------


class TestOCRProcessingFlowIntegration:
    """OCR処理フロー全体の統合テスト

    process_uploaded_files を中心に、
    process_single_file -> create_history_entry の連携を検証する。
    """

    def test_画像アップロードからOCR結果保存までの完全フロー(self):
        """ファイルデータ -> OCR実行 -> 結果保存 -> 履歴記録の一連の流れ"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "統合テスト結果テキスト"

        test_image = _create_test_image_bytes()
        test_file = _make_uploaded_file("integration.png", test_image)

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [test_file])

        # 結果が正しく保存される
        assert "統合テスト結果テキスト" in mock_session["extracted_text"]
        assert "integration.png" in mock_session["extracted_text"]

        # 処理履歴にエントリが追加される
        history = mock_session["processing_history"]
        assert len(history) == 1
        assert history[0]["filename"] == "integration.png"
        assert history[0]["status"] == "success"
        assert "timestamp" in history[0]

        # エラーは呼ばれない
        mock_st.error.assert_not_called()

    def test_OCRProcessorへのバイナリデータ受け渡し(self):
        """アップロードされたファイルのバイナリデータがOCRProcessorに渡される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "テスト"

        test_image_data = _create_test_image_bytes()
        test_file = _make_uploaded_file("data_pass.png", test_image_data)

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [test_file])

        # extract_textが呼ばれ、bytesデータが渡されている
        mock_processor.extract_text.assert_called_once()
        call_args = mock_processor.extract_text.call_args
        assert isinstance(call_args[0][0], bytes)
        assert call_args[0][0] == test_image_data


class TestUIErrorIntegration:
    """UIとエラーハンドリングの統合テスト"""

    def test_ValidationErrorサブクラスのエラーメッセージが日本語で表示される(self):
        """FileSizeError等のサブクラスが日本語エラーメッセージを生成する"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = FileSizeError(
            "ファイルサイズが制限を超えています: 15,000,000バイト（上限: 10,485,760バイト）"
        )

        test_file = _make_uploaded_file("big_file.png", b"data")

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [test_file])

        # エラーメッセージが日本語
        error_arg = mock_st.error.call_args[0][0]
        assert "ファイルサイズ" in error_arg

        # 処理履歴にもエラーが記録される
        history = mock_session["processing_history"]
        assert len(history) == 1
        assert history[0]["status"] == "error"
        assert "ファイルサイズ" in history[0]["error_message"]

    def test_OCR処理エラーと処理履歴の統合(self):
        """OCR処理エラー発生時、エラー表示と処理履歴更新が同時に行われる"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = OCRProcessingError(
            "OCR処理中にエラーが発生しました"
        )

        test_file = _make_uploaded_file("error_test.png", b"data")

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [test_file])

        # st.error が呼ばれている
        mock_st.error.assert_called_once()

        # 処理履歴にエラーが記録されている
        history = mock_session["processing_history"]
        assert len(history) == 1
        assert history[0]["status"] == "error"
        assert "OCR処理中" in history[0]["error_message"]

    def test_複数例外タイプの混合処理(self):
        """異なる例外タイプが混在する複数ファイル処理"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = [
            "正常テキスト",
            FileSizeError("サイズ超過"),
            FileFormatError("形式不正"),
            OCRProcessingError("OCRエラー"),
        ]

        files = [
            _make_uploaded_file("ok.png", b"d1"),
            _make_uploaded_file("big.png", b"d2"),
            _make_uploaded_file("bad.gif", b"d3"),
            _make_uploaded_file("fail.png", b"d4"),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        # 成功したファイルのテキストのみ含まれる
        assert "正常テキスト" in mock_session["extracted_text"]

        # 処理履歴に全4件が記録される
        history = mock_session["processing_history"]
        assert len(history) == 4

        statuses = [h["status"] for h in history]
        assert statuses.count("success") == 1
        assert statuses.count("error") == 3

        # st.errorが3回呼ばれている
        assert mock_st.error.call_count == 3


class TestUIMultipleFilesIntegration:
    """複数ファイル処理の統合テスト"""

    def test_複数ファイルの結果がセパレータで区切られる(self):
        """複数ファイルの結果がファイル名付きで区切られる"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.side_effect = [
            "1枚目の結果",
            "2枚目の結果",
        ]

        files = [
            _make_uploaded_file("page1.png", _create_test_image_bytes()),
            _make_uploaded_file("page2.png", _create_test_image_bytes()),
        ]

        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, files)

        text = mock_session["extracted_text"]
        assert "1枚目の結果" in text
        assert "2枚目の結果" in text
        assert "page1.png" in text
        assert "page2.png" in text

    def test_処理履歴が蓄積される(self):
        """複数回の処理で処理履歴が正しく蓄積される"""
        from src.ui.app import process_uploaded_files

        mock_processor = MagicMock()
        mock_processor.extract_text.return_value = "結果"

        # 1回目の処理
        file1 = _make_uploaded_file("first.png", b"data1")
        mock_session = {
            "extracted_text": "",
            "processing_history": [],
        }

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [file1])

        assert len(mock_session["processing_history"]) == 1

        # 2回目の処理（既存の履歴を保持）
        file2 = _make_uploaded_file("second.png", b"data2")

        with patch("src.ui.app.st") as mock_st:
            mock_st.session_state = mock_session
            process_uploaded_files(mock_processor, [file2])

        # 両方の履歴が蓄積される
        assert len(mock_session["processing_history"]) == 2
        filenames = [h["filename"] for h in mock_session["processing_history"]]
        assert "first.png" in filenames
        assert "second.png" in filenames


class TestFullUIRenderingIntegration:
    """UI描画の統合テスト（AppTest使用）"""

    def test_全コンポーネントが初期表示でエラーなく描画される(self):
        """全UIコンポーネントが初期状態でエラーなく描画される"""
        def _run():
            import streamlit as st
            from src.ui.app import (
                initialize_session_state,
                render_header,
                render_ocr_button,
                render_extracted_text,
                render_download_button,
                render_processing_history,
            )

            initialize_session_state()
            render_header()
            render_ocr_button()
            render_extracted_text()
            render_download_button()
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        assert not at.exception
        assert len(at.error) == 0
        assert len(at.title) > 0
        assert len(at.text_area) > 0
        assert len(at.button) > 0

    def test_ボタンクリック後にファイル未選択エラーが表示される(self):
        """OCR実行ボタンをクリック後、ファイル未選択エラーが表示される"""
        def _run():
            import streamlit as st
            from src.ui.app import (
                initialize_session_state,
                render_header,
                render_ocr_button,
                render_extracted_text,
                render_processing_history,
            )

            initialize_session_state()
            render_header()

            uploaded_files = []

            if render_ocr_button():
                if not uploaded_files:
                    st.error("画像ファイルをアップロードしてください。")

            render_extracted_text()
            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        ocr_buttons = [b for b in at.button if "OCR" in b.label]
        ocr_buttons[0].click().run()

        error_messages = [e.value for e in at.error]
        assert any("アップロード" in msg for msg in error_messages)

    def test_処理履歴がUI上で正しく表示される(self):
        """session_stateに履歴がある場合、UI上で正しく描画される"""
        def _run():
            import streamlit as st
            from src.ui.app import (
                initialize_session_state,
                render_processing_history,
            )

            initialize_session_state()

            # 既存の履歴を設定
            st.session_state["processing_history"] = [
                {
                    "filename": "photo1.png",
                    "status": "success",
                    "timestamp": "2025-06-01 10:00:00",
                },
                {
                    "filename": "photo2.png",
                    "status": "error",
                    "timestamp": "2025-06-01 10:01:00",
                    "error_message": "OCR処理失敗",
                },
            ]

            render_processing_history()

        at = AppTest.from_function(_run)
        at.run()

        all_text = " ".join(m.value for m in at.markdown)
        assert "photo1.png" in all_text
        assert "photo2.png" in all_text
        assert "成功" in all_text
        assert "エラー" in all_text
