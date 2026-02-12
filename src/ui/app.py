"""OCR画像テキスト抽出アプリ

Streamlit を使用した OCR Web アプリケーション。
複数画像のアップロード、OCR実行、結果表示・ダウンロード、処理履歴を提供する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PaddlePaddle 3.x の PIR + oneDNN 互換性問題を回避（Paddle importより前に設定必須）
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_pir_apply_inplace_pass"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"       # PaddlePaddle 2.x/3.0 系のフラグ名
os.environ["FLAGS_use_onednn"] = "0"       # PaddlePaddle 3.x 系の新フラグ名

# プロジェクトルートをsys.pathに追加（Streamlit Cloud対応）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime

import streamlit as st

# app.pyはStreamlitのエントリーポイントとして直接実行されるため、
# 絶対インポートを使用する必要がある（上記のsys.path設定により動作）
from src.ocr.processor import OCRProcessor, OCRProcessingError
from src.validators.file_validator import (
    FileFormatError,
    FileSizeError,
    PathTraversalError,
    ValidationError,
)


def initialize_session_state() -> None:
    """セッション状態を初期化する

    初回起動時のみデフォルト値を設定し、再実行時は既存値を保持する。
    """
    if "extracted_text" not in st.session_state:
        st.session_state["extracted_text"] = ""
    if "processing_history" not in st.session_state:
        st.session_state["processing_history"] = []


def create_history_entry(
    *,
    filename: str,
    status: str,
    error_message: str = "",
) -> dict:
    """処理履歴エントリを作成する（イミュータブル）

    Args:
        filename: 処理対象のファイル名
        status: 処理結果（"success" または "error"）
        error_message: エラー発生時のメッセージ

    Returns:
        処理履歴エントリの辞書
    """
    entry = {
        "filename": filename,
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if error_message:
        return {**entry, "error_message": error_message}
    return entry


def process_single_file(
    processor: OCRProcessor,
    file_data: bytes,
    filename: str,
) -> tuple[str, dict]:
    """単一ファイルのOCR処理を実行する

    Args:
        processor: OCRProcessorインスタンス
        file_data: ファイルのバイナリデータ
        filename: ファイル名

    Returns:
        (抽出テキスト, 処理履歴エントリ) のタプル
        エラー時は抽出テキストは空文字列
    """
    try:
        text = processor.extract_text(file_data)
        history_entry = create_history_entry(
            filename=filename,
            status="success",
        )
        return text, history_entry
    except (ValidationError, OCRProcessingError) as err:
        st.error(str(err))
        history_entry = create_history_entry(
            filename=filename,
            status="error",
            error_message=str(err),
        )
        return "", history_entry
    except Exception as err:
        error_msg = f"予期しないエラーが発生しました: {err}"
        st.error(error_msg)
        history_entry = create_history_entry(
            filename=filename,
            status="error",
            error_message=str(err),
        )
        return "", history_entry


def process_uploaded_files(
    processor: OCRProcessor,
    uploaded_files: list,
) -> None:
    """アップロードされた複数ファイルを一括処理する

    処理結果はst.session_stateに保存される。
    各ファイルの結果はファイル名付きセパレータで区切られる。

    Args:
        processor: OCRProcessorインスタンス
        uploaded_files: アップロードされたファイルのリスト
    """
    all_texts: list[str] = []

    for uploaded_file in uploaded_files:
        file_data = uploaded_file.getvalue()
        filename = uploaded_file.name

        text, history_entry = process_single_file(
            processor, file_data, filename
        )

        # 処理履歴に追加（イミュータブルに新しいリストを作成）
        st.session_state["processing_history"] = [
            *st.session_state["processing_history"],
            history_entry,
        ]

        if text:
            all_texts.append(f"--- {filename} ---\n{text}")

    # 全結果を結合してsession_stateに保存
    st.session_state["extracted_text"] = "\n\n".join(all_texts)


def render_header() -> None:
    """ページヘッダーを描画する"""
    st.title("OCR画像テキスト抽出アプリ")


def render_file_uploader() -> list:
    """ファイルアップローダーを描画する

    Returns:
        アップロードされたファイルのリスト（未選択時は空リスト）
    """
    uploaded_files = st.file_uploader(
        "画像ファイルをアップロード",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="対応形式: JPG, JPEG, PNG, BMP（最大10MB）",
    )
    return uploaded_files or []


def render_ocr_button() -> bool:
    """OCR実行ボタンを描画する

    Returns:
        ボタンが押されたかどうか
    """
    return st.button("OCR実行", type="primary")


def render_extracted_text() -> None:
    """抽出テキスト表示エリアを描画する"""
    st.text_area(
        "抽出テキスト",
        value=st.session_state["extracted_text"],
        height=300,
        help="OCRで抽出されたテキストが表示されます。編集も可能です。",
    )


def render_download_button() -> None:
    """結果ダウンロードボタンを描画する（テキストが存在する場合のみ）"""
    if st.session_state["extracted_text"]:
        st.download_button(
            label="結果をダウンロード",
            data=st.session_state["extracted_text"].encode("utf-8"),
            file_name="ocr_result.txt",
            mime="text/plain",
        )


def render_processing_history() -> None:
    """処理履歴を描画する"""
    st.subheader("処理履歴")

    history = st.session_state["processing_history"]

    if not history:
        st.info("まだ処理履歴がありません。")
        return

    for entry in reversed(history):
        filename = entry["filename"]
        status = entry["status"]
        timestamp = entry["timestamp"]

        if status == "success":
            st.write(f"- {filename}: 成功 ({timestamp})")
        else:
            error_msg = entry.get("error_message", "不明なエラー")
            st.write(f"- {filename}: エラー ({timestamp}) - {error_msg}")


@st.cache_resource
def get_ocr_processor() -> OCRProcessor:
    """OCRProcessorをキャッシュして返す

    Streamlit の cache_resource により、アプリの再実行時に
    PaddleOCR エンジンの重い初期化を繰り返さない。
    """
    return OCRProcessor()


def main() -> None:
    """アプリケーションのメインエントリポイント"""
    # ページ設定
    st.set_page_config(
        page_title="OCR画像テキスト抽出",
        page_icon="",
        layout="centered",
    )

    # セッション状態初期化
    initialize_session_state()

    # ヘッダー描画
    render_header()

    # ファイルアップロード
    uploaded_files = render_file_uploader()

    # OCR実行ボタン
    # OCRプロセッサの初期化はボタン押下時まで遅延させる
    # （Streamlit Cloudのヘルスチェックタイムアウトを回避するため）
    if render_ocr_button():
        if not uploaded_files:
            st.error("画像ファイルをアップロードしてください。")
        else:
            processor = get_ocr_processor()
            with st.spinner("OCR処理中..."):
                process_uploaded_files(processor, uploaded_files)

    # 抽出テキスト表示
    render_extracted_text()

    # ダウンロードボタン
    render_download_button()

    # 処理履歴
    render_processing_history()


if __name__ == "__main__":
    main()
