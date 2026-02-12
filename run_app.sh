#!/bin/bash

# OCR Webアプリケーション起動スクリプト
# PYTHONPATHを設定してStreamlitアプリを起動します

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# PYTHONPATHにプロジェクトルートを追加
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 仮想環境の確認
if [ ! -d "${SCRIPT_DIR}/.venv" ]; then
    echo "エラー: 仮想環境が見つかりません。"
    echo "以下のコマンドで仮想環境を作成してください："
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 仮想環境のPythonを使用してStreamlitを実行
echo "OCR Webアプリケーションを起動します..."
echo "PYTHONPATH=${PYTHONPATH}"
"${SCRIPT_DIR}/.venv/bin/streamlit" run "${SCRIPT_DIR}/src/ui/app.py"
