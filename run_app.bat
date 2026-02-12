@echo off
REM OCR Webアプリケーション起動スクリプト (Windows)
REM PYTHONPATHを設定してStreamlitアプリを起動します

REM カレントディレクトリをスクリプトのディレクトリに設定
cd /d "%~dp0"

REM PYTHONPATHにプロジェクトルートを追加
set PYTHONPATH=%CD%;%PYTHONPATH%

REM 仮想環境の確認
if not exist ".venv\" (
    echo エラー: 仮想環境が見つかりません。
    echo 以下のコマンドで仮想環境を作成してください：
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

REM 仮想環境のPythonを使用してStreamlitを実行
echo OCR Webアプリケーションを起動します...
echo PYTHONPATH=%PYTHONPATH%
.venv\Scripts\streamlit.exe run src\ui\app.py
