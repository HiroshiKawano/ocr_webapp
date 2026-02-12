# TDD開発実践ガイド

## 概要

このガイドでは、**Everything Claude Code**を使用して、実際にTDD（テスト駆動開発）を実践する具体的な手順を説明します。

Streamlit + PaddleOCRを例に、プロジェクトのセットアップからTDD開発の実行までを段階的に解説します。

---

## 前提条件

- Everything Claude Codeのグローバル設定へのインストールが完了していること（セクション1.1参照）
- Claude Code CLIが使用可能であること
- 基本的なPython開発環境が整っていること

---

## 1. ファイル配置場所の理解

Everything Claude Codeの設定ファイルは、**2つの配置場所**があります。

### 1.1 グローバル設定（全プロジェクト共通）

場所: `~/.claude/`

```
~/.claude/
├── agents/              # 全プロジェクトで共通使用するエージェント
│   ├── tdd-guide.md
│   ├── code-reviewer.md
│   ├── security-reviewer.md
│   └── ...
├── commands/            # 全プロジェクトで共通使用するコマンド
│   ├── tdd.md
│   ├── code-review.md
│   └── ...
├── rules/               # 全プロジェクトで共通適用するルール
│   ├── testing.md
│   ├── security.md
│   ├── coding-style.md
│   └── ...
└── skills/              # 全プロジェクトで共通使用するスキル
    ├── tdd-workflow/
    └── ...
```

**インストール方法**:

```bash
# リポジトリをクローン
git clone https://github.com/affaan-m/everything-claude-code.git
cd everything-claude-code

# グローバル設定にコピー
cp agents/*.md ~/.claude/agents/
cp commands/*.md ~/.claude/commands/
cp rules/*.md ~/.claude/rules/
cp -r skills/* ~/.claude/skills/
```

---

### 1.2 プロジェクト固有設定

場所: `<プロジェクトルート>/.claude/`

```
my-project/
├── .claude/
│   └── rules/           # このプロジェクト専用のルール
│       └── security.md  # プロジェクト固有のセキュリティ要件
├── src/
├── tests/
└── ...
```

**使い分けのポイント**:
- **グローバル設定**: 基本的なワークフロー、汎用的なルール（testing.md、coding-style.md など）
- **プロジェクト設定**: プロジェクト固有の要件、特別なルール（例: ファイルアップロードのセキュリティ要件）

Claude Codeは、プロジェクト設定を優先して読み込みます。同名のルールがある場合、プロジェクト設定が優先されます。

---

## 2. プロジェクトセットアップ手順

### 2.1 プロジェクトディレクトリの作成

プロジェクトは、Everything Claude Codeリポジトリの**外側**に作成します。

```bash
# Everything Claude Codeリポジトリと同じ階層に作成する例
cd ~/projects/   # または任意の作業ディレクトリ

# OCR Webアプリプロジェクトを作成
mkdir ocr-webapp
cd ocr-webapp
```

**推奨ディレクトリ構造**:

```
~/projects/
├── everything-claude-code/    # Everything Claude Code リポジトリ
│   ├── agents/
│   ├── commands/
│   ├── rules/
│   └── ...
└── ocr-webapp/                 # 新しいプロジェクト
    ├── .claude/
    ├── src/
    ├── tests/
    └── ...
```

---

### 2.2 プロジェクト構造の作成

OCR Webアプリの基本構造を作成します。

```bash
# プロジェクトフォルダ内で実行
cd ocr-webapp

# ディレクトリ構造を作成
mkdir -p src/ocr          # OCR処理モジュール
mkdir -p src/ui           # Streamlit UIモジュール
mkdir -p src/validation   # ファイル検証モジュール
mkdir -p tests/unit       # ユニットテスト
mkdir -p tests/integration # 統合テスト
mkdir -p tests/e2e        # E2Eテスト
mkdir -p data/uploads     # アップロードファイル保存先

# __init__.pyファイルを作成
touch src/__init__.py
touch src/ocr/__init__.py
touch src/ui/__init__.py
touch src/validation/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
touch tests/e2e/__init__.py

# .gitkeepで空ディレクトリを保持
touch data/uploads/.gitkeep

# 実装ファイル・テストファイルを空で作成
touch src/ocr/processor.py
touch src/ui/app.py
touch src/validation/validator.py
touch tests/unit/test_ocr_processor.py
touch tests/unit/test_validator.py
touch tests/integration/test_ui_integration.py
touch tests/e2e/test_full_workflow.py
```

**空のファイルを用意する意図**:

これらの実装ファイルとテストファイルを空の状態で作成しているのは、**TDD開発を手動で実践して学習するため**です。

- **学習目的**: ガイドを読みながら、自分自身でテストと実装を書いていくことで、TDD開発のサイクルを体験できます
- **構造の準備**: フォルダ構造とファイル配置を事前に用意することで、「どこに何を書くか」を考える手間を省き、TDDの実践に集中できます
- **段階的な実装**: セクション3以降で、RED-GREEN-IMPROVEサイクルに従って、これらの空ファイルに順次コードを追加していきます

`/tdd` コマンドを使用すると、Claude Codeがこれらのファイルにテストと実装を自動的に追加しますが、まずは自分で書いてみることで理解が深まります。

---

### 2.3 基本ファイルの作成

#### requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Streamlit Web Framework
streamlit>=1.31.0

# OCR Engine
paddleocr>=2.7.0
paddlepaddle>=2.5.0

# Image Processing
Pillow>=10.2.0
opencv-python>=4.9.0.80
numpy>=1.26.0

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
EOF
```

#### .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
.venv
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/

# Data
data/uploads/*
!data/uploads/.gitkeep

# OS
.DS_Store

# Streamlit
.streamlit/secrets.toml
EOF
```

#### README.md

```bash
cat > README.md << 'EOF'
# OCR Webapp

Streamlit + PaddleOCR を使用した OCR Web アプリケーション

## 機能

- 画像ファイルアップロード（JPG、PNG、BMP）
- 日本語、英語、中国語のOCR対応
- テキスト抽出結果のダウンロード
- セキュアなファイル検証（10MB制限）

## セットアップ

```bash
# 依存関係のインストール
pip install -r requirements.txt

# テストの実行
pytest --cov=src tests/

# アプリケーションの起動
streamlit run src/ui/app.py
```

## TDD開発フロー

このプロジェクトは Everything Claude Code を使用して TDD で開発されています。

1. `/tdd` コマンドで機能実装を開始
2. RED-GREEN-IMPROVE サイクルで開発
3. テストカバレッジ 80% 以上を維持
4. `/code-review` でコード品質を確認
EOF
```

#### pytest.ini

```bash
cat > pytest.ini << 'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
EOF
```

---

### 2.4 プロジェクト固有の.claude設定

**グローバル設定（`~/.claude/rules/`）について**:

すでにEverything Claude Codeからコピーした汎用ルールがグローバル設定にあるため、プロジェクト固有の特別な要件のみをプロジェクト設定に追加します。

- **グローバル設定で対応**: testing.md、coding-style.md などの汎用ルール
- **プロジェクト設定で追加**: このOCR Webappプロジェクト固有のセキュリティ要件

```bash
# .claude/rules ディレクトリを作成
mkdir -p .claude/rules
```

#### .claude/rules/security.md

このプロジェクト固有のファイルアップロードセキュリティ要件を定義します。

```bash
cat > .claude/rules/security.md << 'EOF'
# セキュリティガイドライン

## ファイルアップロードのセキュリティ

### 必須チェック
- [ ] ファイルサイズ制限: 10MB
- [ ] 許可拡張子のみ: jpg, jpeg, png, bmp
- [ ] MIMEタイプ検証
- [ ] ファイル名のサニタイズ
- [ ] アップロードパスの検証

### 実装例

```python
# 良い例: セキュアなファイル検証
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_uploaded_file(uploaded_file):
    """アップロードされたファイルを検証"""
    if uploaded_file.size > MAX_FILE_SIZE:
        raise ValueError("ファイルサイズが大きすぎます（最大10MB）")

    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"許可されていないファイル形式です。許可: {ALLOWED_EXTENSIONS}")

    return True
```

## エラーハンドリング

すべてのOCR処理でエラーハンドリング：
```python
try:
    result = ocr_processor.extract_text(image)
    return result
except Exception as e:
    logger.error(f"OCR処理エラー: {e}")
    raise OCRProcessingError("画像からテキストを抽出できませんでした")
```
EOF
```

**補足**: テスト要件やコーディングスタイルなどの汎用的なルールは、グローバル設定（`~/.claude/rules/`）から自動的に読み込まれます。

---

## 3. TDD開発の実行手順

プロジェクトセットアップが完了したら、TDD開発を開始します。

### 3.1 `/tdd` コマンドの使用

Claude Codeで以下のように指示します：

```
/tdd
OCR処理機能を実装してください。
PaddleOCRを使って画像からテキストを抽出するOCRProcessorクラスを作成してください。
必要に応じてask_user_inputスタイルでお願いします。
```

### 3.2 TDDサイクル: RED-GREEN-IMPROVE

`/tdd` コマンドは以下のワークフローを自動的に実行します：

#### ステップ1: RED（失敗するテストを書く）

Claude Codeが最初にテストを作成します：

```python
# tests/unit/test_ocr_processor.py
import pytest
from PIL import Image
from src.ocr.processor import OCRProcessor

def test_ocr_processor_initialization():
    """OCRProcessorが正しく初期化される"""
    processor = OCRProcessor(lang='ja')
    assert processor is not None
    assert processor.lang == 'ja'

def test_extract_text_from_image():
    """画像からテキストを抽出できる"""
    processor = OCRProcessor(lang='ja')
    # テスト用の画像を作成
    image = Image.new('RGB', (100, 100), color='white')
    result = processor.extract_text(image)
    assert isinstance(result, list)

def test_extract_text_with_invalid_image():
    """無効な画像でエラーを発生させる"""
    processor = OCRProcessor(lang='ja')
    with pytest.raises(ValueError):
        processor.extract_text(None)
```

テストを実行して、失敗することを確認：

```bash
pytest tests/unit/test_ocr_processor.py
# FAILED - src.ocr.processor モジュールが存在しません
```

#### ステップ2: GREEN（テストが通る最小限の実装）

Claude Codeが実装を作成します：

```python
# src/ocr/processor.py
from typing import List, Optional
from PIL import Image
from paddleocr import PaddleOCR

class OCRProcessor:
    """PaddleOCRを使用したOCR処理クラス"""

    def __init__(self, lang: str = 'ja'):
        """
        Args:
            lang: 言語コード ('ja', 'en', 'ch')
        """
        self.lang = lang
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def extract_text(self, image: Image.Image) -> List[str]:
        """画像からテキストを抽出

        Args:
            image: PIL Image オブジェクト

        Returns:
            抽出されたテキストのリスト

        Raises:
            ValueError: 無効な画像の場合
        """
        if image is None:
            raise ValueError("画像がNoneです")

        # 画像をNumPy配列に変換
        import numpy as np
        image_np = np.array(image)

        # OCR実行
        result = self.ocr.ocr(image_np, cls=True)

        # テキストを抽出
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append(line[1][0])

        return texts
```

テストを実行して、通過することを確認：

```bash
pytest tests/unit/test_ocr_processor.py
# PASSED - すべてのテストが成功
```

#### ステップ3: IMPROVE（リファクタリング）

Claude Codeがコードを改善します：
- エラーハンドリングの追加
- ロギングの追加
- ドキュメントの改善

```python
# src/ocr/processor.py（改善版）
import logging
from typing import List, Optional
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class OCRProcessor:
    """PaddleOCRを使用したOCR処理クラス"""

    SUPPORTED_LANGS = ['ja', 'en', 'ch']

    def __init__(self, lang: str = 'ja'):
        """OCRプロセッサを初期化

        Args:
            lang: 言語コード ('ja', 'en', 'ch')

        Raises:
            ValueError: サポートされていない言語の場合
        """
        if lang not in self.SUPPORTED_LANGS:
            raise ValueError(
                f"サポートされていない言語: {lang}. "
                f"サポート言語: {self.SUPPORTED_LANGS}"
            )

        self.lang = lang
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
            logger.info(f"OCRProcessor initialized with lang={lang}")
        except Exception as e:
            logger.error(f"PaddleOCR初期化エラー: {e}")
            raise

    def extract_text(self, image: Image.Image) -> List[str]:
        """画像からテキストを抽出

        Args:
            image: PIL Image オブジェクト

        Returns:
            抽出されたテキストのリスト

        Raises:
            ValueError: 無効な画像の場合
            RuntimeError: OCR処理エラーの場合
        """
        if image is None:
            raise ValueError("画像がNoneです")

        if not isinstance(image, Image.Image):
            raise ValueError("PIL.Image.Imageオブジェクトが必要です")

        try:
            # 画像をNumPy配列に変換
            image_np = np.array(image)
            logger.debug(f"Image shape: {image_np.shape}")

            # OCR実行
            result = self.ocr.ocr(image_np, cls=True)

            # テキストを抽出
            texts = self._extract_texts_from_result(result)
            logger.info(f"Extracted {len(texts)} text lines")

            return texts

        except Exception as e:
            logger.error(f"OCR処理エラー: {e}")
            raise RuntimeError(f"テキスト抽出に失敗しました: {e}")

    def _extract_texts_from_result(self, result) -> List[str]:
        """OCR結果からテキストを抽出

        Args:
            result: PaddleOCRの結果

        Returns:
            抽出されたテキストのリスト
        """
        texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    texts.append(line[1][0])
        return texts
```

---

### 3.3 テストカバレッジの確認

Claude Codeがテストカバレッジを確認します：

```bash
pytest --cov=src --cov-report=html --cov-report=term tests/

# 出力例:
# ---------- coverage: platform darwin, python 3.11.0 -----------
# Name                    Stmts   Miss  Cover
# -------------------------------------------
# src/__init__.py             0      0   100%
# src/ocr/__init__.py         0      0   100%
# src/ocr/processor.py       35      3    91%
# -------------------------------------------
# TOTAL                      35      3    91%
```

**重要**: カバレッジは **80% 以上** を維持します。

---

### 3.4 コードレビュー

TDD開発が完了したら、`/code-review` コマンドでコード品質を確認します：

```
/code-review
```

Claude Codeが以下をチェックします：
- コード品質
- セキュリティ
- ベストプラクティス遵守
- テストカバレッジ
- ドキュメント

---

## 4. 次の機能を実装

同じTDDサイクルで次の機能を実装します：

### 4.1 ファイル検証機能

```
/tdd
ファイル検証機能を実装してください。
ファイルサイズ（10MB以下）と拡張子（jpg, jpeg, png, bmp）を検証するValidatorクラスを作成してください。
必要に応じてask_user_inputスタイルでお願いします。
```

### 4.2 Streamlit UI

```
/tdd
Streamlit UIを実装してください。
ファイルアップロード、OCR実行、結果表示を行うapp.pyを作成してください。
必要に応じてask_user_inputスタイルでお願いします。
```

---

## 5. 完成したプロジェクト構造

最終的なプロジェクト構造：

```
ocr-webapp/
├── .claude/
│   └── rules/
│       └── security.md           # プロジェクト固有のセキュリティ要件（内容あり）
├── src/
│   ├── __init__.py
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── processor.py          # OCRProcessor クラス（空 - TDD開発で実装）
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py                # Streamlit アプリ（空 - TDD開発で実装）
│   └── validation/
│       ├── __init__.py
│       └── validator.py          # Validator クラス（空 - TDD開発で実装）
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_ocr_processor.py  # 空 - TDD開発でテストを書く
│   │   └── test_validator.py      # 空 - TDD開発でテストを書く
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_ui_integration.py # 空 - TDD開発でテストを書く
│   └── e2e/
│       ├── __init__.py
│       └── test_full_workflow.py  # 空 - TDD開発でテストを書く
├── data/
│   └── uploads/
│       └── .gitkeep
├── .gitignore                     # 内容あり
├── README.md                      # 内容あり
├── requirements.txt               # 内容あり
└── pytest.ini                     # 内容あり
```

**注**:
- testing.md、coding-style.md などの汎用ルールはグローバル設定（`~/.claude/rules/`）から読み込まれるため、プロジェクト内には配置していません
- 実装ファイル（processor.py、app.py、validator.py）とテストファイルは空の状態です。セクション3以降のTDD開発で、これらのファイルに順次コードを追加していきます

---

## 6. まとめ

### TDD開発の流れ

1. **プロジェクトセットアップ**
   - ディレクトリ構造の作成
   - .claude/rules の設定
   - requirements.txt, .gitignore の作成

2. **TDD開発サイクル**
   - `/tdd` コマンドで機能実装を開始
   - RED: 失敗するテストを書く
   - GREEN: テストが通る最小限の実装
   - IMPROVE: リファクタリング
   - カバレッジ確認（80%以上）

3. **コード品質確認**
   - `/code-review` でレビュー
   - セキュリティチェック
   - ベストプラクティス確認

4. **次の機能へ**
   - 同じサイクルを繰り返す

### 重要なポイント

- **テストを先に書く** - これがTDDの核心
- **最小限の実装** - テストが通る最小のコードだけ書く
- **リファクタリング** - テストが通った後で改善
- **80%カバレッジ維持** - 品質基準を守る
- **コードレビュー** - 品質とセキュリティを確保

---

## 参考リンク

- [Everything Claude Code リポジトリ](https://github.com/affaan-m/everything-claude-code)
- [Shorthand Guide](https://x.com/affaanmustafa/status/2012378465664745795)
- [Longform Guide](https://x.com/affaanmustafa/status/2014040193557471352)

---

## 7. アプリケーションの起動と使い方

### 7.1 環境セットアップ

TDD開発が完了したら、アプリケーションを起動できます。

#### 依存関係のインストール

```bash
# プロジェクトディレクトリに移動
cd ocr_webapp

# 仮想環境の作成（推奨）
python3 -m venv .venv

# 仮想環境の有効化
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# pipをアップグレード
pip install --upgrade pip

# 依存パッケージのインストール
pip install -r requirements.txt
```

**注意事項**:

1. **仮想環境の場所**: 必ずプロジェクトディレクトリ（`ocr_webapp`）内に`.venv`として作成してください
2. **OpenCVについて**: `requirements.txt`の`opencv-python`はコメントアウトされています。PaddleOCRが自動的に`opencv-contrib-python`をインストールするためです
3. **macOSユーザー**: 以下のエラーが発生した場合、Xcode Command Line Toolsのインストールが必要です：
   ```
   xcrun: error: active developer path ("/Applications/Xcode.app/Contents/Developer") does not exist
   ```
   解決方法：
   ```bash
   xcode-select --install
   ```
   ダイアログが表示されたら「インストール」をクリックしてください

#### PaddleOCRのモデルダウンロード

初回起動時、PaddleOCRが自動的に言語モデルをダウンロードします（数百MB）。
事前にダウンロードすることも可能です：

```bash
# 仮想環境をアクティベートしてから実行
source .venv/bin/activate

# 日本語モデルのダウンロード
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True, lang='japan')"
```

**注**: `use_angle_cls`パラメータは非推奨となり、代わりに`use_textline_orientation`を使用します

---

### 7.2 アプリケーションの起動

#### 方法1: 起動スクリプトを使用（推奨）

最も簡単な方法は、用意されている起動スクリプトを使用することです：

```bash
# macOS/Linux
./run_app.sh

# Windows
run_app.bat
```

起動スクリプトは以下を自動的に行います：
- `PYTHONPATH` 環境変数の設定（プロジェクトルートを追加）
- 仮想環境の確認
- Streamlitアプリの起動

#### 方法2: 手動で起動

`PYTHONPATH` を設定してから実行する場合：

```bash
# macOS/Linux
export PYTHONPATH=/Users/username/Desktop/dev/ocr_webapp:$PYTHONPATH
streamlit run src/ui/app.py

# Windows (PowerShell)
$env:PYTHONPATH="C:\Users\username\Desktop\dev\ocr_webapp;$env:PYTHONPATH"
streamlit run src/ui/app.py

# Windows (コマンドプロンプト)
set PYTHONPATH=C:\Users\username\Desktop\dev\ocr_webapp;%PYTHONPATH%
streamlit run src/ui/app.py
```

**注意**: パスは実際のプロジェクトディレクトリに合わせて変更してください。

起動すると、ブラウザが自動的に開き、以下のURLでアプリケーションが表示されます：

```
http://localhost:8501
```

#### 起動オプション

```bash
# ポート番号を指定
./run_app.sh --server.port 8080

# 自動ブラウザ起動を無効化
./run_app.sh --server.headless true

# 開発モード（ファイル変更を自動検出）
./run_app.sh --server.runOnSave true
```

#### 起動スクリプトの詳細

起動スクリプト（`run_app.sh` / `run_app.bat`）は以下の処理を行います：

1. **プロジェクトルートの特定**: スクリプトのあるディレクトリをプロジェクトルートとして認識
2. **PYTHONPATH の設定**: Pythonが `src` モジュールを見つけられるようにする
3. **仮想環境の確認**: `.venv` ディレクトリが存在するか確認
4. **Streamlit の起動**: 仮想環境内のStreamlitを使用してアプリを起動

---

### 7.3 使い方

#### ステップ1: 画像ファイルのアップロード

1. 画面上部の「📁 画像ファイルをアップロード」をクリック
2. 複数の画像ファイルを選択可能（対応形式: JPG, JPEG, PNG, BMP）
3. ファイルサイズ制限: 各ファイル10MB以下

**サポートされる画像形式**:
- `.jpg`, `.jpeg` - JPEG画像
- `.png` - PNG画像
- `.bmp` - ビットマップ画像

#### ステップ2: OCR実行

1. 「🚀 OCR実行」ボタンをクリック
2. PaddleOCRがテキストを抽出します（数秒かかる場合があります）
3. 抽出結果が「📝 抽出されたテキスト」エリアに表示されます

**複数ファイルの処理**:
- 複数ファイルをアップロードした場合、自動的に一括処理されます
- 各ファイルの結果は `\n--- [ファイル名] ---\n` で区切られます

#### ステップ3: 結果の確認と編集

- 抽出されたテキストは編集可能です
- 必要に応じて手動で修正できます
- テキストエリアは自動的にリサイズされます

#### ステップ4: 結果のダウンロード

1. 「💾 結果をダウンロード (txt)」ボタンが表示されます
2. クリックすると、抽出テキストがテキストファイルとしてダウンロードされます
3. ファイル名: `ocr_result_YYYYMMDD_HHMMSS.txt`

#### 処理履歴の確認

画面下部の「📋 処理履歴」セクションで、過去の処理結果を確認できます：

- ✅ **成功**: ファイル名、処理日時が表示されます
- ❌ **エラー**: ファイル名、エラー内容、処理日時が表示されます
- 最新の処理が上部に表示されます（新しい順）

**注**: 処理履歴はブラウザセッション内でのみ保持されます。ページをリロードすると履歴はクリアされます。

---

### 7.4 テストの実行

開発中や機能追加後にテストを実行して、品質を確認します。

#### 全テストの実行

```bash
# 全テスト実行 + カバレッジレポート
pytest --cov=src --cov-report=term-missing --cov-report=html tests/

# 出力例:
# ======================== test session starts =========================
# collected 203 items
#
# tests/unit/test_app.py .......................... [ 16%]
# tests/unit/test_file_validator.py ............... [ 60%]
# tests/unit/test_ocr_processor.py ................ [ 74%]
# tests/integration/test_ocr_with_validation.py .. [ 95%]
# tests/integration/test_ui_integration.py ....... [100%]
#
# ======================== 203 passed in 45.2s ========================
#
# ---------- coverage: platform darwin, python 3.11.0 -----------
# Name                              Stmts   Miss  Cover   Missing
# ---------------------------------------------------------------
# src/__init__.py                       0      0   100%
# src/ocr/__init__.py                   8      0   100%
# src/ocr/processor.py                219      4    98%   45-48
# src/ui/app.py                       120     20    83%
# src/validators/__init__.py           15      0   100%
# src/validators/file_validator.py    177      0   100%
# ---------------------------------------------------------------
# TOTAL                               539     24    95%
```

#### カテゴリ別テスト実行

```bash
# ユニットテストのみ
pytest tests/unit/ -v

# 統合テストのみ
pytest tests/integration/ -v

# 特定のテストファイル
pytest tests/unit/test_ocr_processor.py -v

# 特定のテストクラス
pytest tests/unit/test_ocr_processor.py::TestOCRProcessorInit -v

# 特定のテスト関数
pytest tests/unit/test_ocr_processor.py::test_extract_text_from_file_path -v
```

#### HTMLカバレッジレポートの確認

```bash
# HTMLレポート生成
pytest --cov=src --cov-report=html tests/

# ブラウザで開く
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

### 7.5 トラブルシューティング

#### 問題1: ModuleNotFoundError: No module named 'paddleocr'

**エラー**:
```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True, lang='japan')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'paddleocr'
```

**原因**: 正しい仮想環境がアクティベートされていない、または依存関係がインストールされていない

**解決策**:
```bash
# 現在のPythonパスを確認
which python
# 出力例: /Users/username/Desktop/dev/ocr_webapp/.venv/bin/python
# ↑ プロジェクトの.venvディレクトリを指していることを確認

# 正しい仮想環境をアクティベート
cd /path/to/ocr_webapp
source .venv/bin/activate

# プロンプトに(.venv)が表示されることを確認
# (.venv) username@mac ocr_webapp %

# 依存関係を再インストール
pip install -r requirements.txt
```

#### 問題2: opencv-pythonのビルドエラー（macOS）

**エラー**:
```
xcrun: error: active developer path ("/Applications/Xcode.app/Contents/Developer") does not exist
Building wheel for opencv-python (pyproject.toml) ... error
```

**原因**: Xcode Command Line Toolsがインストールされていない

**解決策**:
```bash
# Xcode Command Line Toolsをインストール
xcode-select --install

# ダイアログが表示されたら「インストール」をクリック
# インストール完了後、再度依存関係をインストール
pip install -r requirements.txt
```

**代替案**: `requirements.txt`の`opencv-python`行は既にコメントアウトされています。PaddleOCRが自動的に`opencv-contrib-python`（プリビルド版）をインストールするため、通常はこの問題は発生しません。

#### 問題3: PaddleOCRのインストールエラー

**エラー**:
```
ERROR: Could not find a version that satisfies the requirement paddleocr
```

**解決策**:
```bash
# Python 3.8以上を使用していることを確認
python --version

# pipをアップグレード
pip install --upgrade pip

# 再インストール
pip install paddleocr paddlepaddle
```

#### 問題4: Streamlitが起動しない

**エラー**:
```
ModuleNotFoundError: No module named 'streamlit'
```

**解決策**:
```bash
# 仮想環境が有効化されていることを確認
which python  # プロジェクトの.venv/bin/pythonが表示されるはず

# 正しい仮想環境をアクティベート
source .venv/bin/activate

# 依存関係を再インストール
pip install -r requirements.txt
```

#### 問題5: OCR処理が遅い

**原因**: 初回実行時はモデルのダウンロードで時間がかかります

**解決策**:
- 初回起動時は数分待つ（モデルダウンロード完了後は高速化）
- 2回目以降はキャッシュされたモデルを使用するため高速

#### 問題6: ファイルアップロードエラー

**エラー**:
```
ValidationError: ファイルサイズが制限を超えています
```

**解決策**:
- ファイルサイズを10MB以下に削減
- 画像を圧縮してから再アップロード
- 必要に応じて `src/validators/file_validator.py` の `max_file_size` を変更

#### 問題7: 日本語OCRの精度が低い

**原因**: 画像の品質や解像度が低い

**解決策**:
- より高解像度の画像を使用
- 画像の明るさやコントラストを調整
- 文字が明瞭な画像を使用（手書きよりも印刷文字が認識精度が高い）

#### 問題8: テストが失敗する

**エラー**:
```
FAILED tests/unit/test_ocr_processor.py::test_extract_text
```

**解決策**:
```bash
# 詳細なエラーメッセージを表示
pytest tests/unit/test_ocr_processor.py -v -s

# モックの問題の場合、テストファイルを確認
# 依存パッケージのバージョンを確認
pip list | grep pytest

# キャッシュをクリア
pytest --cache-clear
```

---

### 7.6 開発の継続

新機能を追加する場合も、同じTDDサイクルを使用します：

```bash
# 1. Claude Code CLIを起動
claude-code

# 2. TDDコマンドで新機能を実装
/tdd
複数言語対応を追加してください。
英語、中国語のOCR処理をサポートするように拡張してください。

# 3. コードレビュー
/code-review

# 4. テストカバレッジ確認
/test-coverage

# 5. ビルドエラーがあれば修正
/build-and-fix
```

---

## 8. Streamlit Community Cloudへのデプロイ

### 8.1 Streamlit Community Cloudとは

Streamlit Community Cloudは、Streamlitアプリを無料でホスティングできるクラウドサービスです。

**主な特徴**:
- 完全無料（パブリックリポジトリの場合）
- GitHubと連携した自動デプロイ
- HTTPSに対応
- 簡単な設定で即座にデプロイ可能

**制限事項**:
- CPU: 0.78 cores
- メモリ: 800 MB
- ストレージ: 一時的なファイル保存のみ（再起動で削除）

---

### 8.2 前提条件

デプロイ前に以下を準備してください：

1. **GitHubアカウント** - [github.com](https://github.com) で作成
2. **パブリックGitHubリポジトリ** - プロジェクトをGitHubにプッシュ済み
3. **Streamlit Community Cloudアカウント** - GitHubアカウントでサインイン

---

### 8.3 GitHubリポジトリの準備

#### ステップ1: Gitリポジトリの初期化

```bash
# プロジェクトディレクトリで実行
cd ocr_webapp

# Gitリポジトリを初期化
git init

# すべてのファイルを追加
git add .

# 初回コミット
git commit -m "feat: TDD開発完了したOCR Webアプリの初期コミット"
```

#### ステップ2: GitHubリポジトリの作成

1. [github.com](https://github.com) にログイン
2. 右上の「+」→「New repository」をクリック
3. リポジトリ設定：
   - Repository name: `ocr-webapp`
   - Description: `Streamlit + PaddleOCR を使用した OCR Webアプリケーション`
   - Public を選択（無料デプロイにはPublicが必要）
   - 「Add a README file」はチェックしない（既にREADME.mdがあるため）
4. 「Create repository」をクリック

#### ステップ3: リモートリポジトリに接続してプッシュ

```bash
# リモートリポジトリを追加（<username>は自分のGitHubユーザー名）
git remote add origin https://github.com/<username>/ocr-webapp.git

# メインブランチにプッシュ
git branch -M main
git push -u origin main
```

**注**: GitHubの認証が求められた場合、Personal Access Token（PAT）を使用してください。

---

### 8.4 デプロイに必要なファイルの確認

Streamlit Community Cloudでは、以下のファイルが必要です：

#### 必須ファイル
- ✅ `requirements.txt` - 依存パッケージのリスト（既に作成済み）
- ✅ `src/ui/app.py` - Streamlitアプリのメインファイル（既に作成済み）

#### 推奨ファイル
- `.streamlit/config.toml` - Streamlit設定（オプション）
- `packages.txt` - システムパッケージ（必要に応じて）

#### .streamlit/config.toml の作成（オプション）

アプリの外観や動作をカスタマイズする場合に作成します：

```bash
# .streamlit ディレクトリを作成
mkdir -p .streamlit

# 設定ファイルを作成
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 10

[browser]
gatherUsageStats = false
EOF
```

#### packages.txt の作成（PaddleOCR用）

PaddleOCRが必要とするシステムパッケージをインストールするために作成します：

```bash
cat > packages.txt << 'EOF'
libgomp1
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgl1-mesa-glx
EOF
```

**重要**: `packages.txt`は、PaddleOCRの依存関係であるOpenCVが正しく動作するために必要です。

#### 変更をコミット＆プッシュ

```bash
# 新しいファイルを追加
git add .streamlit/config.toml packages.txt

# コミット
git commit -m "chore: Streamlit Cloud デプロイ用の設定ファイルを追加"

# プッシュ
git push origin main
```

---

### 8.5 Streamlit Community Cloudでのデプロイ

#### ステップ1: Streamlit Community Cloudにサインイン

1. [share.streamlit.io](https://share.streamlit.io/) にアクセス
2. 「Sign in with GitHub」をクリック
3. GitHubアカウントで認証

#### ステップ2: 新しいアプリをデプロイ

1. 「New app」ボタンをクリック
2. デプロイ設定を入力：

   **Repository**:
   - Repository: `<username>/ocr-webapp`
   - Branch: `main`
   - Main file path: `src/ui/app.py`

   **Advanced settings**（オプション）:
   - Python version: `3.11`（推奨）
   - Secrets: 不要（環境変数が必要な場合のみ）

3. 「Deploy!」ボタンをクリック

#### ステップ3: デプロイの進行状況を確認

デプロイが開始されると、以下のプロセスが実行されます：

1. **Building**: 依存パッケージのインストール（数分かかる）
2. **Installing PaddleOCR**: PaddleOCRのインストール
3. **Downloading models**: PaddleOCRモデルのダウンロード
4. **Starting app**: アプリの起動

**注**: 初回デプロイは5〜10分程度かかります。

#### ステップ4: アプリの確認

デプロイが完了すると、以下のようなURLでアプリにアクセスできます：

```
https://<your-app-name>.streamlit.app
```

例: `https://ocr-webapp-affaan.streamlit.app`

---

### 8.6 デプロイ後の管理

#### アプリの再デプロイ

コードを変更してGitHubにプッシュすると、自動的に再デプロイされます：

```bash
# コードを変更
# ...

# コミット＆プッシュ
git add .
git commit -m "feat: 新機能を追加"
git push origin main

# 自動的にStreamlit Cloudが検出して再デプロイ
```

#### 手動での再起動

Streamlit Community Cloudのダッシュボードから：

1. アプリを選択
2. 「⋮」メニュー → 「Reboot app」をクリック

#### ログの確認

デプロイエラーやアプリのログを確認する場合：

1. アプリを選択
2. 「Manage app」→「Logs」タブ
3. エラーメッセージやスタックトレースを確認

#### アプリの削除

1. ダッシュボードでアプリを選択
2. 「⋮」メニュー → 「Delete app」をクリック

---

### 8.7 トラブルシューティング

#### 問題1: ModuleNotFoundError during deployment

**エラー**:
```
ModuleNotFoundError: No module named 'paddleocr'
```

**原因**: `requirements.txt`に依存パッケージが記載されていない、またはバージョンが古い

**解決策**:
```bash
# requirements.txtを確認
cat requirements.txt

# 必要なパッケージが含まれていることを確認
streamlit>=1.31.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
Pillow>=10.2.0

# 変更してコミット＆プッシュ
git add requirements.txt
git commit -m "fix: requirements.txtを更新"
git push origin main
```

#### 問題2: Memory limit exceeded

**エラー**:
```
Your app has exceeded its memory limit (800MB)
```

**原因**: PaddleOCRのモデルがメモリを大量に消費している

**解決策**:

1. **軽量モデルを使用**:
   ```python
   # src/ocr/processor.py
   class OCRProcessor:
       def __init__(self, lang: str = 'japan'):
           self.ocr = PaddleOCR(
               use_textline_orientation=True,
               lang=lang,
               use_gpu=False,  # GPUを無効化（メモリ削減）
               show_log=False   # ログを無効化
           )
   ```

2. **キャッシュを使用**:
   ```python
   import streamlit as st

   @st.cache_resource
   def load_ocr_processor(lang='japan'):
       return OCRProcessor(lang=lang)
   ```

3. **処理する画像サイズを制限**:
   ```python
   # 画像を圧縮
   max_dimension = 1024
   if image.width > max_dimension or image.height > max_dimension:
       image.thumbnail((max_dimension, max_dimension))
   ```

#### 問題3: File upload doesn't work

**エラー**: ファイルアップロード後、何も起こらない

**原因**: `PYTHONPATH`の設定問題またはモジュールインポートエラー

**解決策**:

`src/ui/app.py`のインポート文を相対パスから絶対パスに変更：

```python
# 修正前
from src.ocr.processor import OCRProcessor
from src.validators.file_validator import FileValidator

# 修正後（プロジェクトルートをsys.pathに追加）
import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ocr.processor import OCRProcessor
from src.validators.file_validator import FileValidator
```

#### 問題4: PaddleOCRモデルのダウンロードに失敗

**エラー**:
```
Error downloading PaddleOCR model
```

**原因**: ネットワークタイムアウトまたはストレージ不足

**解決策**:

1. **リトライロジックを追加**:
   ```python
   import time

   def initialize_ocr_with_retry(lang='japan', max_retries=3):
       for i in range(max_retries):
           try:
               return PaddleOCR(use_textline_orientation=True, lang=lang)
           except Exception as e:
               if i < max_retries - 1:
                   time.sleep(2)
                   continue
               raise
   ```

2. **キャッシュを有効化**:
   ```python
   @st.cache_resource
   def load_ocr_processor(lang='japan'):
       return initialize_ocr_with_retry(lang)
   ```

#### 問題5: App URL is not accessible

**エラー**: デプロイ完了後、URLにアクセスできない

**解決策**:

1. **リポジトリがPublicか確認**:
   - GitHubリポジトリ設定 → Settings → General
   - 「Change visibility」でPublicに変更

2. **アプリのステータスを確認**:
   - Streamlit Cloudダッシュボードでアプリのステータスを確認
   - 「Running」になっていることを確認

3. **ログを確認**:
   - 「Manage app」→「Logs」でエラーメッセージを確認

---

### 8.8 パフォーマンス最適化のヒント

#### 1. OCRプロセッサのキャッシング

```python
import streamlit as st

@st.cache_resource
def load_ocr_processor(lang='japan'):
    """OCRプロセッサをキャッシュして再利用"""
    return OCRProcessor(lang=lang)

# 使用
processor = load_ocr_processor()
```

#### 2. 画像の前処理

```python
def preprocess_image(image, max_size=1024):
    """画像を圧縮してメモリ使用量を削減"""
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image
```

#### 3. プログレスバーの表示

```python
import streamlit as st

with st.spinner('OCR処理中...'):
    result = processor.extract_text(image)
st.success('処理完了!')
```

---

### 8.9 本番環境のベストプラクティス

#### セキュリティ

1. **環境変数の使用**: APIキーなどの機密情報は、Streamlit Cloudの「Secrets」機能を使用
   ```toml
   # Streamlit CloudのSecretsに設定
   [secrets]
   api_key = "your-secret-api-key"
   ```

   ```python
   # アプリで使用
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

2. **ファイルサイズ制限**: `.streamlit/config.toml`で制限を設定済み
   ```toml
   [server]
   maxUploadSize = 10  # 10MB
   ```

#### モニタリング

1. **エラーログの記録**:
   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)

   try:
       result = processor.extract_text(image)
   except Exception as e:
       logger.error(f"OCR処理エラー: {e}")
       st.error("処理中にエラーが発生しました")
   ```

2. **使用状況の追跡**: Streamlit Community Cloudのダッシュボードで確認可能
   - アクセス数
   - エラー率
   - リソース使用状況

---

### 8.10 まとめ

Streamlit Community Cloudへのデプロイ手順：

1. ✅ GitHubリポジトリを作成してコードをプッシュ
2. ✅ `requirements.txt`、`packages.txt`を準備
3. ✅ Streamlit Community Cloudでアプリをデプロイ
4. ✅ デプロイの進行状況を確認
5. ✅ アプリのURLにアクセスして動作確認

**デプロイ後の運用**:
- コード変更をプッシュすると自動的に再デプロイ
- ログでエラーを監視
- パフォーマンス最適化を継続

これでOCR Webアプリが世界中からアクセス可能になります！

---

**Happy TDD Development! 🚀**
