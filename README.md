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
