# OCR Webアプリ

画像からテキストを抽出するOCR Webアプリケーション。Python 3.x + Tesseract/EasyOCR + pytest。

## プロジェクト固有ルール

**必読**: [.claude/rules/security.md](.claude/rules/security.md)
- ファイルアップロードセキュリティ（10MB制限、許可拡張子、MIME検証）
- OCR処理エラーハンドリング

グローバルルール（`~/.claude/rules/`）も自動適用。

## 使用推奨Skills

- `/tdd` - 新機能・バグ修正時（80%カバレッジ必須）
- `/security-review` - ファイルアップロード/OCR処理実装時
- `/code-review` - コード変更後
- `/plan` - 複雑な機能実装前
- `/build-and-fix` - ビルドエラー時
- `/test-coverage` - カバレッジ確認
