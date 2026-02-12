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
