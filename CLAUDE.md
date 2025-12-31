# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案簡介

PDF OCR 工具 - 使用 PaddleOCR 將 PDF 檔案轉換成純文字。支援中文（繁體/簡體）、英文、日文等多種語言。

## 開發環境設定

使用 uv 管理 Python 環境：

```bash
# 建立虛擬環境
uv venv --python 3.12

# 安裝 GPU 版本依賴（預設）
uv pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install --index-strategy unsafe-best-match "paddleocr @ git+https://github.com/PaddlePaddle/PaddleOCR.git" pdf2image

# 安裝 CPU 版本依賴
uv pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

系統需要安裝 `poppler-utils`（用於 PDF 轉圖片）。

## 常用指令

```bash
# 基本 OCR（保留段落結構）
uv run python main.py document.pdf

# 表格識別模式
uv run python main.py document.pdf --table

# 簡體轉繁體（台灣用語）+ 表格識別
uv run python pdf_to_tw.py document.pdf

# PDF 翻譯（需設定 LLM API）
uv run python pdf_translate.py document.pdf --api-key YOUR_KEY
```

## 程式架構

專案包含三個主要腳本：

- **main.py**: 核心 OCR 工具
  - 使用 `PaddleOCR` 進行一般文字識別
  - 使用 `PPStructureV3` 進行表格識別（`--table` 模式）
  - 透過位置資訊智慧分組段落

- **pdf_to_tw.py**: 簡體中文轉繁體中文
  - 固定使用 `PPStructureV3` 表格識別模式
  - 使用 OpenCC `s2twp` 配置轉換為台灣用語

- **pdf_translate.py**: LLM 翻譯工具
  - 使用 `PPDocTranslation` 管線
  - 支援百度千帆平台或 OpenAI 兼容 API
  - 輸出 Markdown 格式

## PaddleOCR 3.x API 注意事項

- OCR 結果為字典物件，包含 `dt_polys`（座標）和 `rec_texts`（文字）
- 結構化識別結果包含 `parsing_res_list`、`table_res_list`、`overall_ocr_res`
- 表格輸出為 HTML 格式，需轉換為 ASCII 表格
