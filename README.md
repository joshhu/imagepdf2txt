# imagepdf2txt

使用 PaddleOCR 將圖片型 PDF 檔案轉換成純文字。支援中文（繁體/簡體）、英文、日文等多種語言。

## 功能特色

- **智慧段落識別**：根據文字位置自動分組成段落，保留原始文件結構
- **表格識別**：支援表格結構識別，輸出為 ASCII 表格格式
- **簡繁轉換**：內建簡體中文轉繁體中文（台灣用語）功能
- **LLM 翻譯**：整合 LLM API 進行多語言翻譯
- **GPU 加速**：支援 NVIDIA GPU 加速處理

## 系統需求

- Python 3.10 - 3.12
- NVIDIA GPU（可選，但建議使用以加速處理）
- poppler-utils（用於 PDF 轉圖片）

### 安裝 poppler-utils

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler

# Windows
# 下載 poppler for Windows：https://github.com/oschwartz10612/poppler-windows/releases
```

## 安裝

使用 [uv](https://docs.astral.sh/uv/) 管理 Python 環境：

```bash
# 複製專案
git clone https://github.com/YOUR_USERNAME/imagepdf2txt.git
cd imagepdf2txt

# 建立虛擬環境
uv venv --python 3.12

# 安裝 GPU 版本依賴（建議）
uv pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install --index-strategy unsafe-best-match "paddleocr @ git+https://github.com/PaddlePaddle/PaddleOCR.git" pdf2image opencc-python-reimplemented prettytable lxml

# 或安裝 CPU 版本依賴
uv pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
uv pip install --index-strategy unsafe-best-match "paddleocr @ git+https://github.com/PaddlePaddle/PaddleOCR.git" pdf2image opencc-python-reimplemented prettytable lxml
```

## 使用方式

### 基本 OCR

將 PDF 轉換為純文字（保留段落結構）：

```bash
uv run python main.py document.pdf
```

輸出檔案預設為 `document.txt`。

### 表格識別模式

啟用表格識別，將表格輸出為 ASCII 格式：

```bash
uv run python main.py document.pdf --table
```

### 簡體轉繁體（台灣用語）

將簡體中文 PDF 轉換為繁體中文（台灣用語）：

```bash
uv run python pdf_to_tw.py document.pdf
```

輸出檔案預設為 `document_tw.txt`。

### LLM 翻譯

使用 LLM 進行文件翻譯（需設定 API Key）：

```bash
# 使用百度千帆 API
uv run python pdf_translate.py document.pdf --api-key YOUR_API_KEY

# 使用本地 Ollama 服務
uv run python pdf_translate.py document.pdf \
    --api-key ollama \
    --api-base-url http://localhost:11434/v1 \
    --model qwen2.5:7b

# 翻譯成英文
uv run python pdf_translate.py document.pdf --target-lang en --api-key YOUR_API_KEY
```

## 命令列選項

### main.py（核心 OCR 工具）

| 選項 | 說明 |
|------|------|
| `-o, --output` | 輸出檔案路徑 |
| `--dpi` | PDF 轉圖片解析度（預設：300） |
| `--lang` | OCR 語言（ch/en/japan，預設：ch） |
| `--table` | 啟用表格識別模式 |
| `--para-gap` | 段落間距閾值（預設：1.5） |
| `--no-paragraph` | 不保留段落結構 |
| `--no-separator` | 不加入頁面分隔線 |
| `--cpu` | 強制使用 CPU |

### pdf_to_tw.py（簡繁轉換）

| 選項 | 說明 |
|------|------|
| `-o, --output` | 輸出檔案路徑 |
| `--dpi` | PDF 轉圖片解析度（預設：300） |
| `--no-translate` | 只做 OCR，不轉換簡繁 |
| `--no-separator` | 不加入頁面分隔線 |
| `--cpu` | 強制使用 CPU |

### pdf_translate.py（LLM 翻譯）

| 選項 | 說明 |
|------|------|
| `-o, --output` | 輸出目錄 |
| `-t, --target-lang` | 目標語言代碼（預設：zh-TW） |
| `--api-key` | LLM API Key |
| `--api-base-url` | LLM API 基礎 URL |
| `--model` | 模型名稱（預設：ernie-3.5-8k） |
| `--chunk-size` | 分塊翻譯字符閾值（預設：5000） |
| `--cpu` | 強制使用 CPU |

## 支援的語言代碼（翻譯功能）

| 代碼 | 語言 |
|------|------|
| `zh-TW` | 繁體中文（台灣） |
| `zh-CN` | 簡體中文 |
| `en` | 英文 |
| `ja` | 日文 |
| `ko` | 韓文 |
| `fr` | 法文 |
| `de` | 德文 |
| `es` | 西班牙文 |

## 專案結構

```
imagepdf2txt/
├── main.py          # 核心 OCR 工具
├── pdf_to_tw.py     # 簡體轉繁體工具
├── pdf_translate.py # LLM 翻譯工具
├── pyproject.toml   # 專案配置
└── README.md        # 說明文件
```

## 技術說明

- 使用 **PaddleOCR 3.x** 進行 OCR 識別
- 使用 **PPStructureV3** 進行表格識別與版面分析
- 使用 **OpenCC** 進行簡繁轉換（s2twp 配置）
- 使用 **pdf2image** 將 PDF 轉換為圖片
- 表格輸出使用 **prettytable** 格式化

## 授權

MIT License
