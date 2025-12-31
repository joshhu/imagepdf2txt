#!/usr/bin/env python3
"""
PDF 文檔翻譯工具

使用 PaddleOCR 的 PP-DocTranslation 管線進行文檔翻譯。
支援 PDF、圖片等多種格式，翻譯結果保存為 Markdown 格式。

需要設定 LLM API Key（支援百度千帆平台或 OpenAI 兼容的本地服務）。
"""

import argparse
import os
import sys
from pathlib import Path

from paddleocr import PPDocTranslation


def translate_document(
    input_path: Path,
    output_dir: Path,
    target_language: str = "zh-TW",
    api_key: str = None,
    api_base_url: str = None,
    model_name: str = "ernie-3.5-8k",
    use_gpu: bool = True,
    chunk_size: int = 5000,
):
    """
    翻譯文檔

    Args:
        input_path: 輸入檔案路徑（PDF 或圖片）
        output_dir: 輸出目錄
        target_language: 目標語言代碼（預設: zh-TW 繁體中文）
        api_key: LLM API Key
        api_base_url: LLM API 基礎 URL（可選）
        model_name: 使用的模型名稱
        use_gpu: 是否使用 GPU
        chunk_size: 分塊翻譯的字符閾值
    """
    # 設定設備
    device = "gpu" if use_gpu else "cpu"
    device_info = "GPU" if use_gpu else "CPU"

    print(f"輸入檔案: {input_path}")
    print(f"輸出目錄: {output_dir}")
    print(f"目標語言: {target_language}")
    print(f"使用設備: {device_info}")
    print("-" * 50)

    # 初始化 PP-DocTranslation 管線
    print("正在初始化 PP-DocTranslation 管線...")
    pipeline = PPDocTranslation(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    # 視覺預測（OCR + 佈局分析）
    print("正在進行文檔解析...")
    visual_predict_res = pipeline.visual_predict(
        str(input_path),
        use_doc_orientation_classify=False,
    )

    # 收集 Markdown 內容
    ori_md_info_list = []
    for i, res in enumerate(visual_predict_res, 1):
        print(f"  處理第 {i} 頁...")
        if "layout_parsing_result" in res:
            ori_md_info_list.append(res["layout_parsing_result"])

    if not ori_md_info_list:
        print("錯誤: 無法解析文檔內容", file=sys.stderr)
        sys.exit(1)

    print(f"共解析 {len(ori_md_info_list)} 頁")

    # 設定 LLM 配置
    chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": model_name,
        "api_type": "openai",
        "api_key": api_key,
    }

    if api_base_url:
        chat_bot_config["base_url"] = api_base_url

    # 執行翻譯
    print(f"正在翻譯至 {target_language}...")
    tgt_md_info_list = pipeline.translate(
        ori_md_info_list=ori_md_info_list,
        target_language=target_language,
        chunk_size=chunk_size,
        chat_bot_config=chat_bot_config,
    )

    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)

    # 儲存結果
    print("正在儲存翻譯結果...")
    for i, tgt_md_info in enumerate(tgt_md_info_list, 1):
        output_file = output_dir / f"page_{i:03d}.md"
        tgt_md_info.save_to_markdown(str(output_file))
        print(f"  已儲存: {output_file}")

    # 合併所有頁面到單一檔案
    combined_output = output_dir / "combined.md"
    with open(combined_output, "w", encoding="utf-8") as f:
        for i, tgt_md_info in enumerate(tgt_md_info_list, 1):
            if i > 1:
                f.write("\n\n---\n\n")
            f.write(f"## 第 {i} 頁\n\n")
            f.write(tgt_md_info.markdown if hasattr(tgt_md_info, 'markdown') else str(tgt_md_info))
            f.write("\n")

    print(f"已儲存合併檔案: {combined_output}")
    print("-" * 50)
    print("翻譯完成！")


def main():
    parser = argparse.ArgumentParser(
        description="使用 PP-DocTranslation 翻譯 PDF 文檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 使用百度千帆 API 翻譯成繁體中文
  python pdf_translate.py document.pdf --api-key YOUR_API_KEY

  # 使用本地 Ollama 服務
  python pdf_translate.py document.pdf \\
      --api-key ollama \\
      --api-base-url http://localhost:11434/v1 \\
      --model qwen2.5:7b

  # 翻譯成英文
  python pdf_translate.py document.pdf --target-lang en --api-key YOUR_API_KEY

支援的語言代碼（ISO 639-1）:
  zh-TW  繁體中文（台灣）
  zh-CN  簡體中文
  en     英文
  ja     日文
  ko     韓文
  fr     法文
  de     德文
  es     西班牙文
        """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="要翻譯的檔案路徑（PDF 或圖片）"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="輸出目錄（預設為輸入檔名_translated）"
    )

    parser.add_argument(
        "-t", "--target-lang",
        type=str,
        default="zh-TW",
        help="目標語言代碼（預設: zh-TW 繁體中文）"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="LLM API Key（也可透過環境變數 LLM_API_KEY 設定）"
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="LLM API 基礎 URL（用於本地服務，如 Ollama）"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ernie-3.5-8k",
        help="使用的模型名稱（預設: ernie-3.5-8k）"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="分塊翻譯的字符閾值（預設: 5000）"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU（預設使用 GPU 加速）"
    )

    args = parser.parse_args()

    # 檢查輸入檔案
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"錯誤: 找不到檔案 '{input_path}'", file=sys.stderr)
        sys.exit(1)

    # 設定輸出目錄
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_translated"

    # 取得 API Key
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("錯誤: 請提供 API Key（使用 --api-key 參數或設定環境變數 LLM_API_KEY）", file=sys.stderr)
        print("\n提示: 您可以使用以下方式：")
        print("  1. 百度千帆平台 API: https://qianfan.baidubce.com/")
        print("  2. 本地 Ollama 服務: --api-base-url http://localhost:11434/v1 --model qwen2.5:7b")
        sys.exit(1)

    # 執行翻譯
    translate_document(
        input_path=input_path,
        output_dir=output_dir,
        target_language=args.target_lang,
        api_key=api_key,
        api_base_url=args.api_base_url,
        model_name=args.model,
        use_gpu=not args.cpu,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
