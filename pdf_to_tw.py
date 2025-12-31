#!/usr/bin/env python3
"""
PDF 轉繁體中文工具

使用 PaddleOCR 識別 PDF 內容，並翻譯成繁體中文（台灣用語）。
預設啟用表格識別模式。
"""

import argparse
import sys
from pathlib import Path

from pdf2image import convert_from_path
from paddleocr import PaddleOCR, PPStructureV3
from opencc import OpenCC


# 初始化 OpenCC 轉換器（簡體中文 → 繁體中文台灣用語）
cc = OpenCC('s2twp')


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list:
    """將 PDF 檔案轉換為圖片列表"""
    print(f"正在將 PDF 轉換為圖片（DPI: {dpi}）...")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"共轉換 {len(images)} 頁")
    return images


def html_table_to_ascii(html_str: str) -> str:
    """將 HTML 表格轉換為 ASCII 表格"""
    from prettytable import PrettyTable
    from lxml import etree

    def get_text(element):
        return ''.join(element.itertext()).strip()

    try:
        root = etree.HTML(html_str)
        tables = root.xpath('//table')
        if not tables:
            return html_str

        result = []
        for table in tables:
            pt = PrettyTable()
            pt.header = False

            rows = table.xpath('.//tr')
            max_cols = 0
            all_rows = []

            for row in rows:
                cells = row.xpath('.//td|.//th')
                row_data = [get_text(cell) for cell in cells]
                if row_data:
                    all_rows.append(row_data)
                    max_cols = max(max_cols, len(row_data))

            if all_rows and max_cols > 0:
                pt.field_names = [f'col{i}' for i in range(max_cols)]
                for row_data in all_rows:
                    while len(row_data) < max_cols:
                        row_data.append('')
                    pt.add_row(row_data[:max_cols])
                result.append(pt.get_string())

        return '\n\n'.join(result) if result else html_str
    except Exception as e:
        return f'[表格解析錯誤: {e}]'


def to_traditional_chinese(text: str) -> str:
    """
    將文字轉換為繁體中文（台灣用語）

    使用 OpenCC 的 s2twp 配置：
    - 簡體中文 → 繁體中文
    - 中國用語 → 台灣用語
    """
    if not text:
        return text
    return cc.convert(text)


def structure_ocr_images(images: list, pipeline: PPStructureV3, translate: bool = True) -> list[str]:
    """
    對圖片列表進行結構化 OCR 識別（支援表格）

    Args:
        images: PIL Image 物件列表
        pipeline: PPStructureV3 引擎實例
        translate: 是否翻譯成繁體中文

    Returns:
        每頁識別出的文字列表
    """
    import numpy as np

    all_text = []

    for i, image in enumerate(images, 1):
        print(f"正在識別第 {i}/{len(images)} 頁...")

        image_array = np.array(image)
        result = pipeline.predict(image_array)

        if not result:
            all_text.append("")
            continue

        res = result[0]
        page_parts = []

        # 從 parsing_res_list 獲取所有區塊（按順序）
        if 'parsing_res_list' in res:
            for block in res['parsing_res_list']:
                block_label = getattr(block, 'label', '')
                block_content = getattr(block, 'content', '') or ''

                if block_label == 'table':
                    if block_content and '<table' in block_content.lower():
                        ascii_table = html_table_to_ascii(block_content)
                        page_parts.append(ascii_table)
                    elif block_content:
                        page_parts.append(block_content.strip())
                else:
                    if block_content and block_content.strip():
                        page_parts.append(block_content.strip())

        # 如果 parsing_res_list 沒有內容，嘗試從 table_res_list 直接獲取表格
        if not page_parts and 'table_res_list' in res:
            for table in res['table_res_list']:
                if hasattr(table, 'pred_html'):
                    ascii_table = html_table_to_ascii(table.pred_html)
                    page_parts.append(ascii_table)
                elif isinstance(table, dict) and 'pred_html' in table:
                    ascii_table = html_table_to_ascii(table['pred_html'])
                    page_parts.append(ascii_table)

        # 如果還是沒有內容，嘗試從 overall_ocr_res 獲取文字
        if not page_parts and 'overall_ocr_res' in res:
            ocr_res = res['overall_ocr_res']
            if 'rec_texts' in ocr_res:
                page_parts.extend(ocr_res['rec_texts'])

        page_text = '\n\n'.join(page_parts)

        # 翻譯成繁體中文
        if translate and page_text:
            print(f"  翻譯第 {i} 頁...")
            page_text = to_traditional_chinese(page_text)

        all_text.append(page_text)

    return all_text


def save_text(texts: list[str], output_path: Path, page_separator: bool = True):
    """將識別結果儲存為 TXT 檔案"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts, 1):
            if page_separator and i > 1:
                f.write(f"\n{'='*50}\n")
                f.write(f"=== 第 {i} 頁 ===\n")
                f.write(f"{'='*50}\n\n")
            elif i == 1 and page_separator:
                f.write(f"{'='*50}\n")
                f.write(f"=== 第 {i} 頁 ===\n")
                f.write(f"{'='*50}\n\n")

            f.write(text)
            f.write("\n")

    print(f"已儲存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF 轉繁體中文（台灣用語），支援表格識別",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python pdf_to_tw.py document.pdf                # 基本用法
  python pdf_to_tw.py document.pdf -o output.txt  # 指定輸出檔案
  python pdf_to_tw.py document.pdf --no-translate # 不翻譯，只做 OCR
  python pdf_to_tw.py document.pdf --dpi 200      # 調整解析度
        """
    )

    parser.add_argument(
        "pdf_file",
        type=str,
        help="要轉換的 PDF 檔案路徑"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="輸出的 TXT 檔案路徑（預設為 PDF 檔名加上 _tw.txt）"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF 轉圖片的解析度（預設: 300）"
    )

    parser.add_argument(
        "--no-separator",
        action="store_true",
        help="不在頁面之間加入分隔線"
    )

    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="不翻譯，只做 OCR 識別"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU（預設使用 GPU 加速）"
    )

    args = parser.parse_args()

    # 檢查 PDF 檔案是否存在
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"錯誤: 找不到檔案 '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    # 設定輸出檔案路徑
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pdf_path.with_stem(pdf_path.stem + "_tw").with_suffix(".txt")

    print(f"輸入檔案: {pdf_path}")
    print(f"輸出檔案: {output_path}")
    print(f"翻譯: {'關閉' if args.no_translate else '簡體中文 → 繁體中文（台灣用語）'}")
    print("-" * 50)

    # 初始化 PP-StructureV3（表格識別模式）
    device = "cpu" if args.cpu else "gpu"
    device_info = "CPU" if args.cpu else "GPU"
    print(f"正在初始化 PP-StructureV3 引擎（使用 {device_info}）...")

    pipeline = PPStructureV3(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=True,
        lang='ch',
    )

    # PDF 轉圖片
    images = pdf_to_images(pdf_path, dpi=args.dpi)

    # OCR 識別並翻譯
    texts = structure_ocr_images(
        images,
        pipeline,
        translate=not args.no_translate
    )

    # 儲存結果
    save_text(texts, output_path, page_separator=not args.no_separator)

    print("-" * 50)
    print("轉換完成！")


if __name__ == "__main__":
    main()
