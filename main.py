#!/usr/bin/env python3
"""
PDF 轉 TXT 工具

使用 PaddleOCR 將 PDF 檔案中的文字識別出來並儲存為 TXT 檔案。
支援中文（繁體/簡體）、英文、日文等多種語言。
"""

import argparse
import sys
from pathlib import Path

from pdf2image import convert_from_path
from paddleocr import PaddleOCR, PPStructureV3


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list:
    """
    將 PDF 檔案轉換為圖片列表

    Args:
        pdf_path: PDF 檔案路徑
        dpi: 轉換解析度，預設 300

    Returns:
        圖片列表（PIL Image 物件）
    """
    print(f"正在將 PDF 轉換為圖片（DPI: {dpi}）...")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    print(f"共轉換 {len(images)} 頁")
    return images


def extract_text_with_positions(result) -> list[tuple]:
    """
    從 OCR 結果中提取文字和位置資訊

    Args:
        result: PaddleOCR 的識別結果

    Returns:
        包含 (文字, y_min, y_max, x_min) 的列表
    """
    text_items = []

    for res in result:
        # PaddleOCR 3.x 回傳類似字典的物件
        if 'dt_polys' in res and 'rec_texts' in res:
            dt_polys = res['dt_polys']
            rec_texts = res['rec_texts']
            for poly, text in zip(dt_polys, rec_texts):
                if text.strip():
                    # poly 是四個角的座標 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    y_coords = [p[1] for p in poly]
                    x_coords = [p[0] for p in poly]
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    x_min = min(x_coords)
                    text_items.append((text, y_min, y_max, x_min))

    return text_items


def group_into_paragraphs(text_items: list[tuple], line_threshold: float = 0.5, para_threshold: float = 1.5) -> str:
    """
    根據位置資訊將文字分組成段落

    Args:
        text_items: (文字, y_min, y_max, x_min) 的列表
        line_threshold: 判斷同一行的 Y 座標容差比例（相對於行高）
        para_threshold: 判斷新段落的 Y 間距比例（相對於行高）

    Returns:
        保留段落結構的文字字串
    """
    if not text_items:
        return ""

    # 按 Y 座標排序（從上到下），同一行按 X 座標排序（從左到右）
    sorted_items = sorted(text_items, key=lambda x: (x[1], x[3]))

    # 計算平均行高
    avg_line_height = sum(item[2] - item[1] for item in sorted_items) / len(sorted_items)

    paragraphs = []
    current_line = []
    current_line_y = None
    prev_line_y_max = None

    for text, y_min, y_max, x_min in sorted_items:
        line_height = y_max - y_min

        if current_line_y is None:
            # 第一個文字框
            current_line = [text]
            current_line_y = y_min
            prev_line_y_max = y_max
        elif abs(y_min - current_line_y) < avg_line_height * line_threshold:
            # 同一行（Y 座標接近）
            current_line.append(text)
        else:
            # 新的一行
            line_text = "".join(current_line)

            # 判斷是否為新段落（與上一行的間距）
            gap = y_min - prev_line_y_max
            is_new_paragraph = gap > avg_line_height * para_threshold

            if is_new_paragraph and paragraphs:
                # 新段落：加入空行
                paragraphs.append("")

            paragraphs.append(line_text)

            current_line = [text]
            current_line_y = y_min
            prev_line_y_max = y_max

    # 處理最後一行
    if current_line:
        paragraphs.append("".join(current_line))

    return "\n".join(paragraphs)


def ocr_images(
    images: list,
    ocr_engine: PaddleOCR,
    preserve_paragraphs: bool = True,
    para_threshold: float = 1.5
) -> list[str]:
    """
    對圖片列表進行 OCR 識別

    Args:
        images: PIL Image 物件列表
        ocr_engine: PaddleOCR 引擎實例
        preserve_paragraphs: 是否保留段落結構
        para_threshold: 段落間距閾值（相對於行高的倍數）

    Returns:
        每頁識別出的文字列表
    """
    import numpy as np

    all_text = []

    for i, image in enumerate(images, 1):
        print(f"正在識別第 {i}/{len(images)} 頁...")

        # 將 PIL Image 轉換為 numpy array
        image_array = np.array(image)

        # 執行 OCR 識別
        result = ocr_engine.predict(image_array)

        if preserve_paragraphs:
            # 提取文字和位置，保留段落結構
            text_items = extract_text_with_positions(result)
            page_text = group_into_paragraphs(text_items, para_threshold=para_threshold)
        else:
            # 簡單模式：直接連接所有文字
            page_text = []
            for res in result:
                # PaddleOCR 3.x 回傳類似字典的物件
                if 'rec_texts' in res and res['rec_texts']:
                    page_text.extend(res['rec_texts'])
            page_text = "\n".join(page_text)

        all_text.append(page_text)

    return all_text


def html_table_to_ascii(html_str: str) -> str:
    """
    將 HTML 表格轉換為 ASCII 表格

    Args:
        html_str: HTML 表格字串

    Returns:
        ASCII 格式的表格字串
    """
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


def structure_ocr_images(images: list, pipeline: PPStructureV3) -> list[str]:
    """
    對圖片列表進行結構化 OCR 識別（支援表格）

    Args:
        images: PIL Image 物件列表
        pipeline: PPStructureV3 引擎實例

    Returns:
        每頁識別出的文字列表（表格為 ASCII 格式）
    """
    import numpy as np

    all_text = []

    for i, image in enumerate(images, 1):
        print(f"正在識別第 {i}/{len(images)} 頁（結構化模式）...")

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
                # LayoutBlock 物件使用 .label 和 .content 屬性
                block_label = getattr(block, 'label', '')
                block_content = getattr(block, 'content', '') or ''

                if block_label == 'table':
                    # 表格區塊：內容可能是 HTML 格式
                    if block_content and '<table' in block_content.lower():
                        ascii_table = html_table_to_ascii(block_content)
                        page_parts.append(ascii_table)
                    elif block_content:
                        page_parts.append(block_content.strip())
                else:
                    # 非表格區塊：直接使用文字內容
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

        all_text.append('\n\n'.join(page_parts))

    return all_text


def save_text(texts: list[str], output_path: Path, page_separator: bool = True):
    """
    將識別結果儲存為 TXT 檔案

    Args:
        texts: 每頁文字列表
        output_path: 輸出檔案路徑
        page_separator: 是否在頁面之間加入分隔線
    """
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
        description="使用 PaddleOCR 將 PDF 轉換為 TXT 文字檔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python main.py document.pdf                    # 基本用法（保留段落）
  python main.py document.pdf -o output.txt     # 指定輸出檔案
  python main.py document.pdf --dpi 200         # 調整解析度
  python main.py document.pdf --no-separator    # 不加入頁面分隔線
  python main.py document.pdf --para-gap 2.0    # 調整段落間距敏感度
  python main.py document.pdf --no-paragraph    # 不保留段落結構
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
        help="輸出的 TXT 檔案路徑（預設為 PDF 檔名加上 .txt）"
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
        "--lang",
        type=str,
        default="ch",
        help="OCR 語言設定（ch: 中文, en: 英文, japan: 日文，預設: ch）"
    )

    parser.add_argument(
        "--no-paragraph",
        action="store_true",
        help="不保留段落結構，將所有文字逐行輸出"
    )

    parser.add_argument(
        "--para-gap",
        type=float,
        default=1.5,
        help="段落間距閾值（相對於行高的倍數，預設: 1.5）"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="強制使用 CPU（預設使用 GPU 加速）"
    )

    parser.add_argument(
        "--table",
        action="store_true",
        help="啟用表格識別模式，使用 ASCII 表格格式保留表格結構"
    )

    args = parser.parse_args()

    # 檢查 PDF 檔案是否存在
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"錯誤: 找不到檔案 '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    if not pdf_path.suffix.lower() == ".pdf":
        print(f"警告: '{pdf_path}' 可能不是 PDF 檔案", file=sys.stderr)

    # 設定輸出檔案路徑
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = pdf_path.with_suffix(".txt")

    print(f"輸入檔案: {pdf_path}")
    print(f"輸出檔案: {output_path}")
    if args.table:
        print("模式: 表格識別（ASCII 表格格式）")
    print("-" * 50)

    # 初始化引擎
    device = "cpu" if args.cpu else "gpu"
    device_info = "CPU" if args.cpu else "GPU"

    if args.table:
        # 使用 PP-StructureV3 進行結構化識別（支援表格）
        print(f"正在初始化 PP-StructureV3 引擎（使用 {device_info}）...")
        engine = PPStructureV3(
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_table_recognition=True,
            lang=args.lang,
        )
    else:
        # 使用 PaddleOCR 進行一般 OCR
        print(f"正在初始化 PaddleOCR 引擎（使用 {device_info}）...")
        engine = PaddleOCR(
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=args.lang,
        )

    # PDF 轉圖片
    images = pdf_to_images(pdf_path, dpi=args.dpi)

    # OCR 識別
    if args.table:
        # 表格模式：使用 PP-StructureV3
        texts = structure_ocr_images(images, engine)
    else:
        # 一般模式：使用 PaddleOCR
        preserve_paragraphs = not args.no_paragraph
        texts = ocr_images(
            images,
            engine,
            preserve_paragraphs=preserve_paragraphs,
            para_threshold=args.para_gap
        )

    # 儲存結果
    save_text(texts, output_path, page_separator=not args.no_separator)

    print("-" * 50)
    print("轉換完成！")


if __name__ == "__main__":
    main()
