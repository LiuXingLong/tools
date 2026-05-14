import fitz, io, json, sys, argparse
from fontTools.ttLib import TTCollection

parser = argparse.ArgumentParser(description="PDF 文本替换工具")
parser.add_argument("input_pdf", help="输入 PDF 文件路径")
parser.add_argument(
    "output_pdf", nargs="?", help="输出 PDF 文件路径（默认: 输入文件名_output.pdf）"
)
parser.add_argument(
    "-r",
    "--replacements",
    nargs="+",
    action="append",
    default=[],
    metavar=("旧文本", "新文本"),
    help="替换对，格式: 旧文本 新文本，可重复使用",
)
parser.add_argument(
    "-j",
    "--json",
    type=str,
    help='从 JSON 文件读取替换映射（格式: {"旧": "新"}）',
)
parser.add_argument(
    "-p",
    "--page",
    type=int,
    default=0,
    help="目标页码，从 0 开始（默认 0）",
)
parser.add_argument(
    "-f",
    "--fontsize",
    type=float,
    default=9,
    help="替换文本字号（默认 9）",
)
args = parser.parse_args()

if args.output_pdf is None:
    base = args.input_pdf
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    args.output_pdf = f"{base}_output.pdf"

replacements = {}
if args.json:
    with open(args.json, encoding="utf-8") as f:
        replacements.update(json.load(f))
for pair in args.replacements:
    if len(pair) != 2:
        print(f"警告: 替换参数需成对出现，跳过: {pair}", file=sys.stderr)
        continue
    replacements[pair[0]] = pair[1]

if not replacements:
    print("错误: 未指定任何替换内容", file=sys.stderr)
    sys.exit(1)

# 提取宋体常规体（与原文件 Sun-ExtA 风格一致）
ttc = TTCollection("/System/Library/Fonts/Supplemental/Songti.ttc")
buf = io.BytesIO()
ttc[6].save(buf)
songti_buf = buf.getvalue()

doc = fitz.open(args.input_pdf)
page = doc[args.page]

replace_info = []
for old_text, new_text in replacements.items():
    for rect in page.search_for(old_text):
        replace_info.append((rect, new_text))

page.insert_font(fontbuffer=songti_buf, fontname="SongtiSC")

shape = page.new_shape()
for rect, new_text in replace_info:
    cover = rect + (-0.5, -0.5, 0.5, 0.5)
    shape.draw_rect(cover)
    shape.finish(fill=(1, 1, 1), color=None, fill_opacity=1)
    x = rect.x0
    y = rect.y0 + (rect.y1 - rect.y0) * 0.85
    shape.insert_text(
        (x, y),
        new_text,
        fontsize=args.fontsize,
        color=(0, 0, 0),
        fontname="SongtiSC",
    )
shape.commit()

doc.save(args.output_pdf)
doc.close()
print(f"替换完成，已保存至: {args.output_pdf}")
