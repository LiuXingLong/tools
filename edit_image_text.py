"""图片文字修改工具 - 支持替换、删除和添加图片中的文字"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_pytesseract():
    """延迟导入 pytesseract，避免 numpy 版本冲突"""
    try:
        import pytesseract

        return pytesseract
    except (ImportError, ValueError):
        return None


def _try_download_simhei() -> str:
    """尝试下载 SimHei 字体"""
    target = os.path.expanduser("~/Library/Fonts/SimHei.ttf")
    if os.path.exists(target) and os.path.getsize(target) > 1000:
        return target
    try:
        import ssl, urllib.request

        url = (
            "https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf"
        )
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = resp.read()
            if len(data) > 1000:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(data)
                print(f"已下载 SimHei 字体: {target}")
                return target
    except Exception:
        pass
    return ""


def _find_system_font() -> str:
    """自动查找系统中的中文字体（优先 Noto Sans SC，其次 SimHei/STXihei）"""
    candidates = [
        os.path.expanduser("~/Library/Fonts/NotoSansSC-Variable.ttf"),
        os.path.expanduser("~/Library/Fonts/SimHei.ttf"),
        "/Library/Fonts/SimHei.ttf",
        "/System/Library/Fonts/Supplemental/SimHei.ttf",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/SimHei.ttf",
        "/Library/Fonts/SimHei.ttf",
        os.path.expanduser("~/Library/Fonts/SimHei.ttf"),
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return _try_download_simhei()


def find_text_boxes(
    image_path: str,
    target_text: str = "",
    lang: str = "chi_sim+eng",
    exact_match: bool = True,
) -> list[dict]:
    """使用 OCR 查找图片中的文字位置

    Args:
        image_path: 图片路径
        target_text: 目标文字（为空则返回所有检测到的文字）
        lang: OCR 语言
        exact_match: 是否精确匹配（否则子串匹配）

    Returns:
        包含文字、位置、尺寸的字典列表
    """
    pt = _get_pytesseract()
    if pt is None:
        raise ImportError("请安装 pytesseract: pip install pytesseract")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pt.image_to_data(rgb, lang=lang, output_type=pt.Output.DICT)

    boxes = []
    for i in range(len(results["text"])):
        text = results["text"][i].strip()
        if not text:
            continue
        if target_text:
            if exact_match:
                if text != target_text:
                    continue
            else:
                if target_text not in text:
                    continue

        x, y, w, h = (
            results["left"][i],
            results["top"][i],
            results["width"][i],
            results["height"][i],
        )
        conf = int(results["conf"][i])

        boxes.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": conf,
            }
        )

    return boxes


def _sample_bg_color(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
    """从文字区域周围采样背景颜色"""
    h_img, w_img = img.shape[:2]
    margin = max(2, h // 4)
    # 在区域外扩一圈采样
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    region = img[y1:y2, x1:x2].copy()
    # 抠掉原文字区域
    roi_x = x - x1
    roi_y = y - y1
    cv2.rectangle(region, (roi_x, roi_y), (roi_x + w, roi_y + h), (0, 0, 0), -1)
    # 排除黑色掩码区域，只取非零像素
    mask = ~np.all(region == 0, axis=2)
    if mask.any():
        bg = region[mask].mean(axis=0).astype(int)
        return tuple(bg.tolist())
    return (255, 255, 255)


def _sample_text_color(img: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
    """从文字区域采样文字颜色"""
    roi = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg = np.median(gray[gray > 120]) if np.any(gray > 120) else 255
    # 取明显暗于背景的像素（文字核心）
    text_mask = gray < (bg * 0.4)
    if not text_mask.any():
        text_mask = gray < np.percentile(gray, 20)
    if text_mask.any():
        colors = roi[text_mask].mean(axis=0).astype(int)
        return tuple(colors.tolist())
    return (0, 0, 0)


def _sample_text_position(
    img: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[int, int]:
    """从原图采样文字在框内的垂直位置，返回 (top, height)"""
    roi = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bg = np.median(gray[gray > 120]) if np.any(gray > 120) else 255
    text_mask = gray < (bg * 0.35)
    if not text_mask.any():
        text_mask = gray < np.percentile(gray, 15)
    rows = np.any(text_mask, axis=1)
    if rows.any():
        text_top = int(np.argmax(rows))
        text_bottom = int(rows.shape[0] - np.argmax(rows[::-1]) - 1)
        return text_top, text_bottom - text_top
    return max(0, h // 4), h // 2


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """按指定角度旋转图片（正=逆时针），自动扩大画布"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - center[0]
    matrix[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(
        img, matrix, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _deskew(img: np.ndarray, max_angle: float = 1.0) -> tuple[np.ndarray, float]:
    """自动检测并校正图片倾斜

    Args:
        img: OpenCV 图片数组 (BGR)
        max_angle: 最大检测角度

    Returns:
        (校正后的图片, 旋转角度)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return img, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    angle = -angle

    if abs(angle) < 0.1:
        return img, 0.0
    if abs(angle) > max_angle:
        angle = 0.0
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 扩大画布防止裁剪
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - center[0]
    matrix[1, 2] += nh / 2 - center[1]
    rotated = cv2.warpAffine(
        img, matrix, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


# 中文字号（号数）到磅值（pt）的映射
_HAO_SIZE_MAP = {
    "初号": 42,
    "小初": 36,
    "一号": 26,
    "小一": 24,
    "二号": 22,
    "小二": 18,
    "三号": 16,
    "小三": 15,
    "四号": 14,
    "小四": 12,
    "五号": 10.5,
    "小五": 9,
    "六号": 7.5,
    "小六": 6.5,
    "七号": 5.5,
    "八号": 5,
}


def parse_font_size(size_val: str | int) -> float:
    """解析字体大小，支持 pt 数值或中文字号（如 '六号'、'6号'、'五号'）

    Args:
        size_val: 字体大小值，可以是 int 或 '六号'/'6号' 等中文格式

    Returns:
        磅值（pt）
    """
    if isinstance(size_val, (int, float)):
        return float(size_val)
    s = str(size_val).strip()
    for hao, pt in _HAO_SIZE_MAP.items():
        if s == hao or s == hao.replace("号", "号"):
            return pt
    # 尝试纯数字+"号"格式，如 "6号"
    import re

    m = re.match(r"^(\d+)号$", s)
    if m:
        num = int(m.group(1))
        # 反向查找映射
        for hao, pt in _HAO_SIZE_MAP.items():
            if hao.startswith(str(num)) and hao != f"小{num}":
                return pt
        # 无精确匹配则保守处理：6号 → 7.5pt
        fallback = {1: 26, 2: 22, 3: 16, 4: 14, 5: 10.5, 6: 7.5, 7: 5.5, 8: 5}
        if num in fallback:
            return fallback[num]
    try:
        return float(s)
    except ValueError:
        print(f"警告: 无法解析字体大小 '{size_val}'，使用默认值 12pt")
        return 12.0


def _calc_font_size(font_path: str, text: str, box_w: int, box_h: int) -> int:
    """计算适合放入指定区域的字体大小（二分搜索）"""
    if not font_path:
        return max(box_h - 4, 12)
    low, high = 4, box_h * 2
    best = max(box_h - 4, 12)
    for _ in range(12):
        mid = (low + high) // 2
        font = ImageFont.truetype(font_path, mid)
        left, top, right, bottom = font.getbbox(text)
        tw, th = right - left, bottom - top
        if tw <= box_w - 2 and th <= box_h - 2:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def remove_text_region(
    img: np.ndarray, x: int, y: int, w: int, h: int, method: str = "fill"
) -> np.ndarray:
    """移除图片指定区域的文字"""
    if method == "fill":
        bg_color = _sample_bg_color(img, x, y, w, h)
        img[y : y + h, x : x + w] = bg_color
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pad = max(1, h // 20)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        mask[y1:y2, x1:x2] = 255
        img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return img


def draw_text_pil(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    box_w: int = 0,
    box_h: int = 0,
    font_path: str = "",
    font_size: int = 0,
    color: tuple = (0, 0, 0),
    orig_img: np.ndarray = None,
    blur: float = 0,
    bold: int = 0,
    weight: str = "",
    letter_spacing: float = 0,
) -> np.ndarray:
    """使用 Pillow 在图片上绘制文字

    Args:
        img: OpenCV 图片数组 (BGR)
        text: 要绘制的文字
        x, y: 区域左上角坐标
        box_w, box_h: 区域宽高（用于计算字体大小和居中）
        font_path: 字体文件路径（为空自动查找系统字体）
        font_size: 字体大小（0 则自动适配区域高度）
        color: 文字颜色 RGB
        orig_img: 原图（用于采样文字位置，None 则垂直居中）
        blur: 模糊强度（0=不模糊，建议 0.5~1.5 让文字不那么清晰锐利）
        bold: 加粗强度（0=正常，每次加粗重绘 2 次偏移，建议 1~3）
        weight: 可变字体字重（如 'Thin', 'ExtraLight', 'Light', 'Regular', 'Medium', 'Bold'）
        letter_spacing: 字间距（像素），0=正常，正值加大间距

    Returns:
        处理后的图片
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    if not font_path or not os.path.exists(font_path):
        font_path = _find_system_font()

    if font_path:
        fs = (
            font_size
            if font_size > 0
            else _calc_font_size(font_path, text, box_w or 9999, box_h or 32)
        )
        font = ImageFont.truetype(font_path, fs)
        if weight:
            try:
                font.set_variation_by_name(weight.encode())
            except Exception:
                try:
                    font.set_variation_by_name(weight)
                except Exception:
                    print(f"  警告: 字体不支持字重 '{weight}'，使用默认")
    else:
        font = ImageFont.load_default()
        fs = font_size

    # 确定垂直位置：匹配原文字的中心
    if orig_img is not None and box_h > 0:
        orig_top, orig_th = _sample_text_position(orig_img, x, y, box_w, box_h)
        orig_center = y + orig_top + orig_th // 2
        if font_path:
            l, ft, r, fb = font.getbbox(text)
            new_h = fb - ft
            draw_y = orig_center - new_h // 2 - ft
        else:
            draw_y = orig_center - box_h // 4
    else:
        draw_y = y

    def _draw_text_char(dx: int = 0, dy: int = 0):
        if abs(letter_spacing) < 0.5:
            draw.text((x + dx, draw_y + dy), text, fill=color, font=font)
        else:
            cx = x + dx
            for char in text:
                draw.text((cx, draw_y + dy), char, fill=color, font=font)
                b = font.getbbox(char)
                cw = (b[2] - b[0]) + letter_spacing
                cx += cw

    # 合成加粗：通过多次偏移绘制实现笔画增粗
    if bold > 0 and font_path:
        offsets = []
        for i in range(1, bold + 1):
            offsets.extend([(i, 0), (-i, 0), (0, i), (0, -i)])
        for dx, dy in offsets:
            _draw_text_char(dx, dy)

    _draw_text_char()
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 模糊处理：让文字边缘不那么锐利，更接近原图
    if blur > 0:
        ksize = int(blur * 2 + 1)
        if ksize % 2 == 0:
            ksize += 1
        roi = result[y : y + box_h, x : x + box_w] if box_h > 0 else result
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), blur)
        if box_h > 0:
            result[y : y + box_h, x : x + box_w] = blurred
        else:
            result = blurred

    return result


def replace_text_ocr(
    image_path: str,
    old_text: str,
    new_text: str,
    output_path: str = "",
    lang: str = "chi_sim+eng",
    method: str = "inpaint",
    font_path: str = "",
    font_size: int = 0,
    color: tuple = (0, 0, 0),
    exact_match: bool = True,
    deskew: bool = False,
    rotate: float = 0,
    blur: float = 0,
    bold: int = 0,
    weight: str = "",
    letter_spacing: float = 0,
) -> str:
    """使用 OCR 自动查找并替换文字

    Args:
        image_path: 输入图片路径
        old_text: 要替换的原文字
        new_text: 新文字
        output_path: 输出路径（默认为 input_replaced.png）
        lang: OCR 语言
        method: 移除方法
        font_path: 字体路径
        font_size: 字体大小（0 则自动适配）
        color: 文字颜色 RGB
        exact_match: 是否精确匹配
        blur: 模糊强度（0=不模糊）
        bold: 加粗强度（0=正常）
        weight: 可变字体字重（如 'Thin', 'ExtraLight', 'Light'）
        letter_spacing: 字间距（像素）

    Returns:
        输出图片路径
    """
    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_replaced{p.suffix}")

    boxes = find_text_boxes(
        image_path, target_text=old_text, lang=lang, exact_match=exact_match
    )
    if not boxes:
        print(f"未找到文字 '{old_text}'")
        return ""

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    need_rotate = False
    rotate_angle = 0.0
    orig_h, orig_w = img.shape[:2]
    if deskew or abs(rotate) > 0.01:
        if abs(rotate) > 0.01:
            rotate_angle = rotate
            need_rotate = True
            print(f"  图片将旋转 {rotate:.2f}°")
        else:
            img, angle = _deskew(img)
            if abs(angle) > 0.01:
                rotate_angle = angle
                need_rotate = True
                print(f"  图片已自动校正 {angle:.2f}°")

    if need_rotate:
        # 在原图 OCR 坐标基础上，变换到旋转后的坐标空间
        h, w = orig_h, orig_w
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        matrix[0, 2] += nw / 2 - center[0]
        matrix[1, 2] += nh / 2 - center[1]
        # 变换每个 box 的坐标
        for box in boxes:
            points = np.array(
                [
                    [box["x"], box["y"]],
                    [box["x"] + box["w"], box["y"]],
                    [box["x"], box["y"] + box["h"]],
                    [box["x"] + box["w"], box["y"] + box["h"]],
                ],
                dtype=np.float32,
            )
            transformed = cv2.transform(points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            xs = transformed[:, 0]
            ys = transformed[:, 1]
            box["x"] = int(xs.min())
            box["y"] = int(ys.min())
            box["w"] = int(xs.max() - xs.min())
            box["h"] = int(ys.max() - ys.min())
        # 旋转图片
        img = _rotate_image(img, rotate_angle)

    for box in boxes:
        print(
            f"  找到: '{box['text']}' at ({box['x']},{box['y']}) [{box['w']}x{box['h']}] "
            f"conf={box['confidence']}"
        )

        # 非精确匹配时，在原文中只替换目标子串，保留周围文字
        if not exact_match and old_text in box["text"]:
            draw_text = box["text"].replace(old_text, new_text, 1)
            print(f"  替换为: '{draw_text}'")
        else:
            draw_text = new_text

        # 自动检测文字颜色
        use_color = (
            color
            if color != (0, 0, 0)
            else _sample_text_color(img, box["x"], box["y"], box["w"], box["h"])
        )
        if use_color != color:
            print(f"  文字颜色: RGB{use_color}")
            color = use_color

        font_used = font_path
        if not font_used or not os.path.exists(font_used):
            font_used = _find_system_font()
        print(f"  使用字体: {font_used or 'PIL默认'}")

        # 在修改前保存原图区域用于位置采样
        orig_for_pos = img.copy()

        img = remove_text_region(
            img, box["x"], box["y"], box["w"], box["h"], method=method
        )

        img = draw_text_pil(
            img,
            draw_text,
            box["x"],
            box["y"],
            box_w=box["w"],
            box_h=box["h"],
            font_path=font_path,
            font_size=font_size,
            color=color,
            orig_img=orig_for_pos,
            blur=blur,
            bold=bold,
            weight=weight,
            letter_spacing=letter_spacing,
        )

    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")
    return output_path


def replace_text_manual(
    image_path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    new_text: str,
    output_path: str = "",
    method: str = "inpaint",
    font_path: str = "",
    font_size: int = 0,
    color: tuple = (0, 0, 0),
    blur: float = 0,
    bold: int = 0,
    weight: str = "",
    letter_spacing: float = 0,
) -> str:
    """在指定坐标区域替换文字

    Args:
        image_path: 输入图片路径
        x, y, w, h: 文字区域坐标和尺寸
        new_text: 新文字
        output_path: 输出路径
        method: 移除方法
        font_path: 字体路径
        font_size: 字体大小
        color: 文字颜色
        blur: 模糊强度（0=不模糊）
        bold: 加粗强度（0=正常）
        weight: 可变字体字重（如 'Thin', 'ExtraLight', 'Light'）
        letter_spacing: 字间距（像素）

    Returns:
        输出图片路径
    """
    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_replaced{p.suffix}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    img = remove_text_region(img, x, y, w, h, method=method)

    img = draw_text_pil(
        img,
        new_text,
        x,
        y,
        box_w=w,
        box_h=h,
        font_path=font_path,
        font_size=font_size,
        color=color,
        blur=blur,
        bold=bold,
        weight=weight,
        letter_spacing=letter_spacing,
    )

    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")
    return output_path


def add_text(
    image_path: str,
    text: str,
    x: int,
    y: int,
    output_path: str = "",
    font_path: str = "",
    font_size: int = 20,
    color: tuple = (0, 0, 0),
    bold: int = 0,
    weight: str = "",
    letter_spacing: float = 0,
) -> str:
    """在图片指定位置添加文字

    Args:
        image_path: 输入图片路径
        text: 要添加的文字
        x, y: 位置坐标
        output_path: 输出路径
        font_path: 字体路径
        font_size: 字体大小
        color: 文字颜色
        bold: 加粗强度（0=正常）
        weight: 可变字体字重（如 'Thin', 'ExtraLight', 'Light'）
        letter_spacing: 字间距（像素）

    Returns:
        输出图片路径
    """
    if not output_path:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_added{p.suffix}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    img = draw_text_pil(
        img,
        text,
        x,
        y,
        box_w=0,
        box_h=0,
        font_path=font_path,
        font_size=font_size,
        color=color,
        bold=bold,
        weight=weight,
        letter_spacing=letter_spacing,
    )

    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")
    return output_path


def detect_text(image_path: str, lang: str = "chi_sim+eng") -> None:
    """检测图片中的所有文字（预览模式）

    Args:
        image_path: 图片路径
        lang: OCR 语言
    """
    boxes = find_text_boxes(image_path, target_text="", lang=lang)
    if not boxes:
        print("未检测到文字")
        return

    print(f"\n检测到 {len(boxes)} 个文字区域:")
    print(f"{'文字':<20} {'位置':<20} {'尺寸':<12} {'置信度':<8}")
    print("-" * 60)
    for b in boxes:
        text_display = b["text"][:18]
        print(
            f"{text_display:<20} ({b['x']},{b['y']}){'':<12} "
            f"{b['w']}x{b['h']}{'':<6} {b['confidence']}"
        )

    # 保存标记了文字区域的预览图
    img = cv2.imread(image_path)
    for b in boxes:
        cv2.rectangle(
            img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 0), 2
        )
        cv2.putText(
            img,
            b["text"],
            (b["x"], b["y"] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    preview_path = Path(image_path).stem + "_detected.png"
    cv2.imwrite(preview_path, img)
    print(f"\n预览图已保存: {preview_path}（绿色框标记了检测到的文字区域）")


def main():
    parser = argparse.ArgumentParser(
        description="图片文字修改工具 - 替换、删除、添加图片中的文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s detect input.png                                # 检测文字位置
  %(prog)s replace input.png -o output.png --old "旧文字" --new "新文字"  # OCR 替换
   %(prog)s replace input.png -x 100 -y 200 -w 80 -h 30 --new "新文字"    # 手动指定区域
  %(prog)s replace input.png --old "2023" --new "2024"                     # 精确匹配（默认，不会误改 "2023.01.19"）
   %(prog)s replace input.png --old "2023" --new "2024" --contains          # 子串匹配（会匹配 "2023.01.19" 等）
   %(prog)s replace input.jpg --old "2023" --new "2024" --contains --deskew  # 先自动旋转校正，再替换
   %(prog)s add input.png -x 50 -y 50 --text "Hello" --size 24            # 添加文字
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # detect 子命令
    detect_p = subparsers.add_parser("detect", help="检测图片中的文字位置")
    detect_p.add_argument("image", help="图片路径")
    detect_p.add_argument(
        "--lang", default="chi_sim+eng", help="OCR 语言（默认: chi_sim+eng）"
    )

    # replace 子命令
    replace_p = subparsers.add_parser("replace", help="替换图片中的文字")
    replace_p.add_argument("image", help="图片路径")
    replace_p.add_argument("-o", "--output", default="", help="输出路径")
    replace_p.add_argument("--old", default="", help="原文字（OCR 自动查找）")
    replace_p.add_argument("--new", required=True, help="新文字")
    match_group = replace_p.add_mutually_exclusive_group()
    match_group.add_argument(
        "--exact", action="store_true", dest="exact", help="精确匹配（默认）"
    )
    match_group.add_argument(
        "--contains",
        action="store_false",
        dest="exact",
        help="子串匹配（如 '2023' 可匹配 '2023.01.19')",
    )
    replace_p.set_defaults(exact=True)
    replace_p.add_argument("-x", type=int, default=0, help="文字区域左上角 x")
    replace_p.add_argument("-y", type=int, default=0, help="文字区域左上角 y")
    replace_p.add_argument("-w", type=int, default=0, help="文字区域宽度")
    replace_p.add_argument("--height", type=int, default=0, help="文字区域高度")
    replace_p.add_argument(
        "--method",
        choices=["inpaint", "fill"],
        default="fill",
        help="移除文字方法（默认: fill）",
    )
    replace_p.add_argument("--lang", default="chi_sim+eng", help="OCR 语言")
    replace_p.add_argument("--font", default="", help="字体文件路径")
    replace_p.add_argument(
        "--size",
        default=0,
        help="字体大小（单位 pt），支持数值或中文字号如 '五号'（默认 0=自动适配框大小）",
    )
    replace_p.add_argument(
        "--bold", type=int, default=0, help="加粗强度（1~3，数值越大笔画越粗）"
    )
    replace_p.add_argument(
        "--spacing",
        type=float,
        default=0,
        help="字间距（像素，如 --spacing 0.5 增大间距，负值缩小间距）",
    )
    replace_p.add_argument(
        "--weight",
        default="",
        choices=[
            "Thin",
            "ExtraLight",
            "Light",
            "Regular",
            "Medium",
            "SemiBold",
            "Bold",
            "ExtraBold",
            "Black",
        ],
        help="可变字体字重（仅 Noto Sans SC 等可变字体支持，默认 Regular）",
    )
    replace_p.add_argument(
        "--blur",
        type=float,
        default=0,
        help="模糊强度，让文字不那么清晰锐利（建议 0.5~1.5）",
    )
    replace_p.add_argument(
        "--deskew", action="store_true", help="自动检测并校正图片倾斜（最大 5°）"
    )
    replace_p.add_argument(
        "--rotate",
        type=float,
        default=0,
        help="手动指定旋转角度（度，正=逆时针，如 --rotate 2.5）",
    )
    replace_p.add_argument(
        "--color",
        nargs=3,
        type=int,
        default=[0, 0, 0],
        metavar=("R", "G", "B"),
        help="文字颜色 RGB（默认: 0 0 0）",
    )

    # add 子命令
    add_p = subparsers.add_parser("add", help="在图片中添加文字")
    add_p.add_argument("image", help="图片路径")
    add_p.add_argument("-o", "--output", default="", help="输出路径")
    add_p.add_argument("--text", required=True, help="要添加的文字")
    add_p.add_argument("-x", type=int, default=0, help="左上角 x")
    add_p.add_argument("-y", type=int, default=0, help="左上角 y")
    add_p.add_argument("--font", default="", help="字体文件路径")
    add_p.add_argument(
        "--size", default="五号", help="字体大小，支持 pt 数值或中文字号如 '五号'"
    )
    add_p.add_argument(
        "--bold", type=int, default=0, help="加粗强度（1~3，数值越大笔画越粗）"
    )
    add_p.add_argument(
        "--spacing",
        type=float,
        default=0,
        help="字间距（像素，如 --spacing 0.5 增大间距，负值缩小间距）",
    )
    add_p.add_argument(
        "--weight",
        default="",
        choices=[
            "Thin",
            "ExtraLight",
            "Light",
            "Regular",
            "Medium",
            "SemiBold",
            "Bold",
            "ExtraBold",
            "Black",
        ],
        help="可变字体字重（仅 Noto Sans SC 等可变字体支持，默认 Regular）",
    )
    add_p.add_argument(
        "--color",
        nargs=3,
        type=int,
        default=[0, 0, 0],
        metavar=("R", "G", "B"),
        help="文字颜色 RGB（默认: 0 0 0）",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "detect":
        detect_text(args.image, lang=args.lang)

    elif args.command == "replace":
        color = tuple(args.color)
        font_size = int(parse_font_size(args.size)) if str(args.size) != "0" else 0
        weight = args.weight or ""
        spacing = args.spacing or 0
        if args.old:
            # OCR 自动替换
            replace_text_ocr(
                args.image,
                args.old,
                args.new,
                output_path=args.output,
                lang=args.lang,
                method=args.method,
                font_path=args.font,
                font_size=font_size,
                color=color,
                exact_match=args.exact,
                deskew=args.deskew,
                rotate=args.rotate,
                blur=args.blur,
                bold=args.bold,
                weight=weight,
                letter_spacing=spacing,
            )
        elif args.x or args.y or args.w or args.height:
            # 手动区域替换
            if not (args.w and args.height):
                print("请指定完整的区域参数: -x -y -w --height")
                sys.exit(1)
            replace_text_manual(
                args.image,
                args.x,
                args.y,
                args.w,
                args.height,
                args.new,
                output_path=args.output,
                method=args.method,
                font_path=args.font,
                font_size=font_size,
                color=color,
                blur=args.blur,
                bold=args.bold,
                weight=weight,
                letter_spacing=spacing,
            )
        else:
            print("请指定 --old（OCR 自动查找）或 -x -y -w -h（手动指定区域）")
            sys.exit(1)

    elif args.command == "add":
        color = tuple(args.color)
        font_size = int(parse_font_size(args.size))
        weight = args.weight or ""
        spacing = args.spacing or 0
        add_text(
            args.image,
            args.text,
            args.x,
            args.y,
            output_path=args.output,
            font_path=args.font,
            font_size=font_size,
            color=color,
            bold=args.bold,
            weight=weight,
            letter_spacing=spacing,
        )


if __name__ == "__main__":
    main()
