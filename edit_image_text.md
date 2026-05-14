# edit_image_text.py — 图片文字修改工具

自动识别并替换/删除/添加图片中的文字，支持 OCR 自动定位和手动指定坐标。

## 依赖

```bash
pip install opencv-python numpy pillow pytesseract
```

macOS 还需安装 `tesseract`：

```bash
brew install tesseract tesseract-lang  # chi_sim 中文语言包
```

## 子命令

### `detect` — 检测图片中的文字位置

```bash
python3 edit_image_text.py detect input.jpg
```

输出 OCR 检测到的所有文字区域（坐标、尺寸、置信度），并生成 `input_detected.png` 预览图（绿色框标记）。

可选参数：`--lang`（OCR 语言，默认 `chi_sim+eng`）

### `replace` — 替换图片中的文字

**OCR 自动替换（推荐）：**

```bash
python3 edit_image_text.py replace input.jpg --old "旧文字" --new "新文字"
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--old` | 要替换的原文字（OCR 自动查找位置） |
| `--new` | 新文字 |
| `--exact` | 精确匹配（默认），`2023` 不会匹配到 `2023.01.19` |
| `--contains` | 子串匹配，`2023` 可匹配 `2023.01.19`，替换时保留周围文字 |
| `--deskew` | 自动检测并校正图片倾斜（最大 5°） |
| `--rotate` | 手动指定旋转角度（度，正=逆时针，如 `--rotate 2.5`） |
| `--color R G B` | 指定文字颜色（默认自动从原图采样） |
| `--font` | 指定字体文件路径（默认自动查找 Noto Sans SC → SimHei → STHeiti） |
| `--size` | 字体大小（默认 0=自动适配框大小），支持 pt 数值或中文字号如 `--size "五号"` |
| `--bold` | 加粗强度（1~3，数值越大笔画越粗） |
| `--weight` | 可变字体字重，可选：`Thin`、`ExtraLight`、`Light`、`Regular`、`Medium`、`SemiBold`、`Bold`、`ExtraBold`、`Black` |
| `--spacing` | 字间距（像素），如 `--spacing 1` 增大间距，负值缩小间距 |
| `--blur` | 模糊强度（建议 0.5~1.5），让文字边缘不那么锐利，更接近原图效果 |
| `--method` | 移除文字方式，`fill`（背景色填充，默认）或 `inpaint`（OpenCV 修补） |
| `-o` | 输出路径（默认 `input_replaced.jpg`） |

**手动指定区域：**

```bash
python3 edit_image_text.py replace input.jpg -x 100 -y 200 -w 80 --height 30 --new "新文字"
```

适用于 OCR 无法识别或定位不准的情况。

### `add` — 在图片中添加文字

```bash
python3 edit_image_text.py add input.jpg -x 50 -y 50 --text "Hello" --size 24
python3 edit_image_text.py add input.jpg -x 10 -y 10 --text "标题" --size "六号" --bold 1
```

## 工作原理

1. **OCR 定位**：使用 Tesseract OCR 查找文字在图片中的坐标和尺寸
2. **原样擦除**：采样文字区域周围的背景色填充，或使用 OpenCV inpaint 修补
3. **颜色采样**：通过灰度阈值提取文字核心像素，计算平均颜色
4. **位置对齐**：采样原文字在区域内的垂直中心线，使新文字与其对齐
5. **字体自适应**：二分搜索最佳字号（`--size 0` 时启用），确保文字在区域内完整显示
6. **旋转校正**：`--deskew` 使用 minAreaRect 自动检测倾斜，`--rotate` 手动指定角度，均自动扩大画布防止裁剪
7. **模糊柔化**：`--blur` 对绘制后的文字区域做高斯模糊，使新文字边缘柔和，与原图风格一致

## 中文字号参考

| 中文 | pt | 中文 | pt |
|------|----|------|----|
| 初号 | 42 | 小初 | 36 |
| 一号 | 26 | 小一 | 24 |
| 二号 | 22 | 小二 | 18 |
| 三号 | 16 | 小三 | 15 |
| 四号 | 14 | 小四 | 12 |
| 五号 | 10.5 | 小五 | 9 |
| 六号 | 7.5 | 小六 | 6.5 |
| 七号 | 5.5 | 八号 | 5 |

## macOS 常用中文字体

| 字体 | 路径 | 说明 |
|------|------|------|
| Noto Sans SC | `~/Library/Fonts/NotoSansSC-Variable.ttf` | 可变字体，支持 Thin~Black 字重，自动下载 |
| 华文细黑 | `/System/Library/Fonts/STHeiti Light.ttc` | 默认系统细黑体 |
| 华文中黑 | `/System/Library/Fonts/STHeiti Medium.ttc` | 默认系统中黑体 |
| 黑体 SimHei | `~/Library/Fonts/SimHei.ttf` | Windows 兼容黑体，自动下载 |
| 宋体 Songti | `/System/Library/Fonts/Supplemental/Songti.ttc` | 宋体（含 Light/Regular/Bold） |
| 苹方 PingFang | `/System/Library/Fonts/PingFang.ttc` | 苹果苹方字体 |

## 示例

```bash
# 检测所有文字
python3 edit_image_text.py detect input.jpg

# 精确替换 "2023" 为 "2024"
python3 edit_image_text.py replace input.jpg --old "2023" --new "2024"

# 子串匹配 + 自动旋转校正
python3 edit_image_text.py replace input.jpg --old "2023" --new "2024" --contains --deskew

# 子串匹配 + 指定颜色 + 模糊柔化
python3 edit_image_text.py replace input.jpg --old "2023" --new "2024" --contains --color 67 66 64 --blur 1.2

# 使用中文字号 + 加粗
python3 edit_image_text.py replace input.jpg --old "2023" --new "2024" --size "六号" --bold 1

# 指定字体（宋体）
python3 edit_image_text.py replace input.jpg --old "2023" --new "2024" --font "/System/Library/Fonts/Supplemental/Songti.ttc"

# 添加文字
python3 edit_image_text.py add input.jpg -x 100 -y 100 --text "备注" --size "六号" --bold 1

# 完整日期替换 + 旋转 + inpaint + 模糊 + ExtraLight + 字间距
python3 edit_image_text.py replace input.jpg --old "2023.01.19~2043.01.19" --new "2024.01.19-2044.01.19" --contains --rotate 2.5 --color 61 65 59 --blur 1.3 --method inpaint --weight ExtraLight --spacing 1

# 完整日期替换 + 旋转 + inpaint + 模糊 + ExtraLight + 字体大小 + 字间距
python3 edit_image_text.py replace input.jpg --old "2023.01.19~2043.01.19" --new "2024.01.19-2044.01.19" --contains --rotate 2.5 --blur 1.3 --method inpaint --size 52 --weight ExtraLight --spacing 1
```
