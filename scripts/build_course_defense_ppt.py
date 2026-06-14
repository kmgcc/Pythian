from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Cm, Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "python课设_代码" / "outputs" / "figures"
RES_DIR = ROOT / "python课设_代码" / "outputs" / "results"
OUT = ROOT / "第24组_课程答辩.pptx"


PAPER = "FAFAF8"
INK = "0A0A0A"
GREY_1 = "F0F0EE"
GREY_2 = "D4D4D2"
GREY_3 = "737373"
ACCENT = "002FA7"
WHITE = "FFFFFF"
SUCCESS = "2F855A"
WARN = "B7791F"

FONT = "Microsoft YaHei"
FONT_LATIN = "Aptos"
FONT_MONO = "Consolas"


def rgb(hex_color: str) -> RGBColor:
    h = hex_color.replace("#", "")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def set_run_font(run, size: float, color: str = INK, bold: bool = False, font: str = FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: float = 18,
    color: str = INK,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font: str = FONT,
    line_spacing: float | None = None,
):
    box = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = box.text_frame
    tf.clear()
    tf.margin_left = Cm(0)
    tf.margin_right = Cm(0)
    tf.margin_top = Cm(0)
    tf.margin_bottom = Cm(0)
    tf.vertical_anchor = valign
    tf.word_wrap = True
    first = True
    for part in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        run = p.add_run()
        run.text = part
        set_run_font(run, size, color, bold, font)
    return box


def add_title(slide, title: str, index: str, kicker: str = "COURSE DEFENSE"):
    add_text(slide, f"{kicker} · {index}", 0.92, 0.24, 5.6, 0.30, 7.7, GREY_3, True, font=FONT_MONO)
    add_text(slide, title, 0.92, 0.73, 22.0, 0.76, 17.6, INK, True)
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Cm(0.92), Cm(1.67), Cm(11.2), Cm(0.035))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb(ACCENT)
    line.line.fill.background()


def add_footer(slide, idx: int, text: str = "基于真实天光数据的自然光光谱估计与室内照明补偿设计"):
    add_text(slide, text, 0.92, 18.42, 10.5, 0.35, 7.2, GREY_3, False, font=FONT_MONO)
    add_text(slide, f"{idx:02d}/10", 30.15, 18.42, 2.0, 0.35, 7.2, GREY_3, False, align=PP_ALIGN.RIGHT, font=FONT_MONO)


def set_bg(slide, color: str = PAPER):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(color)


def add_rect(slide, x, y, w, h, fill=GREY_1, line=None, radius=False):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Cm(x), Cm(y), Cm(w), Cm(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.6)
    else:
        shp.line.fill.background()
    return shp


def add_chip(slide, text, x, y, w, fill=GREY_1, color=INK):
    shp = add_rect(slide, x, y, w, 0.62, fill=fill, line=None)
    add_text(slide, text, x + 0.18, y + 0.16, w - 0.36, 0.32, 8.5, color, True, align=PP_ALIGN.CENTER)
    return shp


def add_bullets(slide, items, x, y, w, row_h=0.82, size=11.5, color=INK, accent=ACCENT):
    for i, item in enumerate(items):
        yy = y + i * row_h
        sq = add_rect(slide, x, yy + 0.16, 0.13, 0.13, fill=accent)
        sq.line.fill.background()
        add_text(slide, item, x + 0.34, yy, w - 0.34, row_h, size, color, False, line_spacing=1.07)


def add_caption(slide, text, x, y, w):
    add_text(slide, text, x, y, w, 0.38, 7.2, GREY_3, False, font=FONT_MONO)


def add_picture_fit(slide, path: Path, x, y, w, h, border: bool = False):
    with Image.open(path) as im:
        iw, ih = im.size
    box_ratio = w / h
    img_ratio = iw / ih
    if img_ratio >= box_ratio:
        width = w
        height = w / img_ratio
        xx = x
        yy = y + (h - height) / 2
    else:
        height = h
        width = h * img_ratio
        xx = x + (w - width) / 2
        yy = y
    pic = slide.shapes.add_picture(str(path), Cm(xx), Cm(yy), Cm(width), Cm(height))
    if border:
        frame = add_rect(slide, x, y, w, h, fill=PAPER, line=GREY_2)
        slide.shapes._spTree.remove(frame._element)
        slide.shapes._spTree.insert(2, frame._element)
    return pic


def add_metric(slide, label, value, x, y, w=3.0, color=ACCENT):
    add_text(slide, value, x, y, w, 0.64, 20, color, True, font=FONT_LATIN)
    add_text(slide, label, x, y + 0.76, w, 0.36, 7.6, GREY_3, True, font=FONT_MONO)


def add_card(slide, title, body, x, y, w, h, fill=GREY_1, title_color=ACCENT):
    add_rect(slide, x, y, w, h, fill=fill, line=None)
    add_text(slide, title, x + 0.35, y + 0.30, w - 0.70, 0.38, 10.2, title_color, True)
    add_text(slide, body, x + 0.35, y + 0.92, w - 0.70, h - 1.10, 9.6, INK, False, line_spacing=1.05)


def add_arrow(slide, x1, y1, x2, y2, color=ACCENT, width=1.2):
    con = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    con.line.color.rgb = rgb(color)
    con.line.width = Pt(width)
    con.line.end_arrowhead = True
    return con


def slide_cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_rect(slide, 0, 0, 3.15, 19.05, fill=ACCENT)
    add_text(slide, "PYTHON\nCOURSE\nDESIGN", 0.62, 0.70, 2.0, 2.2, 9.0, WHITE, True, font=FONT_MONO)
    add_text(slide, "第24组", 0.62, 17.62, 1.9, 0.4, 9.0, WHITE, True, font=FONT_MONO)
    add_text(slide, "基于真实天光数据的\n自然光光谱估计与\n室内照明补偿设计", 4.15, 2.08, 16.5, 3.95, 28, INK, True, line_spacing=0.90)
    add_text(slide, "课程名称：Python 应用开发基础\n小组成员：李安逸、黄奕滔", 4.22, 6.70, 10.5, 1.25, 12, INK, False, line_spacing=1.2)
    add_rect(slide, 4.20, 8.70, 7.8, 2.15, fill=GREY_1)
    add_metric(slide, "VISIBLE WAVELENGTH POINTS", "41", 4.62, 9.03, 2.3)
    add_metric(slide, "REAL SAMPLES", "5664", 7.06, 9.03, 2.6)
    add_metric(slide, "PCA CUM. VAR.", "98.3%", 10.02, 9.03, 2.0)
    add_picture_fit(slide, FIG_DIR / "base_spectrum.png", 16.0, 8.58, 14.6, 7.4)
    add_caption(slide, "图：实测自然光基准光谱曲线", 16.0, 16.18, 8.0)
    add_text(slide, "答辩时间约 6 分钟", 25.1, 0.72, 5.0, 0.4, 8.5, GREY_3, True, align=PP_ALIGN.RIGHT, font=FONT_MONO)
    return slide


def slide_background(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "研究背景：照明质量不能只看亮度和色温", "02")
    add_bullets(
        slide,
        [
            "人长期处于室内，学习、办公和居家场景都依赖人工照明。",
            "照度看亮度，色温看冷暖；不能描述完整波长分布。",
            "同亮度、同色温下，光谱形状仍可能明显不同。",
        ],
        1.05,
        2.60,
        10.8,
        row_h=1.05,
        size=11.4,
    )
    add_rect(slide, 1.05, 6.32, 10.65, 2.15, fill=ACCENT)
    add_text(slide, "因此，本项目从“光谱”角度描述自然光，\n并把估计结果用于室内补偿。", 1.45, 6.68, 9.65, 1.05, 13.0, WHITE, True, line_spacing=1.0)
    add_picture_fit(slide, FIG_DIR / "weather_spectrum_compare.png", 13.05, 2.14, 18.0, 13.15)
    add_caption(slide, "关键图 1：不同天气下平均相对光谱存在差异", 13.05, 15.55, 13.5)
    add_footer(slide, 2)


def slide_goal(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "项目目标：由环境特征估计自然光相对光谱", "03")
    add_card(slide, "输入", "天气类别、云量、湿度、温度、降水\n时间、月份、地点\n太阳高度角/方位角\n室外水平照度", 1.0, 3.1, 8.2, 5.8)
    add_card(slide, "输出", "380–780 nm\n每 10 nm 一个点\n共 41 维相对光谱曲线\n模型先预测 PCA 系数，再还原光谱", 12.25, 3.1, 8.2, 5.8)
    add_card(slide, "应用", "根据目标自然光谱与当前自然光贡献\n计算七通道 LED 推荐比例\n形成“预测 → 补偿 → 对比”的闭环", 23.5, 3.1, 8.2, 5.8)
    add_arrow(slide, 9.52, 5.95, 11.55, 5.95)
    add_arrow(slide, 20.78, 5.95, 22.78, 5.95)
    add_rect(slide, 3.7, 10.8, 24.6, 3.15, fill=PAPER, line=GREY_2)
    labels = ["环境特征", "机器学习回归", "PCA逆变换", "光谱曲线", "LED补偿"]
    xs = [4.45, 9.35, 14.45, 19.25, 24.1]
    for i, (label, x) in enumerate(zip(labels, xs)):
        fill = ACCENT if i in [1, 4] else GREY_1
        color = WHITE if fill == ACCENT else INK
        add_chip(slide, label, x, 11.96, 3.1, fill=fill, color=color)
        if i < len(labels) - 1:
            add_arrow(slide, x + 3.28, 12.27, xs[i + 1] - 0.15, 12.27, color=GREY_3, width=0.75)
    add_footer(slide, 3)


def slide_data(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "数据来源：公开真实数据构建完整样本表", "04")
    add_card(slide, "SKYSPECTRA 实测天光光谱", "Zenodo record 8147546\n水平天光光谱 + 站点、时间、天空状况、太阳位置元数据\n重采样到 380–780 nm / 10 nm", 1.0, 2.45, 9.45, 5.35, fill=GREY_1)
    add_card(slide, "Open-Meteo 历史天气 API", "按观测站经纬度和日期范围查询\n云量、湿度、温度、降水\n按最近整点与光谱测量对齐", 11.3, 2.45, 9.45, 5.35, fill=GREY_1)
    add_card(slide, "公开 LED 光谱数据", "用于构造七通道 LED 补偿模型\n深蓝、青、绿、琥珀、红、暖白、冷白\n只用于应用补偿，不参与模型训练", 21.6, 2.45, 9.45, 5.35, fill=GREY_1)
    add_rect(slide, 1.0, 9.45, 30.05, 4.3, fill=PAPER, line=GREY_2)
    add_metric(slide, "样本量", "5664", 2.0, 10.25, 4.0)
    add_metric(slide, "光谱维度", "41", 7.05, 10.25, 3.8)
    add_metric(slide, "时间范围", "2016–2018", 11.25, 10.25, 5.8)
    add_metric(slide, "观测站", "2", 18.25, 10.25, 2.8)
    add_metric(slide, "主要站点", "法国沃昂夫兰", 22.0, 10.25, 5.9)
    add_text(slide, "说明：天气数据为按地点和时间对齐的历史天气特征，不等同于现场同步气象测量。", 2.05, 13.12, 27.2, 0.45, 9.3, GREY_3, False)
    add_footer(slide, 4)


def slide_flow(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "总体技术流程：从真实数据到照明补偿闭环", "05")
    steps = [
        ("01", "数据下载"),
        ("02", "清洗重采样"),
        ("03", "天气对齐"),
        ("04", "特征工程"),
        ("05", "PCA"),
        ("06", "五模型训练"),
        ("07", "光谱还原"),
        ("08", "LED 补偿"),
    ]
    x0, y0, w, h, gap = 1.05, 3.20, 6.65, 2.25, 1.0
    for i, (num, label) in enumerate(steps):
        row = 0 if i < 4 else 1
        col = i if i < 4 else 7 - i
        x = x0 + col * (w + gap)
        y = y0 + row * 4.45
        fill = ACCENT if i in [4, 7] else GREY_1
        txt_color = WHITE if fill == ACCENT else INK
        add_rect(slide, x, y, w, h, fill=fill)
        add_text(slide, num, x + 0.35, y + 0.30, 1.0, 0.4, 9.4, txt_color, True, font=FONT_MONO)
        add_text(slide, label, x + 0.35, y + 1.03, w - 0.7, 0.5, 13.0, txt_color, True)
        if i < 3:
            add_arrow(slide, x + w + 0.15, y + 1.12, x + w + gap - 0.20, y + 1.12, color=GREY_3, width=0.75)
        if i == 3:
            add_arrow(slide, x + w / 2, y + h + 0.2, x + w / 2, y + 4.05, color=GREY_3, width=0.75)
        if 4 <= i < 7:
            x_next = x0 + (col - 1) * (w + gap)
            add_arrow(slide, x - 0.18, y + 1.12, x_next + w + 0.20, y + 1.12, color=GREY_3, width=0.75)
    add_rect(slide, 1.05, 13.40, 30.0, 1.75, fill=PAPER, line=GREY_2)
    add_text(slide, "核心逻辑：真实数据支撑建模，PCA 降低输出维度，模型预测的光谱再进入七通道 LED 补偿。", 1.55, 13.90, 28.7, 0.42, 11.6, INK, True)
    add_footer(slide, 5)


def slide_pca(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "PCA 与特征工程：把 41 维光谱压缩成可学习目标", "06")
    add_picture_fit(slide, FIG_DIR / "pca_variance.png", 1.0, 2.25, 14.0, 9.3)
    add_picture_fit(slide, FIG_DIR / "feature_importance.png", 16.7, 2.25, 14.2, 9.3)
    add_caption(slide, "PCA 解释方差：前 5 个主成分累计约 98.3%", 1.0, 11.78, 12.8)
    add_caption(slide, "随机森林特征重要性：室外照度、云量、太阳位置影响明显", 16.7, 11.78, 14.0)
    add_rect(slide, 1.0, 13.05, 30.0, 2.05, fill=GREY_1)
    add_bullets(
        slide,
        [
            "输入特征：天气、时间、地点、太阳位置、室外照度；类别特征独热编码，数值特征标准化。",
            "输出目标：模型不直接预测 41 维光谱，而是预测 PCA 主成分系数，再逆变换还原。",
        ],
        1.45,
        13.50,
        28.3,
        row_h=0.70,
        size=10.2,
    )
    add_footer(slide, 6)


def slide_models(prs, metrics: pd.DataFrame):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "模型训练与对比：五个回归模型在同一数据上评价", "07")
    add_picture_fit(slide, FIG_DIR / "model_compare.png", 1.0, 2.15, 19.5, 7.3)
    add_caption(slide, "指标柱状图：RMSE、R²、训练时间", 1.0, 9.55, 10.0)
    table_data = [["模型", "MAE", "RMSE", "R²", "训练/s"]]
    order = ["Random Forest", "KNN", "MLP", "Decision Tree", "Linear Regression"]
    m = metrics.set_index("model")
    for name in order:
        row = m.loc[name]
        table_data.append([name, f"{row['MAE']:.4f}", f"{row['RMSE']:.4f}", f"{row['R2']:.3f}", f"{row['train_time_s']:.2f}"])
    rows, cols = len(table_data), len(table_data[0])
    table_shape = slide.shapes.add_table(rows, cols, Cm(21.25), Cm(2.35), Cm(9.85), Cm(5.7))
    table = table_shape.table
    widths = [3.45, 1.45, 1.55, 1.35, 1.55]
    for i, width in enumerate(widths):
        table.columns[i].width = Cm(width)
    for r, row in enumerate(table_data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = val
            cell.margin_left = Cm(0.08)
            cell.margin_right = Cm(0.08)
            cell.margin_top = Cm(0.06)
            cell.margin_bottom = Cm(0.06)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(ACCENT if r == 0 else (GREY_1 if r % 2 else PAPER))
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    set_run_font(run, 7.6 if r else 7.8, WHITE if r == 0 else INK, bold=(r == 0 or (r == 1 and c == 0)), font=FONT_LATIN if c == 0 else FONT_MONO)
    add_rect(slide, 21.25, 8.65, 9.85, 2.35, fill=ACCENT)
    add_text(slide, "最终按 RMSE 最小选择随机森林", 21.75, 9.05, 8.8, 0.48, 12.6, WHITE, True)
    add_text(slide, "KNN 的 MAE 略低，但 RMSE 与预测耗时不占优；MLP 的 R² 略高，但误差更大，稳定性也更敏感。", 21.75, 9.80, 8.75, 0.76, 8.6, WHITE, False, line_spacing=1.0)
    add_footer(slide, 7)


def slide_prediction(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "光谱预测结果：环境特征可以还原相对光谱形状", "08")
    add_picture_fit(slide, FIG_DIR / "prediction_compare.png", 1.0, 2.15, 21.2, 12.2)
    add_rect(slide, 23.1, 2.28, 8.0, 5.35, fill=GREY_1)
    add_text(slide, "预测路径", 23.55, 2.72, 7.1, 0.35, 10.2, ACCENT, True)
    add_bullets(
        slide,
        [
            "输入环境特征",
            "预测 5 个 PCA 系数",
            "PCA 逆变换还原 41 维光谱",
            "与实测光谱曲线比较",
        ],
        23.55,
        3.42,
        6.95,
        row_h=0.70,
        size=9.4,
    )
    add_rect(slide, 23.1, 8.20, 8.0, 3.4, fill=ACCENT)
    add_text(slide, "结果观察", 23.55, 8.62, 7.1, 0.35, 10.2, WHITE, True)
    add_text(slide, "示例样本中，预测曲线与实测曲线基本重合。\n模型能够学习自然光相对光谱的主要变化趋势。", 23.55, 9.30, 7.0, 1.32, 8.6, WHITE, False, line_spacing=1.0)
    add_caption(slide, "关键图 2：实测光谱 vs 模型预测光谱", 1.0, 14.58, 13.8)
    add_footer(slide, 8)


def slide_compensation(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "照明补偿结果：七通道 LED 比双色温更细致", "09")
    add_picture_fit(slide, FIG_DIR / "application_spectra_compare.png", 1.0, 2.10, 20.7, 12.15)
    add_caption(slide, "关键图 3：目标光谱、预测自然光、补偿后光谱与传统双色温方案对比", 1.0, 14.45, 18.0)
    add_rect(slide, 22.4, 2.22, 8.75, 4.7, fill=GREY_1)
    add_text(slide, "补偿思路", 22.85, 2.62, 7.8, 0.35, 10.2, ACCENT, True)
    add_text(slide, "目标光谱 - 当前自然光贡献\n= 需要人工补偿的光谱", 22.85, 3.33, 7.7, 1.0, 11.0, INK, True, line_spacing=1.0)
    add_text(slide, "用非负最小二乘计算 LED 通道比例，使自然光 + 人工光尽量接近目标光谱。", 22.85, 4.75, 7.7, 0.85, 9.0, INK, False, line_spacing=1.05)
    add_rect(slide, 22.4, 7.55, 8.75, 4.95, fill=ACCENT)
    add_text(slide, "七通道 LED", 22.85, 7.98, 7.8, 0.35, 10.2, WHITE, True)
    add_text(slide, "深蓝/蓝光、青光、绿光、琥珀光、红光、暖白、冷白\n\n通道更多，可按波段局部补偿；\n双色温主要调冷暖，光谱自由度较低。", 22.85, 8.60, 7.6, 2.55, 8.4, WHITE, False, line_spacing=1.0)
    add_footer(slide, 9)


def slide_closing(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide)
    add_title(slide, "项目特色、不足与分工", "10")
    add_card(
        slide,
        "项目特色",
        "真实公开数据\nPCA 降维表示光谱\n五模型统一对比\n预测结果进入 LED 补偿闭环",
        1.0,
        2.35,
        9.7,
        5.2,
        fill=GREY_1,
    )
    add_card(
        slide,
        "主要不足",
        "样本集中在少数站点\n天气数据非现场同步\n模型不能替代现场光谱仪\n未接入真实硬件",
        11.45,
        2.35,
        9.7,
        5.2,
        fill=GREY_1,
        title_color=WARN,
    )
    add_card(
        slide,
        "后续改进",
        "补充更多地区实测数据\n接入实时天气与廉价传感器\n结合智能灯具进行硬件验证\n进一步评估舒适度和显色效果",
        21.9,
        2.35,
        9.2,
        5.2,
        fill=GREY_1,
        title_color=SUCCESS,
    )
    add_rect(slide, 1.0, 9.55, 30.1, 4.7, fill=PAPER, line=GREY_2)
    add_text(slide, "小组分工", 1.55, 10.05, 5.0, 0.45, 12, ACCENT, True)
    add_text(slide, "黄奕滔", 2.0, 11.05, 3.0, 0.42, 12, INK, True)
    add_text(slide, "数据处理、模型训练、天气 API 获取、Streamlit 页面、代码注释与运行结果整理", 5.2, 11.03, 24.5, 0.5, 10.2, INK, False)
    add_text(slide, "李安逸", 2.0, 12.22, 3.0, 0.42, 12, INK, True)
    add_text(slide, "课程报告、背景与方法说明、PCA 与模型分析、PPT 逻辑整理与答辩表达", 5.2, 12.20, 24.5, 0.5, 10.2, INK, False)
    add_text(slide, "共同完成：课程答辩、PPT、最终材料整合。", 5.2, 13.36, 24.5, 0.5, 10.2, GREY_3, False)
    add_rect(slide, 1.0, 15.45, 30.1, 1.45, fill=ACCENT)
    add_text(slide, "总结：本项目把真实天光数据、机器学习光谱估计和室内照明补偿连成了一条可复现的课程设计流程。", 1.55, 15.89, 29.0, 0.52, 12.2, WHITE, True)
    add_footer(slide, 10)


def build():
    metrics = pd.read_csv(RES_DIR / "model_metrics.csv")
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    # Drop default empty slide if present in some templates.
    while len(prs.slides) > 0:
        r_id = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(r_id)
        del prs.slides._sldIdLst[0]

    slide_cover(prs)
    slide_background(prs)
    slide_goal(prs)
    slide_data(prs)
    slide_flow(prs)
    slide_pca(prs)
    slide_models(prs, metrics)
    slide_prediction(prs)
    slide_compensation(prs)
    slide_closing(prs)

    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
