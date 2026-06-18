# Python 课程设计答辩 - Design Spec

> Human-readable design narrative — rationale, audience, style, color choices, content outline. Read once by downstream roles for context.
>
> Machine-readable execution contract: `spec_lock.md` (color / typography / icon / image short form). Executor re-reads `spec_lock.md` before every SVG page to resist context-compression drift. Keep both in sync; on divergence, `spec_lock.md` wins.

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | python_course_defense |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 13 |
| **Design Style** | 温暖轻盈学术演示 |
| **Target Audience** | Python 课程授课教师 + 同班同学 |
| **Use Case** | 课程答辩演示（约 6-8 分钟） |
| **Created Date** | 2026-06-15 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | 左右 60px，上下 50px |
| **Content Area** | 1160×620 |

---

## III. Visual Theme

### Theme Style

- **Mode**: narrative — 以"问题→数据→模型→应用"为主线逐步推进
- **Visual style**: custom — 温暖轻盈风格，暖白底色，每页可切换不同强调色
- **Theme**: Light theme（暖白基底）
- **Tone**: 学术但不沉闷，温暖但不随意，专业且清晰

### Color Scheme

**全局基底**：

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FDFBF7` | 暖白页面底色 |
| **Secondary bg** | `#FFF8EE` | 卡片、区块辅助背景 |
| **Primary** | `#E8842A` | 封面/标题装饰（琥珀暖橙） |
| **Accent** | `#E06B5E` | 问题提出页强调 |
| **Secondary accent** | `#3D8B5E` | 数据页强调 |
| **Body text** | `#2D2A26` | 深棕黑正文 |
| **Secondary text** | `#8C8578` | 温暖灰棕辅助文字 |
| **Tertiary text** | `#B0A898` | 页码、脚注 |
| **Border/divider** | `#E8E0D4` | 暖灰边线 |
| **Success** | `#3D8B5E` | 正向指标 |
| **Warning** | `#E06B5E` | 问题标记 |

**分区强调色**（每页选用不同色系形成节奏）：

| 色系 | HEX | 适用页面 |
|------|-----|----------|
| 琥珀暖橙 | `#E8842A` | P01 封面 |
| 暖红珊瑚 | `#E06B5E` | P02 研究背景、P03 问题提出 |
| 森林绿 | `#3D8B5E` | P04 数据获取 |
| 青蓝绿 | `#2AA3A0` | P05-P06 数据观察 |
| 暖紫 | `#8B6DB0` | P07 PCA 降维 |
| 赤陶棕 | `#C07040` | P08-P09 模型训练与对比 |
| 金黄 | `#D4940A` | P10 模型选择 |
| 橄榄绿 | `#6B8E3D` | P11 预测结果 |
| 玫瑰粉 | `#D46B8C` | P12 照明补偿 |
| 石板灰 | `#64748B` | P13 总结 |

### Gradient Scheme

```xml
<!-- 封面装饰渐变 -->
<linearGradient id="coverDecor" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#E8842A" stop-opacity="0.15"/>
  <stop offset="100%" stop-color="#D4940A" stop-opacity="0.05"/>
</linearGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: CJK 优先的现代无衬线，宽松大字版，适合远距离投影

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | "Microsoft YaHei" | Arial | sans-serif |
| **Body** | "Microsoft YaHei", "PingFang SC" | Arial | sans-serif |
| **Emphasis** | "Microsoft YaHei" | Arial | sans-serif |
| **Code** | — | Consolas | monospace |

**Per-role font stacks**:

- Title: `"Microsoft YaHei", Arial, sans-serif`
- Body: `"Microsoft YaHei", "PingFang SC", Arial, sans-serif`
- Emphasis: same as Body
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px

| Purpose | Ratio to body | Size @ body=20 | Weight |
| ------- | ------------- | -------------- | ------ |
| Cover title (hero) | 2.6x | 52px | Bold |
| Page title | 1.7x | 34px | Bold |
| Subtitle / section | 1.2x | 24px | SemiBold |
| **Body content** | **1x** | **20px** | Regular |
| Annotation / caption | 0.75x | 15px | Regular |
| Page number / footnote | 0.55x | 11px | Regular |

---

## V. Layout Principles

### Page Structure

- **Header area**: 60px，页面标题 + 页码
- **Content area**: 610px，主要信息区域
- **Footer area**: 50px，项目名称 + 页码

### Layout Pattern Library

| Pattern | Suitable Scenarios |
| ------- | ----------------- |
| **Single column centered** | 封面、总结 |
| **Asymmetric split (3:7)** | 左侧图表 + 右侧文字说明 |
| **Top-bottom split** | 流程图、时间线 |
| **Three/four column cards** | 特征列表、多模型对比 |
| **Negative-space-driven** | 呼吸页、关键结论 |

### Spacing Specification

**Universal**:

| Element | Recommended Range | Current Project |
| ------- | ---------------- | --------------- |
| Safe margin from canvas edge | 50-60px | 60px |
| Content block gap | 30-40px | 35px |
| Icon-text gap | 10-16px | 12px |

---

## VI. Icon Usage Specification

### Source

- **User icon pack**: `/Users/kmg/Documents/vscode/570+Icons-v1/`（574 个 SVG 图标）
- **Built-in library**: `templates/icons/chunk-filled`（备用）
- **Usage method**: 用户图标包的 SVG 文件嵌入为 `<image>` 元素，放大至 48×48px 或更大

### Recommended Icon List

| Purpose | Icon Source | Page |
| ------- | ----------- | ---- |
| 流程箭头 | user pack: arrow_right.svg | P04, P07 |
| 天气 | user pack: cloud.svg, cloudy.svg | P02, P05 |
| 数据/图表 | user pack: chart 相关 | P05, P06 |
| 齿轮/模型 | user pack: adjustment.svg | P08, P09 |
| 太阳/光谱 | user pack: 自然类图标 | P01, P11 |
| 灯泡/应用 | user pack: lightbulb 相关 | P12 |

---

## VII. Visualization Reference List

| Page | Visualization | Source | Usage |
| ---- | ------------- | ------ | ----- |
| P02 | 天气光谱对比图 | `weather_spectrum_compare.png` | 不同天气下平均相对光谱差异 |
| P05 | 特征重要性图 | `feature_importance.png` | 随机森林特征重要性排序 |
| P06 | 光谱样本图 | `base_spectrum.png` | 原始光谱曲线示例 |
| P07 | PCA 方差图 | `pca_variance.png` | 前 10 主成分累计解释方差 |
| P09 | 模型对比图 | `model_compare.png` | 五模型 RMSE/R²/训练时间对比 |
| P11 | 预测对比图 | `prediction_compare.png` | 实测光谱 vs 预测光谱 |
| P12 | 补偿结果图 | `compensation_result.png` | 目标/自然光/多通道/双色温对比 |

---

## VIII. Image Resource List

| Filename | Purpose | Type | Acquire Via | Status | text_policy | page_role |
| -------- | ------- | ---- | ----------- | ------ | ----------- | --------- |
| weather_spectrum_compare.png | 天气光谱对比 | Photography | user | Existing | none | local |
| feature_importance.png | 特征重要性 | Photography | user | Existing | none | local |
| base_spectrum.png | 光谱样本 | Photography | user | Existing | none | local |
| pca_variance.png | PCA 方差 | Photography | user | Existing | none | local |
| model_compare.png | 模型对比 | Photography | user | Existing | none | local |
| prediction_compare.png | 预测对比 | Photography | user | Existing | none | local |
| compensation_result.png | 补偿结果 | Photography | user | Existing | none | local |

---

## IX. Content Outline

### Part 1: 开场

#### Slide 01 - 封面

- **Layout**: 单列居中，大面积留白
- **Title**: 基于真实天光数据的自然光光谱估计与室内照明补偿设计
- **Subtitle**: 使用 Python 数据处理、PCA 降维与机器学习模型，估计自然光相对光谱
- **Info**: 第24组 · Python 应用开发基础 · 李安逸、黄奕滔
- **Core message**: 本项目用低成本环境特征估计自然光光谱，并用于室内照明补偿
- **Visualization**: 光谱曲线装饰背景元素

#### Slide 02 - 研究背景：为什么要从光谱角度看照明

- **Layout**: 左图右文（3:7 非对称分割）
- **Title**: 照明不只关乎"够不够亮"
- **Core message**: 亮度和色温不能完整描述光的组成，光谱才是更细致的光照描述方式
- **Content**:
  - 现代人大量时间在室内，人工照明影响学习、办公和生活体验
  - 常见照明调节关注亮度和色温，但这两个指标不能完整描述光的组成
  - 光谱表示不同波长上的能量分布，是更细致的光照描述方式
  - 重点句：基于光谱的室内照明调节，是未来高质量照明的一个潜在方向
- **Visualization**: weather_spectrum_compare.png（不同天气下平均光谱对比）

#### Slide 03 - 问题提出：用低成本特征估计自然光谱

- **Layout**: 上下分割（上：问题描述，下：流程图）
- **Title**: 光谱有价值，但直接测量成本较高
- **Core message**: 用更容易获得的环境特征，通过机器学习估计自然光相对光谱
- **Content**:
  - 专业光谱仪难以在普通场景普及
  - 输入：天气 / 时间 / 地点 / 太阳位置
  - → 机器学习模型 → 自然光相对光谱 → LED 补偿方案
  - 一句话目标：输入低成本环境特征，输出自然光相对光谱，并进一步用于照明补偿
- **Visualization**: 无（纯文字流程图）

### Part 2: 数据

#### Slide 04 - 数据获取与预处理

- **Layout**: 上下分割（上：三列数据来源卡片，下：关键数字）
- **Title**: 把真实观测数据整理成可训练的数据表
- **Core message**: 使用公开真实天光光谱数据，经过完整预处理流水线构建建模数据
- **Content**:
  - 数据来源：SKYSPECTRA 实测天光光谱（Zenodo）+ Open-Meteo 历史天气 API
  - 处理流程：文件合并 → 时间对齐 → 波长重采样 → 缺失值处理 → 特征编码
  - 关键数字：5664 样本、41 波长点（380-780nm）、2016-2018 年
- **Visualization**: 无（卡片式信息展示）

#### Slide 05 - 数据观察：模型输入特征

- **Layout**: 左侧表格 + 右侧特征重要性图（5:5 对称分割）
- **Title**: 模型输入：低成本环境特征
- **Core message**: 12 个输入特征覆盖时间、天气、太阳位置和地点信息
- **Content**:
  - 数值特征：hour, month, solar_altitude, solar_azimuth, cloud_cover, humidity, temperature, precipitation, outdoor_lux
  - 类别特征：weather, sky_condition, location_code（独热编码）
  - 特征重要性：outdoor_lux 占 54%，cloud_cover 占 9%，solar_azimuth 占 6%
- **Visualization**: feature_importance.png

#### Slide 06 - 数据观察：光谱输出与 PCA 铺垫

- **Layout**: 左图右文（4:6 非对称分割）
- **Title**: 模型输出：41 维相对光谱
- **Core message**: 光谱维度高、相邻波长相关性强，为 PCA 降维做铺垫
- **Content**:
  - 输出目标：380-780nm 范围，每 10nm 一个点，共 41 维
  - 相对光谱：每条光谱除以自身最大值，模型学习光谱形状而非绝对强度
  - 光谱曲线连续变化，相邻波长高度相关 → 适合 PCA 压缩
- **Visualization**: base_spectrum.png（光谱曲线示例）

### Part 3: 方法

#### Slide 07 - PCA 降维：从高维光谱到低维目标

- **Layout**: 上下分割（上：流程图，下：PCA 方差图）
- **Title**: 用 PCA 压缩光谱，同时保留主要变化信息
- **Core message**: 前 5 个主成分累计解释 98.35% 方差，大幅降低模型输出维度
- **Content**:
  - 流程：原始光谱 → PCA 降维 → 5 个主成分系数 → 模型预测 → PCA 逆变换 → 还原光谱
  - 前 5 个主成分：PC1=84.03%, PC2=6.64%, PC3=3.34%, PC4=2.42%, PC5=1.91%
  - 累计：98.35%
  - PCA 不是为了更"高级"，而是让模型更容易学习
- **Visualization**: pca_variance.png

#### Slide 08 - 模型训练：五种回归模型

- **Layout**: 三列卡片（五模型名称与简介）
- **Title**: 尝试多种课内模型，用结果选择方案
- **Core message**: 把光谱估计作为回归问题，尝试多种课程中涉及的模型
- **Content**:
  - Linear Regression：线性基准，观察问题是否接近线性
  - KNN：传统机器学习，基于邻域
  - Decision Tree：单棵决策树，中间对比
  - Random Forest：集成学习，关注稳定性和非线性
  - MLP：神经网络，更复杂的非线性对比
  - 所有模型使用相同数据划分、相同输入、相同 PCA 输出
- **Visualization**: 无（文字卡片）

#### Slide 09 - 模型对比：统一指标评价

- **Layout**: 上方表格 + 下方对比图（上下分割）
- **Title**: 五个模型在同一数据上评价
- **Core message**: Random Forest RMSE 最低，综合表现最稳
- **Content**:
  - 指标表：MAE / RMSE / R² / 训练时间 / 预测时间
  - 随机森林 RMSE=0.0086 最小，R²=0.959
  - MLP R²=0.962 略高，但 RMSE 更大、训练更慢
  - 线性回归明显落后 → 非线性关系
- **Visualization**: model_compare.png

### Part 4: 结果

#### Slide 10 - 最优模型选择：Random Forest

- **Layout**: 左文右图（5:5 对称分割）
- **Title**: 选择 Random Forest：效果与成本更平衡
- **Core message**: 综合 RMSE、稳定性和推理成本，Random Forest 是最佳选择
- **Content**:
  - 不选线性回归：误差大，说明非线性关系
  - 关注 MLP：代表神经网络路线，但训练成本高
  - 选择 RF：RMSE 最低、推理快、稳定性好
  - 关键参数：树的数量、深度、随机种子等（按实际结果填写）
- **Visualization**: 无（文字为主）

#### Slide 11 - 预测结果：光谱还原

- **Layout**: 左图右文（5:5 对称分割）
- **Title**: 环境特征可以还原相对光谱形状
- **Core message**: 预测光谱与实测光谱整体接近，模型能学习主要变化趋势
- **Content**:
  - 预测路径：环境特征 → 5 个 PCA 系数 → PCA 逆变换 → 41 维光谱
  - 示例样本中预测曲线与实测曲线基本重合
  - 结果边界：预测相对光谱形状，不能替代现场光谱仪
- **Visualization**: prediction_compare.png

#### Slide 12 - 照明补偿：七通道 LED

- **Layout**: 上方文字说明 + 下方补偿对比图（上下分割）
- **Title**: 七通道 LED 比双色温更细致
- **Core message**: 多通道 LED 有更多自由度，能对不同波段分别补偿
- **Content**:
  - 补偿思路：目标光谱 - 当前自然光 = 需要补偿的光谱
  - 算法：非负最小二乘求 7 通道比例
  - 七通道：深蓝、青、绿、琥珀、红、暖白、冷白
  - 对比：多通道可按波段局部补偿 vs 双色温主要调冷暖
  - 课程定位：算法演示，暂未接入真实硬件
- **Visualization**: compensation_result.png

### Part 5: 收尾

#### Slide 13 - 项目总结

- **Layout**: 左右两栏（左：特色与不足，右：分工与大模型使用）
- **Title**: 项目总结
- **Core message**: 完整但谨慎的课程项目：数据真实、流程可复现、结果能进入应用演示
- **Content**:
  - 特色：真实公开数据、PCA 降维、五模型统一对比、LED 补偿闭环
  - 不足：样本集中少数站点、天气非现场同步、不能替代光谱仪、未接入硬件
  - 分工：李安逸（报告、方法分析、PPT）、黄奕滔（代码、数据处理、模型、页面）
  - 大模型使用：数据收集、报告结构、代码注释、可视化辅助；核心方案和结果由团队完成

---

## X. Speaker Notes Requirements

One speaker note file per page, saved to `notes/`:

- **Filename**: match SVG name (e.g., `01_cover.md`)
- **Content**: script key points, timing cues, transition phrases

---

## XI. Technical Constraints Reminder

### SVG Generation Must Follow:

1. viewBox: `0 0 1280 720`
2. Background uses `<rect>` elements
3. Text wrapping uses `<tspan>` (`<foreignObject>` FORBIDDEN)
4. Transparency uses `fill-opacity` / `stroke-opacity`; `rgba()` FORBIDDEN
5. FORBIDDEN: `mask`, `<style>`, `class`, `foreignObject`
6. FORBIDDEN: `textPath`, `animate*`, `script`
7. Text characters: write as raw Unicode; HTML named entities FORBIDDEN
8. `marker-start` / `marker-end` conditionally allowed
9. `clipPath` conditionally allowed **only on `<image>` elements**

### PPT Compatibility Rules:

- `<g opacity="...">` FORBIDDEN (group opacity); set on each child element individually
- Image transparency uses overlay mask layer
- Inline styles only; external CSS and `@font-face` FORBIDDEN
