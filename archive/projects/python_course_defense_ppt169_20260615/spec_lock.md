# Execution Lock

> Machine-readable execution contract. Executor MUST `read_file` this before every SVG page. Values not listed here must NOT appear in SVGs. For design narrative (rationale, audience, style), see `design_spec.md`.

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## mode
- mode: narrative

## visual_style
- visual_style: custom
- visual_style_behavior: 温暖轻盈风格，暖白底色（#FDFBF7），圆角卡片柔和，留白充足松弛，不同页面使用不同强调色形成节奏感。每页内不同板块使用不同颜色区分。装饰克制，以数据图表为核心视觉元素。关键数据用荧光笔叠色效果高亮。

## colors
- bg: #FDFBF7
- bg_secondary: #FFF8EE
- primary: #E8842A
- accent: #E06B5E
- secondary_accent: #3D8B5E
- tertiary_accent: #2AA3A0
- quaternary_accent: #8B6DB0
- brown_accent: #C07040
- gold_accent: #D4940A
- olive_accent: #6B8E3D
- rose_accent: #D46B8C
- gray_accent: #64748B
- sky_accent: #4A90D9
- lime_accent: #7CB342
- text: #2D2A26
- text_secondary: #8C8578
- text_tertiary: #B0A898
- border: #E8E0D4
- success: #3D8B5E
- warning: #E06B5E
- highlight_yellow: #FFF3B0
- highlight_red: #FFE0E0

## typography
- font_family: "Microsoft YaHei", Arial, sans-serif
- body_family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif
- code_family: Consolas, "Courier New", monospace
- body: 22
- title: 38
- subtitle: 26
- cover_title: 56
- annotation: 16
- page_number: 12

## icons
- library: chunk-filled
- inventory: arrow-right, cloud, chart-bar, sun, lightbulb, cog, target, checkmark

## images
- weather_spectrum_compare: images/weather_spectrum_compare.png | no-crop
- feature_importance: images/feature_importance.png | no-crop
- base_spectrum: images/base_spectrum.png | no-crop
- pca_variance: images/pca_variance.png | no-crop
- model_compare: images/model_compare.png | no-crop
- prediction_compare: images/prediction_compare.png | no-crop
- compensation_result: images/compensation_result.png | no-crop

## page_rhythm
- P01: anchor
- P02: breathing
- P03: dense
- P04: dense
- P05: dense
- P06: breathing
- P07: dense
- P08: dense
- P09: breathing
- P10: dense
- P11: breathing
- P12: dense
- P13: dense
- P14: anchor
- P15: anchor

## page_layouts
- P01: 01_cover

## page_charts
