# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python course project for **natural light spectrum estimation and indoor lighting compensation**. It uses real measured daylight spectra from the SKYSPECTRA dataset (Zenodo) combined with weather, time, location, and solar position features to train ML models that estimate natural light relative spectra (380nm–780nm), then applies 7-channel LED compensation.

The project lives in the `python课设_代码/` subdirectory (not the repo root). All commands below assume you are in `python课设_代码/`.

## Common Commands

```bash
# Install dependencies
cd python课设_代码
pip install -r requirements.txt

# One-click pipeline (download data → preprocess → PCA → train 5 models → evaluate → export figures)
python -m src.pipeline

# Jupyter notebook (7 chapters, step-by-step demo)
jupyter notebook main.ipynb

# Streamlit dashboard (5 tabs: data overview / PCA / model comparison / spectrum prediction / lighting compensation)
streamlit run app.py

# Or use the convenience launcher (starts Streamlit + opens browser)
python start_web.py
```

## Architecture

The codebase follows a modular pipeline pattern. Each step is a standalone module in `src/` that can be run independently (via notebook) or orchestrated by `src/pipeline.py`.

**Data flow:**
```
Zenodo (SKYSPECTRA) + Open-Meteo weather API
  → real_data_pipeline.py (download, clean, resample to 380–780nm/10nm)
  → data/real_spectrum_weather_dataset.csv (5664 rows × 58 cols)
  → preprocessing.py (one-hot encode, standardize, normalize spectra)
  → spectrum_pca.py (5 PCA components, >99% variance)
  → model_training.py (5 regressors: LR, KNN, DT, RF, MLP)
  → evaluation.py (MAE/RMSE/R² metrics)
  → lighting_compensation.py (7-channel LED optimization vs dual-white LED)
  → application_demo.py (weather presets → prediction → compensation comparison)
```

**Key modules:**
- `src/real_data_pipeline.py` — Downloads raw data from Zenodo, calls `weather_api.py` for Open-Meteo historical weather, cleans and merges everything into the final CSV
- `src/preprocessing.py` — Defines `FEATURE_COLUMNS` (the input features for models) and `build_preprocessor()`
- `src/spectrum_pca.py` — `SpectrumPCA` class wraps sklearn PCA with fit/transform/inverse_transform for spectral data
- `src/model_training.py` — `SpectrumPredictor` class bundles the best model + PCA + feature columns into a single prediction interface; `train_all_models()` returns `dict[str, TrainedModel]`
- `src/lighting_compensation.py` — Core compensation algorithm: given a predicted spectrum, optimizes 7-channel LED weights to match a target spectrum
- `src/color_conversion.py` — CIE 2015 10° observer (not 1931) for spectrum → approximate screen color
- `src/application_demo.py` — Weather presets (晴天中午/多云下午/阴雨天气/傍晚) with parameters from real dataset medians
- `src/pipeline.py` — Orchestrator; `PipelineResult` dataclass holds all artifacts; `load_or_train()` caches via joblib

**Entry points:**
- `python -m src.pipeline` — full pipeline (must run from `python课设_代码/`)
- `main.ipynb` — interactive notebook (7 chapters, same modules called step-by-step)
- `app.py` — Streamlit dashboard (uses `pipeline.load_or_train()`)

## Data & Model Artifacts

- Raw Zenodo data auto-downloads to `data/external/` (gitignored, ~78MB spectrum file)
- Weather API responses cached in `data/external/weather_cache/`
- Final dataset: `data/real_spectrum_weather_dataset.csv`
- Trained model: `models/spectrum_model.joblib` (PipelineResult via joblib)
- LED spectra: `data/dual_white_led_spectrum.csv`
- CIE 2015 color matching functions: `data/color_matching/cie2015_xyz_10deg.csv`
- Outputs: `outputs/figures/*.png` and `outputs/results/*.csv`

## Conventions

- Python 3.10+ required (uses `from __future__ import annotations` throughout)
- All modules support both `python -m src.xxx` direct execution and package import
- Chinese comments and docstrings are used throughout — this is a Chinese-language course project
- PCA uses 5 components by default (`N_COMPONENTS = 5` in pipeline.py)
- Spectra are normalized to relative intensity (per-sample max normalization)
- The project explicitly uses CIE 2015 10° observer (not CIE 1931) for color conversion

## PPT / Office 文档编辑

编辑 PPT 时**必须优先使用 `officecli` skill**，不要用 python-pptx 等 Python 库写脚本。设计排版 PPT 时使用 **`ppt-master` skill**。

### officecli 使用步骤

**1. 安装（如未安装）：**
```bash
curl -fsSL https://d.officecli.ai/install.sh | bash
```

**2. 基本操作流程（L1 → L2 → L3 逐层深入）：**

```bash
# 创建 / 读取
officecli create slides.pptx                        # 创建空白 PPT
officecli view slides.pptx outline                  # 查看结构
officecli get slides.pptx '/slide[1]' --depth 1     # 查看某页幻灯片的元素

# 添加元素
officecli add slides.pptx '/slide[1]' --type shape --prop text="标题" --prop x=2cm --prop y=1cm --prop font=Arial --prop size=32
officecli add slides.pptx '/slide[1]' --type slide --prop title="新页面"  # 添加新幻灯片

# 修改属性（L2 DOM 操作）
officecli set slides.pptx '/slide[1]/shape[1]' --prop fill=FF0000        # 改颜色
officecli set slides.pptx '/slide[1]/shape[1]' --prop font.size=24pt     # 改字号

# 批量操作（一次保存）
echo '[{"command":"set","path":"/slide[1]/shape[1]","props":{"fill":"blue"}}]' | officecli batch slides.pptx --json

# 预览与交互选择
officecli watch slides.pptx          # 启动实时预览（浏览器点击选中元素）
officecli get slides.pptx selected   # 读取浏览器中选中的元素路径

# 验证
officecli validate slides.pptx
```

**3. 重要约定：**
- 路径从 1 开始：`'/slide[1]/shape[1]'`；`--index` 从 0 开始
- 所有属性通过 `--prop key=value` 传递，不要用 `--name`
- 不确定属性名时运行 `officecli help pptx shape` 查看
- zsh/bash 中路径必须加引号：`'/slide[1]'`，否则 `[` 会被 glob 展开
- `shape[1]` 通常是标题占位符，内容从 `shape[2]` 开始
- 加载专业 skill：`officecli load_skill pptx`（通用 PPT）或 `officecli load_skill morph-ppt`（Morph 动画）
