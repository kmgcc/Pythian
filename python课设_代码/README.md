# 基于真实天光数据的自然光光谱估计与室内照明补偿设计

Python 课程设计代码部分。本项目面向未来高质量室内照明需求，使用公开实测天光光谱数据，结合天气、时间、地点、太阳位置等低成本环境特征，训练机器学习模型估计自然光相对光谱，并进一步用于室内照明补偿设计。

整条流程：

> 环境特征 → 模型预测 PCA 主成分系数 → PCA 逆变换 → 380nm-780nm 相对光谱 → 与目标光谱对比 → 七通道 LED 补偿比例 → 补偿前后误差

本项目不试图完整解决真实建筑照明中的复杂问题，而是从课程设计角度出发，验证"低成本环境特征 → 自然光光谱估计 → 照明补偿"这条流程的可行性。

## 数据来源

全部数据为公开真实数据，由 Python 代码联网获取并本地缓存：

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 实测天光光谱及元数据 | SKYSPECTRA 数据集（[Zenodo record 8147546](https://zenodo.org/records/8147546)） | 光谱标签、地点、时间、太阳位置、室外照度、天空状况 |
| 历史小时天气 | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | 云量、湿度、温度、降水 |
| 实测 LED 光谱 | [Harald Brendel LED SPD 数据](https://haraldbrendel.com/ledspd.html) | 补偿用七通道 LED 光谱 |

最终数据集 `data/real_spectrum_weather_dataset.csv`：5664 行 × 58 列，光谱统一为 380nm-780nm、每 10nm 一点（41 维，列名 `wavelength_380` ~ `wavelength_780`）。

说明：天气数据为按地点和时间对齐的历史天气特征，不等同于现场同步气象测量。

## 安装依赖

建议 Python 3.10 及以上：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn streamlit requests
```

或直接安装完整依赖（含 Notebook 环境）：

```bash
pip install -r requirements.txt
```

## 运行方式

**一键流水线**（数据下载 → 预处理 → PCA → 五模型训练 → 评价 → 图表导出），可用于最终检查：

```bash
python -m src.pipeline
```

**Notebook 演示**（推荐，7 个章节分步展示完整流程）：

```bash
jupyter notebook main.ipynb
```

**Streamlit 演示看板**（5 个板块：数据概览 / 光谱 PCA / 模型对比 / 光谱预测 / 照明补偿）：

```bash
streamlit run app.py
```

首次运行会自动下载数据并训练模型（光谱原始文件约 78MB，需要网络）；之后均使用本地缓存。

## 项目结构

```text
.
├── main.ipynb              # 7 章节流程演示
├── app.py                  # Streamlit 五板块看板
├── requirements.txt
├── data/                   # 数据集与外部原始数据缓存
├── models/                 # 训练结果 (joblib)
├── outputs/
│   ├── figures/            # 导出图表
│   └── results/            # 指标与结果 CSV
└── src/
    ├── real_data_pipeline.py   # 真实数据下载、清洗、合并、天气对齐、重采样
    ├── weather_api.py          # Open-Meteo 历史天气（带缓存）
    ├── data_loader.py          # 数据集读取与信息汇总
    ├── preprocessing.py        # 特征工程：独热编码、标准化、相对光谱归一化
    ├── spectrum_pca.py         # 光谱 PCA 降维与还原
    ├── model_training.py       # 五模型训练（含 MLP）、计时、最佳模型封装
    ├── evaluation.py           # MAE/RMSE/R²/时间 统一评价
    ├── led_spectrum_data.py    # 实测 LED 光谱获取与通道构建
    ├── lighting_compensation.py # 七通道 LED 补偿算法
    ├── visualization.py        # matplotlib/seaborn 图表
    └── pipeline.py             # 一键编排
```

## 输出结果

运行流水线后生成：

- `data/real_spectrum_weather_dataset.csv` —— 最终建模数据集
- `models/spectrum_model.joblib` —— 训练结果（含最佳模型与 PCA）
- `outputs/results/model_metrics.csv` —— 五模型指标表（MAE/RMSE/R²/训练时间/预测时间）
- `outputs/results/pca_variance.csv` —— PCA 解释方差
- `outputs/results/feature_importance.csv` —— 随机森林特征重要性
- `outputs/figures/*.png` —— 全部图表

## 注意事项

- 模型为相对光谱估计（每条光谱按自身最大值归一化），不预测绝对辐照强度；
- 模型估计不能替代现场光谱仪实测，LED 补偿为算法演示，未接入真实硬件；
- 若 Zenodo 或 Open-Meteo 请求失败，流水线会给出明确报错并中止，不会用伪造数据顶替，请检查网络后重试。
