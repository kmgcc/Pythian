# 基于真实天光数据的自然光光谱估计与室内照明补偿设计

Python 课程设计代码部分。本项目面向未来高质量室内照明需求，使用公开实测天光光谱数据，结合天气、时间、地点、太阳位置等低成本环境特征，训练机器学习模型估计自然光相对光谱，并进一步用于室内照明补偿设计。

整条流程：

> 环境特征 → 模型预测 PCA 主成分系数 → PCA 逆变换 → 380nm-780nm 相对光谱 → 与目标光谱对比 → 七通道 LED 补偿比例 → 与传统双色温 LED 对照 → 补偿前后误差与近似色度预览

应用演示提供四个天气预设（晴天中午 / 多云下午 / 阴雨天气 / 傍晚低太阳高度角），预设参数从真实数据集对应场景子集的中位数/众数生成，也支持自定义参数。

本项目不试图完整解决真实建筑照明中的复杂问题，而是从课程设计角度出发，验证"低成本环境特征 → 自然光光谱估计 → 照明补偿"这条流程的可行性。

## 数据来源

全部数据为公开真实数据，由 Python 代码联网获取并本地缓存：

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 实测天光光谱及元数据 | SKYSPECTRA 数据集（[Zenodo record 8147546](https://zenodo.org/records/8147546)） | 光谱标签、地点、时间、太阳位置、室外照度、天空状况 |
| 历史小时天气 | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | 云量、湿度、温度、降水 |
| 实测 LED 光谱 | [Harald Brendel LED SPD 数据](https://haraldbrendel.com/ledspd.html) | 补偿用七通道 LED 光谱 |
| CIE 2015 标准观察者色匹配函数 | [CVRL（UCL 颜色与视觉研究实验室）](http://www.cvrl.org/)，CIE 170-2:2015，基于 CIE 2006 生理学数据 | 光谱 → 屏幕近似色度换算（缓存在 `data/color_matching/`，刻意不使用 1931 年旧观察者） |

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

**Streamlit 演示看板**（5 个板块：数据概览 / 光谱 PCA / 模型对比 / 光谱预测 / 照明补偿应用演示）：

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
    ├── lighting_compensation.py # 七通道 LED 补偿算法 + 传统双色温对照
    ├── color_conversion.py     # CIE 2015 观察者光谱→屏幕近似颜色换算
    ├── application_demo.py     # 应用演示：天气预设 → 光谱预测 → 补偿对比
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
- `outputs/results/application_preset_summary.csv` —— 四个天气预设的补偿效果汇总（多通道 vs 双色温）
- `outputs/results/application_color_preview.csv` —— 各方案光谱的近似显示色
- `outputs/figures/*.png` —— 全部图表（含应用演示光谱对比图与白圆色度对比图）

## 注意事项

- 模型为相对光谱估计（每条光谱按自身最大值归一化），不预测绝对辐照强度；
- 模型估计不能替代现场光谱仪实测，LED 补偿为算法演示，未接入真实硬件；
- 白圆演示中的屏幕颜色仅为根据光谱计算得到的近似色度预览（CIE 2015 10° 观察者，亮度归一化），不能替代真实光谱视觉效果——不同光谱可能在屏幕上显示为相近颜色，但其光谱组成仍然不同；
- 若 Zenodo 或 Open-Meteo 请求失败，流水线会给出明确报错并中止，不会用伪造数据顶替，请检查网络后重试。
