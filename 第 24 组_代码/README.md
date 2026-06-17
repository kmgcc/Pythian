# 自然光光谱估计与室内照明补偿

## 运行

```bash
pip install -r requirements.txt
```

**一键流水线**（数据下载 → 预处理 → PCA → 模型训练 → 评价 → 图表导出）：

```bash
python -m src.pipeline
```

**Notebook 演示**（7 章节分步展示）：

```bash
jupyter notebook main.ipynb
```

**Web 演示看板**（数据概览 / PCA / 模型对比 / 光谱预测 / 照明补偿）：

```bash
python start_web.py
```

首次运行会自动下载数据并训练模型（约 78MB），之后使用本地缓存。

## 项目结构

```text
python课设_代码/
├── main.ipynb                  # Notebook 演示
├── app.py                      # Streamlit 看板
├── start_web.py                # Web 看板启动器（自动打开浏览器）
├── requirements.txt
├── src/
│   ├── real_data_pipeline.py   # 数据下载、清洗、合并、重采样
│   ├── weather_api.py          # Open-Meteo 历史天气（带缓存）
│   ├── data_loader.py          # 数据集读取
│   ├── preprocessing.py        # 特征工程（独热编码、标准化）
│   ├── spectrum_pca.py         # PCA 降维与还原
│   ├── model_training.py       # 五模型训练与对比
│   ├── evaluation.py           # MAE/RMSE/R² 评价
│   ├── hyperparameter_tuning.py # GridSearchCV 超参数调优
│   ├── led_spectrum_data.py    # 实测 LED 光谱获取
│   ├── lighting_compensation.py # 七通道 LED 补偿算法
│   ├── color_conversion.py     # CIE 2015 光谱→颜色换算
│   ├── application_demo.py     # 天气预设演示
│   ├── visualization.py        # 图表绘制
│   └── pipeline.py             # 一键编排
├── data/                       # 数据集（运行时生成/下载）
├── models/                     # 训练模型（运行时生成）
└── outputs/                    # 图表与结果 CSV（运行时生成）
```
