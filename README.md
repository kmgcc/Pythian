# 基于机器学习的自然光相对光谱预测与室内照明补偿设计

这是 Python 课程设计的代码部分，实现了计划书中的完整流程：

- 生成 380nm-780nm、每 10nm 一个采样点的自然光相对光谱数据集；
- 使用天气、时间、太阳高度角、云量、湿度、室外照度等特征构建模拟样本；
- 使用 PCA 将 41 维光谱降到 5 个主成分；
- 对比 Linear Regression、KNN、Decision Tree、Random Forest 回归模型；
- 使用 MAE、RMSE、R² 评价模型；
- 将预测 PCA 系数还原为完整光谱曲线；
- 根据目标光谱计算七通道 LED 室内照明补偿比例；
- 输出 Notebook、Streamlit 页面、CSV 数据、模型结果和图表。

## 运行环境

建议使用 Python 3.10 及以上版本。

课程要求的核心依赖安装命令：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

核心依赖包括：

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

完整演示环境还需要 Streamlit、Notebook、Requests、Joblib 等辅助依赖，推荐直接安装：

```bash
pip install -r requirements.txt
```

项目正式可视化统一使用 matplotlib，必要时使用 seaborn；如果运行环境缺少 matplotlib 或 seaborn，请先安装依赖后再生成图表。

## 一键生成数据、模型和图表

```bash
python run_pipeline.py
```

可选参数：

```bash
python run_pipeline.py --samples 1800 --seed 42
```

运行后会生成：

- `data/base_spectrum.csv`
- `data/simulated_spectrum_dataset.csv`
- `data/sample_weather.csv`
- `models/spectrum_model.joblib`
- `outputs/results/model_metrics.csv`
- `outputs/results/pca_variance.csv`
- `outputs/results/feature_importance.csv`
- `outputs/figures/*.png`

说明：正式实验流程使用 `scikit-learn` 的标准实现完成 PCA、训练集/测试集划分、特征预处理、回归建模和模型评价。

## Notebook 演示

打开并按顺序运行：

```bash
jupyter notebook main.ipynb
```

Notebook 包含数据预览、光谱图、PCA 解释方差、模型对比、预测光谱与照明补偿结果。

## Streamlit 可视化页面

```bash
streamlit run app.py
```

页面包含：

- 数据集预览；
- 不同天气下的光谱对比；
- PCA 主成分解释方差；
- 多模型评价指标；
- 输入天气条件后的光谱预测；
- 七通道 LED 补偿比例；
- 目标光谱、当前自然光贡献和补偿后光谱对比。

## 项目结构

```text
.
├── app.py
├── main.ipynb
├── run_pipeline.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── outputs/
│   ├── figures/
│   └── results/
└── src/
    ├── data_generator.py
    ├── lighting_compensation.py
    ├── pipeline.py
    ├── spectrum_model.py
    ├── visualization.py
    └── weather_api.py
```

## 说明

本项目使用的是基于标准日光光谱和简化天气规则生成的模拟数据，目的是验证“自然光相对光谱预测 + 室内照明补偿”的课程设计流程。模型结果不代表真实工程环境中的高精度太阳光谱预测；LED 补偿也属于算法演示方案，未接入真实硬件。
