# 课程设计重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已实现的"自然光光谱估计与照明补偿"课程设计整理为流程清晰、表述真实、适合答辩的版本(题目统一为《基于真实天光数据的自然光光谱估计与室内照明补偿设计》)。

**Architecture:** 保留 real_data_pipeline / weather_api / led_spectrum_data / lighting_compensation / visualization / spectrum_utils,把 spectrum_model.py 拆为 data_loader / preprocessing / spectrum_pca / model_training / evaluation,pipeline.py 仍作一键编排。Notebook 重写为 7 节真实流程,app.py 重写为 5 板块 matplotlib 演示,报告与 README 同步。

**Tech Stack:** pandas / numpy / scikit-learn / matplotlib / seaborn / streamlit / requests / joblib。无测试基建,验证方式为:数据管线可重跑、notebook 全量执行、streamlit 可启动、pipeline 一键可跑。

**关键决策(已根据现状确认):**
1. FR-VLX 站点当前被硬编码为"巴黎"——错误,实为法国 Vaulx-en-Velin(里昂附近)。改为站名直接取自 meta_location.csv,不再硬编码城市映射。
2. 模型目标从绝对光谱改为**相对光谱**(逐行 max 归一化,代码注释+报告说明),与题目"相对光谱估计"一致,也消除 outdoor_lux 主导重要性的尴尬。
3. 输入特征扩充:hour、month、solar_altitude、solar_azimuth、cloud_cover、humidity、temperature、precipitation、outdoor_lux(数值)+ weather、location_code(类别独热)。sun_azimuth 取自 meta_sun_positions.csv,需重建数据集(原始文件已全部本地缓存,可离线重建)。
4. 五模型:LinearRegression、KNeighborsRegressor、DecisionTreeRegressor、RandomForestRegressor、MLPRegressor;统一记录 MAE/RMSE/R²/训练时间/预测时间;最佳模型由真实结果决定。
5. app.py 去掉 plotly、astral 依赖,统一 matplotlib;5 板块:数据概览 / 光谱 PCA / 模型训练与对比 / 光谱预测结果 / 照明补偿演示。
6. 清除一切"模拟数据主流程/模拟真实光谱"表述(visualization.py 图题、notebook、README、报告)。
7. 无 PPT 草稿,跳过第十二节;docs/计划书保持原样(历史文档)。

### Task 1: src 模块重构

**Files:**
- Create: `python课设_代码/src/data_loader.py` — load_dataset() + dataset_summary()
- Create: `python课设_代码/src/preprocessing.py` — 特征列定义、build_preprocessor(标准化+独热)、extract_features_targets(含相对光谱归一化)、split_train_test
- Create: `python课设_代码/src/spectrum_pca.py` — SpectrumPCA(StandardScaler+PCA 封装:fit/transform/inverse_transform/variance_frame)
- Create: `python课设_代码/src/model_training.py` — build_models()(5 模型)、train_models()(循环训练,逐模型打印状态、计时)、TrainingResult、predict_spectrum、save/load
- Create: `python课设_代码/src/evaluation.py` — evaluate_predictions(MAE/RMSE/R²)、metrics 表(含时间)、select_best_model
- Delete: `python课设_代码/src/spectrum_model.py`(逻辑迁入上述模块)
- Modify: `python课设_代码/src/real_data_pipeline.py` — 拆 main 为 ensure_raw_files()/build_dataset(),站名取元数据,新增 month/sun_azimuth/latitude/longitude 列
- Modify: `python课设_代码/src/visualization.py` — 修图题违禁词、模型对比图适配 5 模型+时间、hourly_lux 标题改"实测"
- Modify: `python课设_代码/src/pipeline.py` — 用新模块重写编排,加 `__main__` 入口
- [ ] 写新模块,迁移逻辑,删 spectrum_model.py
- [ ] 重建数据集(离线缓存):`python -m src.real_data_pipeline`
- [ ] 一键流水线跑通:`python -m src.pipeline`,生成 metrics/figures/model

### Task 2: Notebook 重写(7 节)

**Files:** Rewrite `python课设_代码/main.ipynb`
- [ ] 1 数据来源与下载(展示来源/缓存/文件列表/行列数,不训练)
- [ ] 2 数据观察与预处理(head/shape/columns/缺失/光谱列数/站点/时间范围)
- [ ] 3 特征工程与光谱 PCA(特征列表、独热/标准化说明、解释方差图、主成分数选择)
- [ ] 4 模型训练(循环训练 5 模型,打印状态与耗时)
- [ ] 5 模型评估与最优模型选择(指标表、对比图、最佳模型+理由)
- [ ] 6 光谱预测与还原(测试样本真实 vs 预测曲线+误差)
- [ ] 7 照明补偿应用展示(目标/预测/补偿后光谱、通道比例、误差下降)
- [ ] `jupyter nbconvert --execute` 全量跑通

### Task 3: Streamlit 重写(5 板块)

**Files:** Rewrite `python课设_代码/app.py`;Modify `requirements.txt`(去 astral,补齐实际依赖)
- [ ] 5 板块、每板块一句说明+核心图表/表格,matplotlib 绘图
- [ ] 样本选择器/随机样本按钮 → 真实 vs 预测 → 补偿演示链路
- [ ] `streamlit run app.py` 启动验证

### Task 4: 文档同步

**Files:** Rewrite `python课设_代码/README.md`;Rewrite `python课设_课程报告/课程报告.md`;Copy figures → `python课设_课程报告/images/`
- [ ] README:简介/数据来源/依赖/运行方式/结构/输出/注意事项
- [ ] 报告:题目更新、15 节结构、5 模型真实结果、相对光谱说明、成本方案对比、困难与解决、不足与改进;学生口吻
- [ ] 图表重新生成并同步到报告 images/

### Task 5: 质量检查清单 + 提交

- [ ] 按需求第十四节逐项输出检查结果
- [ ] 全文 grep 违禁表述("模拟数据主流程/模拟真实光谱/基于规则生成"等)
- [ ] 分逻辑块 git commit
