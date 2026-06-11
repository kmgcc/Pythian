"""Streamlit 课程设计演示看板。

五个板块对应项目流程：数据概览 → 光谱 PCA → 模型训练与对比 → 光谱预测结果 → 照明补偿演示。
启动方式：streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.application_demo import (
    PRESET_DEFINITIONS,
    PRESET_NAMES,
    build_preset_input,
    channel_percent_frame,
    error_comparison_frame,
    pca_coefficient_frame,
    run_application_demo,
)
from src.data_loader import dataset_summary
from src.evaluation import sample_error_frame
from src.lighting_compensation import band_error_frame, compensation_summary_frame
from src.pipeline import load_or_train
from src.spectrum_utils import SCENE_TARGET_LUX, WAVELENGTHS
from src.visualization import (
    configure_plot_style,
    plot_application_spectra,
    plot_color_circles,
    plot_model_compare,
    plot_pca_variance,
    plot_prediction_compare,
    plot_weather_spectrum_compare,
)

st.set_page_config(page_title="自然光光谱估计与照明补偿", layout="wide")


@st.cache_resource(show_spinner="正在加载数据与模型（首次运行会自动训练）...")
def cached_load():
    return load_or_train()


try:
    dataset, result = cached_load()
except Exception as exc:
    st.error("数据或模型尚未准备好，请先安装依赖并运行流水线。")
    st.code("pip install -r requirements.txt\npython -m src.pipeline", language="bash")
    st.exception(exc)
    st.stop()

configure_plot_style()

st.title("基于真实天光数据的自然光光谱估计与室内照明补偿设计")
st.caption(
    "流程：真实天光光谱数据 → 特征工程 → 光谱 PCA → 五模型对比 → 相对光谱估计 → LED 照明补偿。"
    "数据来源：SKYSPECTRA 实测天光光谱（Zenodo）+ Open-Meteo 历史天气 + 实测 LED 光谱。"
)

tab_data, tab_pca, tab_models, tab_predict, tab_app = st.tabs(
    ["① 数据概览", "② 光谱 PCA", "③ 模型训练与对比", "④ 光谱预测结果", "⑤ 照明补偿应用演示"]
)

# ---------------- ① 数据概览 ----------------
with tab_data:
    st.markdown("**真实公开数据按地点和时间合并成建模数据集，天气为对齐的历史天气特征。**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数", f"{len(dataset):,}")
    c2.metric("观测站数", dataset["station_name"].nunique())
    c3.metric("光谱维度", "41 (380-780nm)")
    c4.metric("时间范围", f"{dataset['date'].min()} ~ {dataset['date'].max()}")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("数据集前几行（环境特征 + 41 列光谱）：")
        st.dataframe(dataset.head(8), width="stretch", hide_index=True, height=240)
    with right:
        st.dataframe(dataset_summary(dataset), width="stretch", hide_index=True, height=240)

    fig = plot_weather_spectrum_compare(dataset)
    st.pyplot(fig, clear_figure=True)

# ---------------- ② 光谱 PCA ----------------
with tab_pca:
    st.markdown(
        "**PCA 把 41 维相对光谱压缩成 5 个主成分系数：模型只需预测 5 个数，"
        "再用 inverse_transform 还原完整光谱曲线。**"
    )
    n_keep = result.spectrum_pca.n_components
    cum = float(result.spectrum_pca.explained_variance_ratio_.sum())
    c1, c2 = st.columns(2)
    c1.metric("保留主成分数量", n_keep)
    c2.metric("累计解释方差", f"{cum * 100:.2f}%")

    left, right = st.columns([1.2, 1])
    with left:
        st.pyplot(plot_pca_variance(result.spectrum_pca.pca), clear_figure=True)
    with right:
        st.dataframe(result.spectrum_pca.variance_frame(), width="stretch", hide_index=True)

# ---------------- ③ 模型训练与对比 ----------------
with tab_models:
    st.markdown(
        "**五个 sklearn 模型在相同特征、相同 PCA 目标、相同训练/测试划分下对比，"
        "按 RMSE 选出最佳模型。**"
    )
    best_row = result.metrics.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("最佳模型", result.best_model_name)
    c2.metric("最佳 RMSE", f"{best_row['RMSE']:.4f}")
    c3.metric("最佳 R²", f"{best_row['R2']:.4f}")

    st.dataframe(result.metrics, width="stretch", hide_index=True)
    st.pyplot(plot_model_compare(result.metrics), clear_figure=True)

# ---------------- ④ 光谱预测结果 ----------------
with tab_predict:
    st.markdown("**从测试集选一条样本：实测相对光谱 vs 最佳模型预测光谱。**")
    n_test = len(result.y_test_spectrum)
    if "sample_pos" not in st.session_state:
        st.session_state["sample_pos"] = 0

    c_btn, c_slider = st.columns([1, 3])
    with c_btn:
        if st.button("🎲 随机抽一条测试样本"):
            st.session_state["sample_pos"] = int(np.random.randint(0, n_test))
    with c_slider:
        sample_pos = st.slider("测试样本序号", 0, n_test - 1, key="sample_pos")

    true_spec = result.y_test_spectrum[sample_pos]
    pred_spec = result.y_pred_spectrum[sample_pos]

    st.markdown("该样本的环境输入：")
    st.dataframe(result.x_test.iloc[[sample_pos]], width="stretch", hide_index=True)

    left, right = st.columns([1.6, 1])
    with left:
        st.pyplot(plot_prediction_compare(true_spec, pred_spec), clear_figure=True)
    with right:
        st.markdown("该样本误差：")
        st.dataframe(sample_error_frame(true_spec, pred_spec), width="stretch", hide_index=True)

# ---------------- ⑤ 照明补偿应用演示 ----------------
with tab_app:
    st.markdown(
        "**应用演示：天气场景 → 光谱预测 → 照明补偿。**"
        "选择天气预设后，系统从真实数据生成模型输入，预测自然光相对光谱，"
        "计算多通道 LED 补偿方案，并与传统双色温 LED 对照。"
    )

    col_params, col_result = st.columns([1, 2.1])
    with col_params:
        preset = st.radio(
            "天气预设",
            PRESET_NAMES,
            index=0,
            captions=[PRESET_DEFINITIONS[name]["说明"] for name in PRESET_NAMES],
        )
        scene = st.selectbox("室内场景", list(SCENE_TARGET_LUX.keys()), index=0)
        target_lux = st.number_input(
            "目标照度 / lux", 80, 1000, SCENE_TARGET_LUX[scene], step=10
        )

        # 预设值作为默认值；widget key 带预设名，切换预设时各自保留修改
        preset_row = build_preset_input(dataset, preset).iloc[0]
        overrides: dict[str, object] = {}
        with st.expander("自定义参数（默认取预设值）"):
            k = f"ov_{preset}_"
            overrides["weather"] = st.selectbox(
                "天气类别", sorted(dataset["weather"].unique()),
                index=sorted(dataset["weather"].unique()).index(preset_row["weather"]), key=k + "weather",
            )
            overrides["sky_condition"] = st.selectbox(
                "天空状况", sorted(dataset["sky_condition"].unique()),
                index=sorted(dataset["sky_condition"].unique()).index(preset_row["sky_condition"]), key=k + "sky",
            )
            overrides["hour"] = float(st.slider("小时", 7, 19, int(preset_row["hour"]), key=k + "hour"))
            overrides["month"] = float(st.slider("月份", 1, 12, int(preset_row["month"]), key=k + "month"))
            overrides["cloud_cover"] = st.slider("云量", 0.0, 1.0, float(preset_row["cloud_cover"]), 0.05, key=k + "cloud")
            overrides["humidity"] = st.slider("相对湿度", 0.0, 1.0, float(preset_row["humidity"]), 0.05, key=k + "hum")
            overrides["temperature"] = st.slider("气温 / °C", -10.0, 40.0, float(preset_row["temperature"]), 0.5, key=k + "temp")
            overrides["precipitation"] = st.slider("降水量 / mm", 0.0, 20.0, float(preset_row["precipitation"]), 0.1, key=k + "precip")
            overrides["solar_altitude"] = st.slider("太阳高度角 / °", 0.0, 90.0, float(preset_row["solar_altitude"]), 0.5, key=k + "alt")
            overrides["solar_azimuth"] = st.slider("太阳方位角 / °", 0.0, 360.0, float(preset_row["solar_azimuth"]), 1.0, key=k + "azi")
            overrides["outdoor_lux"] = float(
                st.number_input("室外照度 / lux", 500, 120000, int(preset_row["outdoor_lux"]), step=1000, key=k + "lux")
            )

    demo = run_application_demo(
        result.predictor,
        dataset,
        preset=preset,
        overrides=overrides,
        scene=scene,
        target_lux=float(target_lux),
    )

    with col_result:
        st.markdown("当前输入参数（由预设和自定义参数生成）：")
        st.dataframe(demo.features, width="stretch", hide_index=True)

        c_pca, c_channel = st.columns([1, 1.4])
        with c_pca:
            st.markdown("模型预测的 PCA 主成分系数：")
            st.dataframe(pca_coefficient_frame(demo), width="stretch", hide_index=True)
        with c_channel:
            st.markdown("多通道 LED 补偿比例：")
            st.dataframe(channel_percent_frame(demo), width="stretch", hide_index=True)
        st.caption(
            f"传统双色温对照方案：{demo.dual_cct_label}"
            f"（在冷暖配比和亮度两个旋钮下与目标光谱误差最小的组合）"
        )

    st.divider()
    c_curve, c_circle = st.columns([1.15, 1])
    with c_curve:
        st.pyplot(plot_application_spectra(demo), clear_figure=True)
    with c_circle:
        st.pyplot(plot_color_circles(demo), clear_figure=True)
        st.dataframe(error_comparison_frame(demo), width="stretch", hide_index=True)

    st.caption(
        "屏幕颜色仅为根据光谱计算得到的近似色度预览（CIE 2015 10° 观察者，亮度归一化），"
        "不能替代真实光谱视觉效果。不同光谱可能在屏幕上显示为相近颜色，但其光谱组成仍然不同。"
    )
    with st.expander("分波段误差与补偿明细"):
        st.dataframe(band_error_frame(demo.compensation), width="stretch", hide_index=True)
        st.dataframe(compensation_summary_frame(demo.compensation), width="stretch", hide_index=True)

st.caption(
    "说明：本系统为课程设计演示。天气特征为按地点和时间对齐的历史数据；"
    "光谱估计不能替代现场光谱仪实测；LED 补偿为算法演示，未接入真实硬件。"
)
