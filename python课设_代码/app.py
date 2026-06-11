"""Streamlit 课程设计演示看板。

五个板块对应项目流程：数据概览 → 光谱 PCA → 模型训练与对比 → 光谱预测结果 → 照明补偿演示。
启动方式：streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import dataset_summary
from src.evaluation import sample_error_frame
from src.lighting_compensation import (
    band_error_frame,
    channel_recommendation_frame,
    compensation_summary_frame,
    compute_compensation,
)
from src.pipeline import load_or_train
from src.spectrum_utils import SCENE_TARGET_LUX, WAVELENGTHS
from src.visualization import (
    configure_plot_style,
    plot_compensation_result,
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

tab_data, tab_pca, tab_models, tab_predict, tab_comp = st.tabs(
    ["① 数据概览", "② 光谱 PCA", "③ 模型训练与对比", "④ 光谱预测结果", "⑤ 照明补偿演示"]
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

# ---------------- ⑤ 照明补偿演示 ----------------
with tab_comp:
    st.markdown(
        "**预测光谱的应用：与目标光谱（实测晴天日光均值）对比，"
        "用非负最小二乘求七通道 LED 补偿比例，展示补偿前后误差变化。**"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        scene = st.selectbox("室内场景", list(SCENE_TARGET_LUX.keys()), index=0)
    with c2:
        target_lux = st.number_input("目标照度 / lux", 80, 1000, SCENE_TARGET_LUX[scene], step=10)
    with c3:
        outdoor_lux = st.number_input("室外照度 / lux", 1000, 120000, 50000, step=1000)

    sample_pos = st.session_state.get("sample_pos", 0)
    pred_spec = result.y_pred_spectrum[sample_pos]
    compensation = compute_compensation(
        current_spectrum=pred_spec,
        scene=scene,
        target_lux=float(target_lux),
        outdoor_lux=float(outdoor_lux),
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("补偿前 RMSE", f"{compensation.before_rmse:.4f}")
    m2.metric("补偿后 RMSE", f"{compensation.after_rmse:.4f}")
    m3.metric("误差下降", f"{compensation.improvement_percent:.1f}%")

    left, right = st.columns([1.6, 1])
    with left:
        st.pyplot(plot_compensation_result(compensation), clear_figure=True)
    with right:
        st.markdown("七通道 LED 输出比例：")
        rec = channel_recommendation_frame(compensation.channel_weights)
        st.dataframe(
            rec[["通道", "峰值波长/nm", "输出百分比"]],
            width="stretch",
            hide_index=True,
        )

    with st.expander("分波段误差与补偿明细"):
        st.dataframe(band_error_frame(compensation), width="stretch", hide_index=True)
        st.dataframe(compensation_summary_frame(compensation), width="stretch", hide_index=True)

st.caption(
    "说明：本系统为课程设计演示。天气特征为按地点和时间对齐的历史数据；"
    "光谱估计不能替代现场光谱仪实测；LED 补偿为算法演示，未接入真实硬件。"
)
