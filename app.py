from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import SCENE_TARGET_LUX, SPECTRUM_COLUMNS, WAVELENGTHS, solar_altitude_from_hour
from src.lighting_compensation import (
    band_error_frame,
    channel_recommendation_frame,
    compensation_summary_frame,
    compute_compensation,
    CHANNEL_NAMES,
)
from src.pipeline import load_or_train, run_full_pipeline
from src.spectrum_model import predict_spectrum
from src.visualization import (
    light_color_comparison_frame,
    plot_light_halo_comparison,
)

# Page configuration
st.set_page_config(
    page_title="自然光光谱预测与照明补偿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fb;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e9ef;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 0.55rem 0.9rem;
        font-weight: 500;
    }
    .section-note {
        border-left: 4px solid #2f6fed;
        background: #ffffff;
        padding: 0.8rem 1rem;
        color: #334155;
        border-radius: 0 6px 6px 0;
        margin: 0.6rem 0 1rem 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 10px;
        margin: 0.8rem 0 1rem 0;
    }
    .flow-step {
        background: #ffffff;
        border: 1px solid #e5e9ef;
        border-radius: 8px;
        padding: 0.72rem 0.78rem;
        min-height: 72px;
        color: #172033;
        font-size: 0.92rem;
        line-height: 1.35;
        box-shadow: 0 1px 2px rgba(0,0,0,0.01);
    }
    @media (max-width: 760px) {
        .flow-grid {
            grid-template-columns: repeat(2, minmax(120px, 1fr));
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_load():
    return load_or_train()


def estimate_outdoor_lux(weather: str, cloud_cover: float, humidity: float, solar_altitude: float) -> float:
    weather_transmission = {"晴": 1.00, "多云": 0.68, "阴": 0.42, "雨": 0.25}[weather]
    altitude_factor = max(np.sin(np.deg2rad(max(solar_altitude, 0.0))), 0.0) ** 0.55
    cloud_factor = 1.0 - 0.62 * cloud_cover
    humidity_factor = 1.0 - 0.13 * humidity
    return float(np.clip(105000 * altitude_factor * weather_transmission * cloud_factor * humidity_factor, 300, 115000))


st.title("自然光相对光谱预测与室内照明补偿")
st.caption("基于机器学习的自然光光谱还原与多通道可调 LED 光谱级室内补光仿真系统")

# Load dataset and trained models
try:
    with st.spinner("正在准备数据与模型..."):
        dataset, model_result = cached_load()
except Exception as exc:
    st.error("项目依赖或模型文件尚未准备好。")
    st.code("pip install -r requirements.txt\npython run_pipeline.py\nstreamlit run app.py", language="bash")
    st.exception(exc)
    st.stop()

# Basic setup and best model retrieval
metrics = model_result.metrics.copy()
best_row = metrics[metrics["model"] == model_result.best_model_name].iloc[0]

# Side bar input variables
with st.sidebar:
    st.subheader("环境输入条件")
    weather = st.selectbox("天气状况", ["晴", "多云", "阴", "雨"], index=0)
    hour = st.slider("时间 (小时)", min_value=6, max_value=18, value=12, step=1)
    cloud_cover = st.slider("云量", min_value=0.0, max_value=1.0, value=0.18, step=0.01)
    humidity = st.slider("环境湿度", min_value=0.2, max_value=1.0, value=0.55, step=0.01)
    temperature = st.slider("环境温度 / ℃", min_value=0.0, max_value=40.0, value=25.0, step=0.5)
    precipitation = st.slider("降水强度", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    
    solar_altitude = solar_altitude_from_hour(hour)
    outdoor_lux = estimate_outdoor_lux(weather, cloud_cover, humidity, solar_altitude)
    
    st.divider()
    st.subheader("室内场景与照度")
    scene = st.selectbox("室内使用场景", list(SCENE_TARGET_LUX.keys()), index=0)
    target_lux = st.number_input("目标照度 / lux", min_value=80, max_value=1000, value=SCENE_TARGET_LUX[scene], step=10)
    
    st.divider()
    if st.button("重新生成训练结果"):
        st.cache_resource.clear()
        with st.spinner("正在重新生成数据、训练模型并导出图表..."):
            run_full_pipeline()
        st.rerun()

# Run predictions and compensation logic
input_features = {
    "hour": hour,
    "cloud_cover": cloud_cover,
    "humidity": humidity,
    "temperature": temperature,
    "precipitation": precipitation,
    "solar_altitude": solar_altitude,
    "outdoor_lux": outdoor_lux,
    "weather": weather,
}
pred_spectrum = predict_spectrum(model_result, input_features)
compensation = compute_compensation(
    current_spectrum=pred_spectrum,
    scene=scene,
    target_lux=float(target_lux),
    outdoor_lux=outdoor_lux,
)

# Top 5 core metrics cards
top_cols = st.columns(5)
with top_cols[0]:
    st.metric("数据集样本量", f"{len(dataset):,}")
with top_cols[1]:
    st.metric("预测最佳模型", model_result.best_model_name)
with top_cols[2]:
    st.metric("累计解释方差", f"{model_result.pca.explained_variance_ratio_.sum() * 100:.1f}%")
with top_cols[3]:
    st.metric("室外照度估计", f"{outdoor_lux:,.0f} lx")
with top_cols[4]:
    st.metric("补偿后综合误差", f"{compensation.after_rmse:.4f}")

# Streamlit Tab navigation layout
tabs = st.tabs(["数据预览", "光谱分析", "PCA 与模型", "预测与补偿"])

with tabs[0]:
    st.subheader("气象特征与样本光谱数据预览")
    preview_cols = [
        "sample_id",
        "date",
        "hour",
        "weather",
        "scene",
        "solar_altitude",
        "cloud_cover",
        "humidity",
        "temperature",
        "precipitation",
        "outdoor_lux",
        "target_lux",
    ]
    st.dataframe(dataset[preview_cols].head(15), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            dataset,
            x="solar_altitude",
            y="outdoor_lux",
            color="weather",
            size="cloud_cover",
            opacity=0.68,
            title="太阳高度角、云量与室外照度的物理映射",
            labels={"solar_altitude": "太阳高度角 (°)", "outdoor_lux": "室外照度 (lux)", "weather": "天气"},
            color_discrete_map={"晴": "#f9a825", "多云": "#78909c", "阴": "#546e7a", "雨": "#1e88e5"},
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        hourly_lux = dataset.groupby(["hour", "weather"], as_index=False)["outdoor_lux"].mean()
        fig = px.line(
            hourly_lux,
            x="hour",
            y="outdoor_lux",
            color="weather",
            markers=True,
            title="一天中各天气下平均照度变化曲线",
            labels={"hour": "时间 (小时)", "outdoor_lux": "平均室外照度 (lux)", "weather": "天气"},
            color_discrete_map={"晴": "#f9a825", "多云": "#78909c", "阴": "#546e7a", "雨": "#1e88e5"},
        )
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("日光光谱物理特性分析")
    
    # Load standard solar spectrum from data folder
    base_spectrum_df = pd.read_csv("data/base_spectrum.csv")
    fig = px.line(
        base_spectrum_df,
        x="wavelength_nm",
        y="relative_intensity",
        title="标准太阳光相对光谱（AM1.5G 可见光区 380nm-780nm）",
        labels={"wavelength_nm": "波长 (nm)", "relative_intensity": "相对光谱强度"},
    )
    fig.update_traces(line=dict(color="#f9a825", width=3))
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Calculate average weather relative spectrum
    compare_rows = []
    for w in ["晴", "多云", "阴", "雨"]:
        subset = dataset[dataset["weather"] == w]
        if not subset.empty:
            mean_spec = subset[SPECTRUM_COLUMNS].mean().to_numpy()
            mean_spec = mean_spec / mean_spec.max()  # normalize
            for wl, val in zip(WAVELENGTHS, mean_spec):
                compare_rows.append({"波长(nm)": int(wl), "相对强度": float(val), "天气状况": f"{w}天平均相对光谱"})
    compare_df = pd.DataFrame(compare_rows)
    
    fig2 = px.line(
        compare_df,
        x="波长(nm)",
        y="相对强度",
        color="天气状况",
        title="不同天气状况下自然光平均相对光谱对比",
        color_discrete_map={
            "晴天平均相对光谱": "#f9a825",
            "多云天平均相对光谱": "#78909c",
            "阴天平均相对光谱": "#546e7a",
            "雨天平均相对光谱": "#1e88e5",
        }
    )
    fig2.update_traces(line=dict(width=3))
    fig2.update_layout(height=400, hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
    st.plotly_chart(fig2, use_container_width=True)

with tabs[2]:
    st.subheader("PCA 降维提取与多模型回归拟合评估")
    
    explained_variance = model_result.pca.explained_variance_ratio_
    components = [f"PC{i}" for i in range(1, len(explained_variance) + 1)]
    
    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        fig = go.Figure()
        fig.add_bar(x=components, y=explained_variance, name="单个主成分方差占比", marker_color="#6f4e9b")
        fig.add_trace(
            go.Scatter(
                x=components,
                y=np.cumsum(explained_variance),
                name="累计解释方差",
                mode="lines+markers",
                line=dict(color="#d95f02", width=3),
            )
        )
        fig.update_layout(
            title="PCA 主成分提取解释方差",
            yaxis_title="方差占比",
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        metric_long = metrics.melt(id_vars="model", value_vars=["MAE", "RMSE", "R2"], var_name="指标", value_name="数值")
        fig = px.bar(
            metric_long,
            x="model",
            y="数值",
            color="指标",
            barmode="group",
            title="回归预测模型效果评价对比 (MAE / RMSE / R²)",
            color_discrete_map={"MAE": "#1e88e5", "RMSE": "#e53935", "R2": "#43a047"},
        )
        fig.update_layout(height=290, margin=dict(l=20, r=20, t=50, b=20), xaxis_title="", yaxis_title="指标得分")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("模型测试集预测表现抽样比对")
    sample_number = st.slider("测试集样本序号", 0, len(model_result.y_test_spectrum) - 1, 0)
    true_spec = model_result.y_test_spectrum[sample_number]
    pred_spec = model_result.y_pred_spectrum[sample_number]
    
    pred_compare_rows = []
    for wl, t_val, p_val in zip(WAVELENGTHS, true_spec, pred_spec):
        pred_compare_rows.append({"波长(nm)": int(wl), "相对强度": float(t_val), "曲线": "模拟真实光谱"})
        pred_compare_rows.append({"波长(nm)": int(wl), "相对强度": float(p_val), "曲线": "模型预测光谱"})
    pred_compare_df = pd.DataFrame(pred_compare_rows)
    
    fig3 = px.line(
        pred_compare_df,
        x="波长(nm)",
        y="相对强度",
        color="曲线",
        title=f"随机样本 #{sample_number}：预测光谱与模拟真实光谱的吻合度",
        color_discrete_map={"模拟真实光谱": "#2f6fed", "模型预测光谱": "#f59e0b"},
    )
    fig3.update_traces(line=dict(width=3))
    fig3.update_layout(height=380, hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20), legend_title_text="")
    st.plotly_chart(fig3, use_container_width=True)

with tabs[3]:
    st.subheader("室内七通道 LED 补偿仿真实验")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("有效 LED 通道", f"{compensation.active_channel_count} / 7", help="输出比例 >= 0.5% 的通道数")
    m2.metric("重合度改善比例", f"{compensation.improvement_percent:.1f}%")
    m3.metric("补光前 RMSE", f"{compensation.before_rmse:.4f}")
    m4.metric("补光后 RMSE", f"{compensation.after_rmse:.4f}")

    # Plot predicted spectrum
    predict_df = pd.DataFrame({
        "波长(nm)": WAVELENGTHS,
        "相对强度": pred_spectrum
    })
    fig_pred = px.line(predict_df, x="波长(nm)", y="相对强度", title="输入条件预测出的自然光相对光谱形态")
    fig_pred.update_traces(line=dict(color="#1f77b4", width=3))
    fig_pred.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_pred, use_container_width=True)

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        # Build LED weights bar chart
        channel_colors = {
            "深蓝/蓝光": "#1e88e5",
            "青光": "#00acc1",
            "绿光": "#2f9e44",
            "琥珀光": "#f9a825",
            "红光": "#e53935",
            "暖白": "#ffb300",
            "冷白": "#80deea"
        }
        weights_df = pd.DataFrame({
            "通道": CHANNEL_NAMES,
            "推荐比例": compensation.channel_weights,
            "输出比例(%)": np.round(compensation.channel_weights * 100, 1),
            "颜色": [channel_colors[name] for name in CHANNEL_NAMES]
        })
        fig_weights = px.bar(
            weights_df,
            x="通道",
            y="输出比例(%)",
            color="通道",
            title="七通道 LED 单色输出负荷比 (优化目标计算)",
            color_discrete_sequence=weights_df["颜色"].tolist(),
            text="输出比例(%)",
        )
        fig_weights.update_traces(textposition="outside", cliponaxis=False)
        fig_weights.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=70),
            xaxis_title="LED 通道",
            yaxis_title="输出比例 (%)",
            yaxis_range=[0, 112],
        )
        st.plotly_chart(fig_weights, use_container_width=True)
    with c2:
        # Plot target vs current vs compensated spectrum curves
        comp_rows = []
        for wl, t_val, n_val, c_val in zip(WAVELENGTHS, compensation.target_spectrum, compensation.current_spectrum, compensation.compensated_spectrum):
            comp_rows.append({"波长(nm)": int(wl), "相对强度": float(t_val), "光谱曲线": "目标光谱"})
            comp_rows.append({"波长(nm)": int(wl), "相对强度": float(n_val), "光谱曲线": "当前室内自然光"})
            comp_rows.append({"波长(nm)": int(wl), "相对强度": float(c_val), "光谱曲线": "合成后室内光谱"})
        comp_df = pd.DataFrame(comp_rows)

        fig_comp = px.line(
            comp_df,
            x="波长(nm)",
            y="相对强度",
            color="光谱曲线",
            title="目标相对光谱、室内自然光贡献及补光后合成光谱比对",
            color_discrete_map={"目标光谱": "#111111", "当前室内自然光": "#1f77b4", "合成后室内光谱": "#2f9e44"},
        )
        fig_comp.update_traces(line=dict(width=3))
        fig_comp.update_layout(height=400, hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
        st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed dataframes for compensation
    st.subheader("通道作用说明与比例清单")
    rec_frame = channel_recommendation_frame(compensation.channel_weights)
    st.dataframe(
        rec_frame[["通道", "峰值波长/nm", "推荐比例", "输出百分比", "光谱作用"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("分波段 RMSE 误差评估 (照明质量量化)")
    band_err = band_error_frame(compensation)
    band_err_long = band_err.melt(id_vars="波段", value_vars=["补偿前RMSE", "补偿后RMSE"], var_name="阶段", value_name="RMSE")
    
    c_left, c_right = st.columns([1.1, 0.9])
    with c_left:
        st.dataframe(band_err, use_container_width=True, hide_index=True)
    with c_right:
        fig_band = px.bar(
            band_err_long,
            x="波段",
            y="RMSE",
            color="阶段",
            barmode="group",
            title="各物理波段补光前后的残差对比",
            color_discrete_map={"补偿前RMSE": "#d95f02", "补偿后RMSE": "#1b9e77"},
        )
        fig_band.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="", yaxis_title="RMSE")
        st.plotly_chart(fig_band, use_container_width=True)

    # Premium CIE color halo simulation card
    st.divider()
    st.subheader("CIE 1964 标准色度空间映射及光色对比")
    color_comp = light_color_comparison_frame(compensation)
    
    st.dataframe(color_comp, use_container_width=True, hide_index=True)
    
    st.pyplot(plot_light_halo_comparison(compensation), clear_figure=True)
