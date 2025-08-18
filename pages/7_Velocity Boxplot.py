import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Velocity Boxplot", layout="wide")
st.title("Velocity Boxplot")

# ---- Load ----
CSV = "data/velocity_data.csv"
df = pd.read_csv(CSV)

# ---- Basic column picks (tune to your headers) ----
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.columns if df[c].dtype == "object"]

# Sensible defaults from your datasets
group_col = st.sidebar.selectbox("Group by", options=cat_cols, index=cat_cols.index("Brush") if "Brush" in cat_cols else 0)
value_col = st.sidebar.selectbox("Value", options=num_cols, index=num_cols.index("Velocity") if "Velocity" in num_cols else 0)

# Optional filter by pressure (if present)
if "Pressure" in df.columns and pd.api.types.is_numeric_dtype(df["Pressure"]):
    pmin, pmax = float(df["Pressure"].min()), float(df["Pressure"].max())
    p_rng = st.sidebar.slider("Pressure filter", pmin, pmax, (pmin, pmax))
    df = df[(df["Pressure"] >= p_rng[0]) & (df["Pressure"] <= p_rng[1])]

# ---- Prepare groups (sort by median) ----
summary = df.groupby(group_col)[value_col].agg(["median","count"]).sort_values("median")
order = summary.index.tolist()

# ---- Controls ----
show_points = st.sidebar.checkbox("Show data points", True)
target = st.sidebar.number_input("Target velocity (optional)", value=0.0, step=0.1, format="%.1f")
spec_low = st.sidebar.number_input("Spec low (optional)", value=0.0, step=0.1, format="%.1f")
spec_high = st.sidebar.number_input("Spec high (optional)", value=0.0, step=0.1, format="%.1f")

# ---- Figure ----
fig = go.Figure()

# Boxes
for grp in order:
    vals = df.loc[df[group_col] == grp, value_col].dropna()
    fig.add_trace(go.Box(
        y=vals,
        x=[grp]*len(vals),
        name=f"{grp}",
        boxpoints="all" if show_points else "outliers",
        jitter=0.35 if show_points else 0.0,
        pointpos=0.0,
        marker=dict(size=5, opacity=0.5),
        notched=True,
        boxmean="sd",  # show mean and SD whisker mark
        hovertemplate=f"{group_col}: {grp}<br>{value_col}: "+"%{y:.3f}<extra></extra>"
    ))
    # N label above each box
    fig.add_annotation(
        x=grp, y=np.nanmax(vals) if len(vals) else 0, yshift=16,
        text=f"N={len(vals)}", showarrow=False, font=dict(size=11)
    )

# Target/spec lines
ymin = float(df[value_col].min()) if len(df) else 0
ymax = float(df[value_col].max()) if len(df) else 1

if target > 0:
    fig.add_hline(y=target, line_dash="dash", line_width=2, annotation_text="", annotation_position="top left",
                  name="Target")

if spec_low > 0 or spec_high > 0:
    lo = spec_low if spec_low > 0 else ymin
    hi = spec_high if spec_high > 0 else ymax
    fig.add_shape(type="rect", xref="paper", x0=0, x1=1, y0=lo, y1=hi,
                  line=dict(width=0), fillcolor="rgba(0,0,0,0.05)")
    # add legend-like traces for band and/or lines if you want them in legend:
    if spec_low > 0:
        fig.add_trace(go.Scatter(x=[order[0], order[-1]], y=[spec_low, spec_low],
                                 mode="lines", name="Spec Low", line=dict(dash="dot")))
    if spec_high > 0:
        fig.add_trace(go.Scatter(x=[order[0], order[-1]], y=[spec_high, spec_high],
                                 mode="lines", name="Spec High", line=dict(dash="dot")))

fig.update_layout(
    template="plotly_white",
    height=520,
    margin=dict(l=40, r=20, t=20, b=40),
    showlegend=True,
    legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
    yaxis_title=value_col,
    xaxis_title=None,
)

st.plotly_chart(fig, use_container_width=True)

# Optional: quick table of medians & IQR
st.dataframe(
    df.groupby(group_col)[value_col].describe()[["count","mean","50%","std","min","25%","75%","max"]]
    .rename(columns={"50%":"median", "25%":"q1", "75%":"q3"})
    .loc[order]
)
