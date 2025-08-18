import streamlit as st
import pandas as pd
import plotly.express as px

# ---- Page config ----
st.set_page_config(page_title="Velocity Boxplot", layout="wide")
st.title("Velocity Boxplot — Simple")

CSV = "data/velocity_data.csv"
try:
    df = pd.read_csv(CSV)
except FileNotFoundError:
    st.error(f"File not found: {CSV}")
    st.stop()
except Exception as e:
    st.error(f"Couldn't read {CSV}: {e}")
    st.stop()

# ---- Assume common headers; fall back gracefully ----
# Preferred columns
group_col = "Brush" if "Brush" in df.columns else None
value_col = "Velocity" if "Velocity" in df.columns else None

# If missing, pick first categorical and first numeric
if group_col is None:
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    group_col = cat_cols[0] if cat_cols else df.columns[0]
if value_col is None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.error("No numeric column found for values.")
        st.stop()
    value_col = num_cols[0]

# Drop NA and compute order by median for a clean, intuitive ranking
clean = df[[group_col, value_col]].dropna()
order = (
    clean.groupby(group_col)[value_col]
    .median()
    .sort_values()
    .index.tolist()
)

# ---- Minimal boxplot (no jitter, no extra adornments) ----
fig = px.box(
    clean,
    x=group_col,
    y=value_col,
    category_orders={group_col: order},
    points=False,            # keep it clean
    template="plotly_white",
)

fig.update_layout(
    height=480,
    margin=dict(l=40, r=20, t=20, b=40),
    showlegend=False,
    xaxis_title=None,
    yaxis_title=value_col,
)

# Subtle grid only
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

st.plotly_chart(fig, use_container_width=True)
