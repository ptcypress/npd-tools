import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Title ---
st.title("Decibel Measurements")

# --- Read CSV ---
CSV_PATH = "data/decibel_measurements.csv"
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"File not found: {CSV_PATH}")
    st.stop()
except Exception as e:
    st.error(f"Couldn't read {CSV_PATH}: {e}")
    st.stop()

# --- Identify time column (optional) ---
cols_lower = {c.lower(): c for c in df.columns}
time_col = None
if "timestamp" in cols_lower:
    time_col = cols_lower["timestamp"]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        x = df[time_col]
    except Exception:
        x = df.index
else:
    x = df.index

# --- Choose series to plot: all numeric columns except timestamp ---
# Try to coerce likely level columns to numeric if needed
likely_level_names = [c for c in df.columns if any(k in c.lower() for k in ["dba", "dbc", "db", "spl", "level", "value"]) ]
for c in likely_level_names:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = pd.to_numeric(df[c], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if time_col in numeric_cols:
    numeric_cols.remove(time_col)

if not numeric_cols:
    st.error("No numeric decibel columns found to plot. Add columns like dBA, dB, SPL, or Level.")
    st.stop()

# --- Build figure with all numeric series ---
fig = go.Figure()
for col in numeric_cols:
    fig.add_trace(go.Scatter(x=x, y=df[col], mode="lines", name=col))

# OSHA TWA Action Level line at 85 dBA (legend-visible)
if len(df) > 0:
    if hasattr(x, "iloc"):
        x0, x1 = x.iloc[0], x.iloc[-1]
    else:
        x0, x1 = x[0], x[-1]
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[85, 85],
            mode="lines",
            name="OSHA TWA Action Level (85 dBA)",
            line=dict(dash="dash", width=2)
        )
    )

fig.update_layout(
    template="plotly_white",
    height=460,
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis_title=None,
    yaxis_title="dBA",
)

st.plotly_chart(fig, use_container_width=True)
