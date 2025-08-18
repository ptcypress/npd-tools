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

# --- Pick columns (simple rules) ---
cols_lower = {c.lower(): c for c in df.columns}
# y/level column
level_col = None
for c in ["dba", "db", "level", "spl", "value"]:
    if c in cols_lower:
        level_col = cols_lower[c]
        break
if level_col is None:
    if len(df.columns) >= 2:
        level_col = df.columns[1]
    else:
        st.error("Need a decibel column (e.g., dBA, dB, level).")
        st.stop()

# x/time column
if "timestamp" in cols_lower:
    time_col = cols_lower["timestamp"]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        x = df[time_col]
    except Exception:
        x = df.index
else:
    x = df.index

# --- Simple line chart ---
fig = go.Figure(
    go.Scatter(x=x, y=df[level_col], mode="lines", name=level_col)
)
fig.update_layout(
    template="plotly_white",
    height=440,
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(orientation="h", x=0, y=1.1, bgcolor="rgba(0,0,0,0)"),
    xaxis_title=None,
    yaxis_title="dB(A)" if level_col.lower()=="dba" else "dB",
)

st.plotly_chart(fig, use_container_width=True)
