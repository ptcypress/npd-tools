import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as qual
from io import StringIO

# ---------------------------
# Page setup
# ---------------------------
st.title("Decibel Measurements")
st.caption("Clean, consistent visuals with quick metrics and an intuitive time‑series view.")

# ---------------------------
# Sidebar controls (kept minimal to match other pages)
# ---------------------------
st.sidebar.header("Options")
", type=["csv"]) 
show_raw = st.sidebar.checkbox("Show raw trace", value=False)
smooth_sec = st.sidebar.number_input("Smoothing window (seconds)", min_value=0, value=5, step=1, help="0 = no smoothing; uses centered rolling average.")
show_limits = st.sidebar.checkbox("Show OSHA/NIOSH bands", value=True)

# Palette
COLORS = qual.Set2  # aligns with repo preference for plotly qualitative palettes

# ---------------------------
# Helpers
# ---------------------------
def compute_leq(levels_db: np.ndarray) -> float:
    # L_eq = 10 * log10( mean( 10^(L/10) ) )
    if len(levels_db) == 0:
        return float("nan")
    return 10 * np.log10(np.mean(10 ** (levels_db / 10)))

@st.cache_data(show_spinner=False)
def load_example_data(n_seconds: int = 1800, seed: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-08-01 08:00:00", periods=n_seconds, freq="S")
    # Base around ~72 dBA with gentle drift + bursts
    base = 72 + 0.5 * np.sin(np.linspace(0, 12*np.pi, n_seconds))
    bursts = (rng.random(n_seconds) < 0.02) * rng.normal(8, 2, n_seconds)
    noise = rng.normal(0, 1.2, n_seconds)
    dba = base + bursts + noise
    return pd.DataFrame({"timestamp": t, "dBA": dba})

# ---------------------------
# Load data
# ---------------------------
csv_path = "data/decibel_measurements.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"CSV not found at {csv_path}. Please add the file to your repo.")
    st.stop()
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Basic normalization of column names
cols_lower = {c.lower(): c for c in df.columns}
level_col = None
for candidate in ["dba", "db", "level", "spl", "sound_level", "value"]:
    if candidate in cols_lower:
        level_col = cols_lower[candidate]
        break
if level_col is None:
    # If only two columns, assume 2nd is level
    if len(df.columns) >= 2:
        level_col = df.columns[1]
        st.warning(f"Couldn't detect a standard level column; assuming '{level_col}'.")
    else:
        st.error("Couldn't find a decibel column (e.g., dBA, dB, level). Please add one.")
        st.stop()

# Timestamp or index as x
if "timestamp" in cols_lower:
    time_col = cols_lower["timestamp"]
    # Ensure datetime
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        st.warning("Timestamp column couldn't be parsed as datetime; using row index for x‑axis.")
        time_col = None
else:
    time_col = None

# Drop NaNs in the chosen level column
df = df[[c for c in [time_col, level_col] if c is not None]].copy()
df = df.dropna(subset=[level_col])

# Optional smoothing
if smooth_sec and smooth_sec > 0:
    # If timestamps exist and are regular, convert seconds to samples; else use window=seconds directly
    if time_col is not None and len(df) > 2:
        # estimate sampling interval in seconds (median)
        dt = df[time_col].diff().dropna().dt.total_seconds().median()
        if pd.isna(dt) or dt <= 0:
            win = smooth_sec
        else:
            win = max(int(round(smooth_sec / dt)), 1)
    else:
        win = smooth_sec
    smooth = df[level_col].rolling(window=win, center=True, min_periods=max(1, win//3)).mean()
else:
    smooth = None

# Metrics
values = df[level_col].to_numpy()
Leq = compute_leq(values)
Lmax = np.max(values) if len(values) else float("nan")
Lmin = np.min(values) if len(values) else float("nan")
L10 = float(np.percentile(values, 90)) if len(values) else float("nan")  # exceeded 10% of time
L50 = float(np.percentile(values, 50)) if len(values) else float("nan")
L90 = float(np.percentile(values, 10)) if len(values) else float("nan")  # background / exceeded 90%

# Show quick KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("L_eq", f"{Leq:.1f} dBA")
c2.metric("L_max", f"{Lmax:.1f} dBA")
c3.metric("L_10", f"{L10:.1f} dBA")
c4.metric("L_50", f"{L50:.1f} dBA")
c5.metric("L_90", f"{L90:.1f} dBA")

# Build figure
fig = go.Figure()

x = df[time_col] if time_col is not None else df.index

if show_raw:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[level_col],
            mode="lines",
            name="Raw",
            opacity=0.35,
            line=dict(width=1),
            hovertemplate="%{y:.1f} dB<extra>Raw</extra>",
        )
    )

if smooth is not None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=smooth,
            mode="lines",
            name="Smoothed",
            line=dict(width=3),
            hovertemplate="%{y:.1f} dB<extra>Smoothed</extra>",
        )
    )
else:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[level_col],
            mode="lines",
            name="Level",
            line=dict(width=3),
            hovertemplate="%{y:.1f} dB<extra>Level</extra>",
        )
    )

# Color assignment aligned with qualitative palette
for i, tr in enumerate(fig.data):
    tr.line.color = COLORS[i % len(COLORS)]

# Optional limits visualization (bands)
if show_limits and len(df) > 0:
    y_bands = [
        (85, 90, "NIOSH REL (85 dBA) → OSHA PEL (90 dBA)", 0.07),
        (90, 95, "Elevated Risk Zone", 0.05),
    ]
    x0 = x.iloc[0] if hasattr(x, "iloc") else x[0]
    x1 = x.iloc[-1] if hasattr(x, "iloc") else x[-1]
    for (y0, y1, label, alpha) in y_bands:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref="x", yref="y",
                      line=dict(width=0), fillcolor=f"rgba(255,165,0,{alpha})")
        fig.add_annotation(x=x0, y=(y0+y1)/2, xref="x", yref="y", text=label,
                           showarrow=False, xanchor="left", font=dict(size=11))

# Styling to match repo's look: minimal grid, left‑aligned legend, readable ticks
fig.update_layout(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    height=520,
    legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    yaxis=dict(title="dB(A)", showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
)

fig.update_yaxes(range=[max(min(Lmin, 40) - 2, 0), max(Lmax + 5, 60)])
fig.update_layout(title_text="Time‑Series of Sound Level")

st.plotly_chart(fig, use_container_width=True)

# Download options
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
st.download_button("Download cleaned data (CSV)", data=csv_buf.getvalue(), file_name="decibel_cleaned.csv", mime="text/csv")

# Notes section (short, non‑technical)
with st.expander("Notes"):
    st.markdown(
        """
        **L_eq** is the energy‑equivalent continuous sound level over the displayed period. Percentiles like **L_10** and **L_90**
        indicate the levels exceeded 10% and 90% of the time, respectively (L_90 ~ background). The shaded bands are common
        reference zones (NIOSH/OSHA). This page reads from `data/decibel_measurements.csv` with a `dBA` (or `dB`/`level`) column and an optional `timestamp` column.
        """
    )
