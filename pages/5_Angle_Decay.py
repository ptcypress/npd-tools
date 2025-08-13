import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# Page config (match other pages)
# ---------------------------
st.set_page_config(
    page_title="Angle Decay Model",
    page_icon="📉",
    layout="wide",
)

# Optional: minimal style tweaks to match look/feel
st.markdown(
    """
    <style>
    /* Tighten top padding, match other pages spacing */
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities
# ---------------------------

def _to_days(series: pd.Series) -> np.ndarray:
    """Convert a datetime series to day offsets from the first timestamp (t=0 at first row)."""
    t0 = series.min()
    return (series - t0).dt.total_seconds().to_numpy() / 86400.0


def _exp_decay(t, A, k, C):
    """Exponential decay: y = A * exp(-k*t) + C"""
    return A * np.exp(-k * t) + C


def fit_exp_decay(df: pd.DataFrame, date_col: str, angle_col: str):
    """Robustly fit exponential decay returning params and fitted curve.
    Falls back to a simple grid search if scipy isn't available.
    """
    t = _to_days(df[date_col])
    y = df[angle_col].to_numpy()

    # Initial guesses
    A0 = max(y) - min(y)
    k0 = 0.1 if len(df) < 8 else 1.0 / (t[-1] - t[0] + 1e-9)
    C0 = min(y)

    try:
        from scipy.optimize import curve_fit

        popt, pcov = curve_fit(
            _exp_decay, t, y, p0=[A0, k0, C0], maxfev=20000, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        )
        A, k, C = popt
        yhat = _exp_decay(t, A, k, C)
        return (A, k, C), yhat
    except Exception:
        # Fallback: simple coarse-to-fine grid on k and C, solve A in closed-form by least squares
        k_grid = np.geomspace(1e-3, 5.0, 50)
        C_grid = np.linspace(min(y) - 2.0, max(y), 50)
        best = None
        for k in k_grid:
            e = np.exp(-k * t)
            X = np.column_stack([e, np.ones_like(e)])  # [A, C]
            # We'll scan C explicitly and fit A each time
            for C in C_grid:
                A = np.linalg.lstsq(e.reshape(-1, 1), (y - C), rcond=None)[0][0]
                y_pred = A * e + C
                rss = float(np.sum((y - y_pred) ** 2))
                if (best is None) or (rss < best[0]):
                    best = (rss, A, k, C, y_pred)
        _, A, k, C, yhat = best
        return (float(A), float(k), float(C)), yhat


def stabilization_time(A: float, k: float, C: float, y0: float, eps_deg: float) -> float:
    """Return t* (in days from t0) when |y(t) - C| <= eps_deg for the first time.
    If already stable, returns 0.
    """
    # y(t) = A*exp(-k t) + C
    # |A| * exp(-k t) <= eps => exp(-k t) <= eps / |A| => t >= (1/k) * ln(|A|/eps)
    A_eff = abs(y0 - C) if A == 0 else abs(A)
    if A_eff <= eps_deg:
        return 0.0
    if k <= 0:
        return np.inf
    return float(np.log(A_eff / eps_deg) / k)


# ---------------------------
# Sidebar controls
# ---------------------------
st.title("Angle Decay (Exponential) 📉")
st.caption("Fit an exponential decay A·e^{-k t} + C and estimate when the angle stabilizes.")

with st.sidebar:
    st.header("Inputs")
    st.markdown(
        """
        **Data source:** `data/angle_decay.csv`
        
        **Required columns**
        - `Date` (any parsable date format)
        - `Angle` (float)
        Optional: `St Dev` for reference/error bars
        """
    )

    default_eps = st.number_input("Stabilization band ± (deg)", min_value=0.01, max_value=5.0, value=0.25, step=0.01)
    show_points = st.checkbox("Show data points", value=True)
    show_residuals = st.checkbox("Show residuals", value=False)
    ref_year = st.number_input("Assumed year (for dates like '31-Jul')", min_value=2000, max_value=2100, value=datetime.today().year, step=1)

# ---------------------------
# Data loading
# ---------------------------
DATA_PATH = "data/angle_decay.csv"
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Couldn't read {DATA_PATH}: {e}")
    st.stop()

# Normalize / validate columns (strip header whitespace, case-insensitive)
norm_map = {str(c).strip().lower(): c for c in df.columns}
if "date" not in norm_map or "angle" not in norm_map:
    st.error("CSV must include `Date` and `Angle` columns.")
    st.stop()

DATE_COL = norm_map["date"]
ANGLE_COL = norm_map["angle"]

# Optional std dev column (several possible header spellings)
STD_COL = None
for key in ["st dev", "stdev", "std dev", "std_dev", "st deviation", "st_deviation", "sd"]:
    if key in norm_map:
        STD_COL = norm_map[key]
        break

# Parse dates
_date_strategy = "unknown"
if not np.issubdtype(df[DATE_COL].dtype, np.datetime64):
    raw_date_col = df[DATE_COL].copy()
    df[DATE_COL] = pd.to_datetime(raw_date_col, errors="coerce")
    _date_strategy = "direct"
    if df[DATE_COL].isna().all():
        s = raw_date_col.astype(str).str.strip()
        try:
            # e.g., "31-Jul" -> "31-Jul 2025" (assume dayfirst for dd-Mon)
            df[DATE_COL] = pd.to_datetime(s + f" {int(ref_year)}", errors="coerce", dayfirst=True)
            _date_strategy = "append_year_space_dayfirst"
        except Exception:
            pass
        if df[DATE_COL].isna().all():
            try:
                # e.g., "7/31" -> "7/31/2025" (US month/day)
                df[DATE_COL] = pd.to_datetime(s + f"/{int(ref_year)}", errors="coerce")
                _date_strategy = "append_slash_year"
            except Exception:
                pass
        if df[DATE_COL].isna().all():
            _date_strategy = "coerce_failed"

# Coerce Angle to numeric (strip non-numeric chars like degree symbols first)
if df[ANGLE_COL].dtype == object:
    df[ANGLE_COL] = (
        df[ANGLE_COL]
        .astype(str)
        .str.strip()
        .str.replace(r"[^0-9+\-\.eE]", "", regex=True)
    )
df[ANGLE_COL] = pd.to_numeric(df[ANGLE_COL], errors="coerce")

# Diagnostics before filtering
_total_rows = len(df)
_invalid_date = int(df[DATE_COL].isna().sum())
_invalid_angle = int(df[ANGLE_COL].isna().sum())

# Drop NaNs in core columns only (keep optional St Dev even if missing), then sort
core_cols = [DATE_COL, ANGLE_COL]
opt_cols = [STD_COL] if STD_COL else []
_df = (
    df[core_cols + opt_cols]
    .dropna(subset=core_cols)
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

# Guard: need at least 2 valid rows to fit the model
if len(_df) < 2:
    st.error(
        f"Need at least 2 rows with valid `Date` and `Angle` to fit the model. "
        f"Rows read: {_total_rows}, invalid dates: {_invalid_date}, non-numeric angles: {_invalid_angle}, "
        f"valid rows: {len(_df)}."
    )
    with st.expander("Preview & parsing diagnostics"):
        st.write("First 20 rows after parsing (before dropping invalids):")
        st.dataframe(df.head(20), use_container_width=True)
        st.write({
            "rows_read": _total_rows,
            "invalid_dates": _invalid_date,
            "non_numeric_angles": _invalid_angle,
            "valid_rows": len(_df),
            "date_parse_strategy": _date_strategy,
        })
    st.stop()

# ---------------------------
# Modeling
# ---------------------------
params, yhat = fit_exp_decay(_df, DATE_COL, ANGLE_COL)
A, k, C = params

# Compute stabilization date
_t = _to_days(_df[DATE_COL])
y0 = _df[ANGLE_COL].iloc[0]
t_star_days = stabilization_time(A, k, C, y0, default_eps)
first_date = _df[DATE_COL].min()
stabilizes_on = None if not np.isfinite(t_star_days) else (first_date + pd.to_timedelta(t_star_days, unit="D"))

# Build dense curve for plotting
if len(_df) > 1:
    t_dense = np.linspace(_t.min(), _t.max() + max(7.0, 0.2 * (_t.max() - _t.min() + 1e-9)), 400)
else:
    t_dense = np.linspace(0, 14, 200)
T0 = _df[DATE_COL].min()
dates_dense = pd.to_datetime(T0) + pd.to_timedelta(t_dense, unit="D")
y_dense = _exp_decay(t_dense, A, k, C)

# ---------------------------
# Main layout
# ---------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    fig = go.Figure()

    if show_points:
        fig.add_trace(
            go.Scatter(
                x=_df[DATE_COL], y=_df[ANGLE_COL],
                mode="markers",
                name="Observed",
                marker=dict(size=8),
            )
        )

    # Fitted curve
    fig.add_trace(
        go.Scatter(
            x=dates_dense, y=y_dense,
            mode="lines",
            name="Fit: A·e^{-k t} + C",
        )
    )

    # Asymptote C
    fig.add_hline(y=C, line_dash="dot", annotation_text=f"Asymptote C = {C:.3f}°", annotation_position="top left")

    # Stabilization band
    fig.add_hline(y=C + default_eps, line_dash="dash", annotation_text=f"+{default_eps:.2f}°", annotation_position="top right")
    fig.add_hline(y=C - default_eps, line_dash="dash", annotation_text=f"-{default_eps:.2f}°", annotation_position="bottom right")

    # Stabilization date marker
    if stabilizes_on is not None and stabilizes_on >= _df[DATE_COL].min():
        fig.add_vline(x=stabilizes_on, line_dash="dot", annotation_text=f"Stabilizes ≈ {stabilizes_on.date()}", annotation_position="top")

    fig.update_layout(
        template="plotly_white",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Angle (deg)",
        colorway=qualitative.Set2,  # match palette used elsewhere
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Model summary")
    st.metric("A (initial drop)", f"{A:.4f}")
    st.metric("k (rate per day)", f"{k:.4f}")
    st.metric("C (long‑term angle)", f"{C:.4f}°")
    if stabilizes_on is None or not np.isfinite(t_star_days):
        st.info("Model does not stabilize with current parameters (k≤0).")
    else:
        st.metric("Stabilizes by", stabilizes_on.strftime("%Y-%m-%d"))
        st.caption(f"Within ±{default_eps:.2f}° of C by ~{t_star_days:.1f} days after first measurement.")

    with st.expander("Data preview"):
        st.dataframe(_df, use_container_width=True, height=240)

# ---------------------------
# Residuals (optional)
# ---------------------------
if show_residuals:
    import plotly.graph_objects as go
    res_fig = go.Figure()
    residuals = _df[ANGLE_COL] - _exp_decay(_t, A, k, C)
    res_fig.add_trace(go.Scatter(x=_df[DATE_COL], y=residuals, mode="markers+lines", name="Residuals"))
    res_fig.update_layout(template="plotly_white", height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Date", yaxis_title="Residual (deg)")
    st.plotly_chart(res_fig, use_container_width=True)

# ---------------------------
# Notes
# ---------------------------
st.caption("Formatting aligned with other pages: set_page_config(layout='wide'), controls in sidebar, Plotly charts with 'plotly_white' template and qualitative.Set2 palette, container_width charts, and consistent spacing.")
