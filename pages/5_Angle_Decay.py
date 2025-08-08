import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load Data ---
st.title("Angle Stabilization Over Time")
csv_path = "data/angle_decay.csv"  # Replace with your actual file name
df = pd.read_csv(csv_path)

# Add year to dates if missing and convert to datetime
df["Date"] = pd.to_datetime(df["Date"] + "-2024", format="%d-%b-%Y")

# Convert dates to numeric days since start
df["Day"] = (df["Date"] - df["Date"].min()).dt.days
x_data = df["Day"].values
y_data = df["Angle"].values

# --- Define exponential decay model ---
def decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Fit model to data ---
popt, pcov = curve_fit(decay_model, x_data, y_data, p0=(3, 0.05, 19))
a, b, c = popt

# --- Prediction curve and rate of change ---
x_pred = np.linspace(0, 100, 300)
y_pred = decay_model(x_pred, *popt)
dy_dx = -a * b * np.exp(-b * x_pred)

# --- Choose threshold and calculate stabilization day ---
threshold = st.sidebar.slider("Stabilization threshold (°/day)", 0.001, 0.05, 0.01)
stable_idx = np.argmax(dy_dx < threshold)
stable_day = x_pred[stable_idx]
stable_date = df["Date"].min() + pd.Timedelta(days=int(stable_day))
stable_angle = decay_model(stable_day, *popt)

# --- Main Plot ---
fig, ax1 = plt.subplots(figsize=(10, 5))

# Observed data
ax1.plot(df["Date"], y_data, 'o', label="Observed", color="blue")

# Fitted model curve
model_dates = df["Date"].min() + pd.to_timedelta(x_pred, unit='D')
ax1.plot(model_dates, y_pred, '-', label="Exponential Fit", color="black")

# Reference lines at stabilization
ax1.axvline(stable_date, color='red', linestyle='--', linewidth=1)
ax1.axhline(stable_angle, color='red', linestyle='--', linewidth=1)

# Annotation at intersection
ax1.annotate(
    f"Stabilizes ≈ {stable_angle:.2f}°\nOn ≈ {stable_date.date()}",
    xy=(stable_date, stable_angle),
    xytext=(10, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
    color="red"
)

# Axis labels and formatting
ax1.set_ylabel("Angle (°)")
ax1.set_xlabel("Date")
ax1.grid(True)
ax1.legend()

# Show main plot
st.pyplot(fig)

# --- Optional Derivative Plot ---
with st.expander("Show Rate of Change (dy/dx)"):
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(model_dates, dy_dx, color="green", label="dy/dx")
    ax2.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax2.set_ylabel("Rate of Change (°/day)")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# --- Optional Data Table ---
with st.expander("Show Raw Data"):
    st.dataframe(df[["Date", "Angle"]])

st.markdown(f"### 📉 Predicted Long-Term Minimum Angle: **{c:.3f}°**")

