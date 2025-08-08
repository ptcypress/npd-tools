import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("angle_decay.csv")
df["Date"] = pd.to_datetime(df["Date"] + "-2024")  # Add year if missing
df["Day"] = (df["Date"] - df["Date"].min()).dt.days

x_data = df["Day"].values
y_data = df["Angle"].values

# --- Define Exponential Decay Model ---
def decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Fit Model ---
popt, pcov = curve_fit(decay_model, x_data, y_data, p0=(3, 0.05, 19))
a, b, c = popt

# --- Prediction and Derivative ---
x_pred = np.linspace(0, 100, 200)
y_pred = decay_model(x_pred, *popt)
dy_dx = -a * b * np.exp(-b * x_pred)

# --- Find Stabilization Point ---
threshold = st.sidebar.slider("Stabilization threshold (°/day)", 0.001, 0.05, 0.01)
stable_idx = np.argmax(dy_dx < threshold)
stable_day = x_pred[stable_idx]
stable_date = df["Date"].min() + pd.Timedelta(days=int(stable_day))

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(df["Date"], y_data, 'o', label="Observed", color="blue")
ax1.plot(df["Date"].min() + pd.to_timedelta(x_pred, unit='D'), y_pred, '-', label="Exponential Fit", color="black")
ax1.axvline(stable_date, color='red', linestyle='--', label=f'Stabilizes ~{stable_date.date()}')
ax1.set_ylabel("Angle (°)")
ax1.set_xlabel("Date")
ax1.legend()
ax1.grid(True)

st.title("Angle Stabilization Over Time")
st.pyplot(fig)

# --- Optional Derivative Plot ---
with st.expander("Show Rate of Change (dy/dx)"):
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(df["Date"].min() + pd.to_timedelta(x_pred, unit='D'), dy_dx, color="green")
    ax2.axhline(threshold, color="red", linestyle="--")
    ax2.set_ylabel("Rate of Change (°/day)")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    st.pyplot(fig2)

# --- Optional Table ---
with st.expander("Show Data"):
    st.dataframe(df)
