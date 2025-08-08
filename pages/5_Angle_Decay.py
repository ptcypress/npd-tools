import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# --- App title ---
st.title("Angle Stabilization Over Time")

# --- Load Data ---
csv_path = "data/angle_decay.csv"  # Adjust if needed
df = pd.read_csv(csv_path)

# Add year and convert date
df["Date"] = pd.to_datetime(df["Date"] + "-2024", format="%d-%b-%Y")
df["Day"] = (df["Date"] - df["Date"].min()).dt.days

# Prepare x/y
x_data = df["Day"].values
y_data = df["Angle"].values

# --- Define exponential decay model ---
def decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# --- Fit model ---
popt, pcov = curve_fit(decay_model, x_data, y_data, p0=(3, 0.05, 19))
a, b, c = popt  # c is the long-term minimum
min_angle = c

# --- Generate smooth prediction curve ---
x_pred = np.linspace(0, 100, 300)
y_pred = decay_model(x_pred, *popt)
model_dates = df["Date"].min() + pd.to_timedelta(x_pred, unit='D')

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

# Plot observed data and model
ax.plot(df["Date"], y_data, 'o', label="Observed", color="blue")
ax.plot(model_dates, y_pred, '-', label="Exponential Fit", color="black")

# Horizontal line for predicted long-term minimum
ax.axhline(min_angle, color='red', linestyle='--', linewidth=1)

# Annotate predicted minimum on right side of plot
rightmost_date = df["Date"].max() + pd.Timedelta(days=5)
ax.annotate(
    f"Predicted Long-Term\nMinimum ≈ {min_angle:.2f}°",
    xy=(rightmost_date, min_angle),
    xytext=(0, 0),
    textcoords="offset points",
    fontsize=10,
    color="red",
    ha='left',
    va='bottom'
)

# Final formatting
ax.set_ylabel("Angle (°)")
ax.set_xlabel("Date")
ax.grid(True)
ax.legend()

# --- Display plot and summary ---
st.pyplot(fig)
st.markdown(f"### 📉 Predicted Long-Term Minimum Angle: **{min_angle:.3f}°**")

# --- Optional raw data view ---
with st.expander("Show Raw Data"):
    st.dataframe(df[["Date", "Angle"]])
