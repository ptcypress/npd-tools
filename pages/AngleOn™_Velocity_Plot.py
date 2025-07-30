import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Load dataset from the user-provided data
data = {
    "Weight (lbs)": [0.167]*5 + [0.246]*5 + [1.875]*5 + [3.75]*5 + [5.625]*5 + [7.5]*5 + [9.375]*5 + [11.25]*5 + [13.125]*5 + [15]*5 + [18.75]*5,
    "Time (sec)": [6.64, 6.64, 6.65, 6.67, 6.67, 7.44, 7.46, 7.46, 7.48, 7.48, 9.63, 9.69, 9.88, 9.83, 9.81,
                   11.11, 10.96, 11.08, 11.07, 11.02, 12.32, 12.22, 12.83, 12.67, 12.62, 13.90, 13.73, 14.14, 13.65, 13.73,
                   15.22, 15.01, 14.90, 14.98, 15.12, 16.48, 16.68, 16.41, 16.53, 16.65, 19.06, 18.90, 19.19, 19.07, 18.95,
                   23.16, 22.78, 22.93, 22.90, 23.05, 36.90, 35.68, 35.25, 36.08, 36.12],
    "Velocity (in/sec)": [4.56, 4.56, 4.55, 4.54, 4.54, 4.57, 4.56, 4.56, 4.55, 4.55, 3.17, 3.15, 3.09, 3.10, 3.11,
                          2.75, 2.78, 2.75, 2.76, 2.77, 2.48, 2.50, 2.38, 2.41, 2.42, 2.19, 2.22, 2.16, 2.23, 2.22,
                          2.00, 2.03, 2.05, 2.04, 2.02, 1.85, 1.83, 1.86, 1.85, 1.83, 1.60, 1.61, 1.59, 1.60, 1.61,
                          1.32, 1.34, 1.33, 1.33, 1.32, 0.83, 0.85, 0.87, 0.85, 0.84],
    "Pressure (lbs/in²)": [0.05]*5 + [0.17]*5 + [0.25]*5 + [0.50]*5 + [0.75]*5 + [1.00]*5 + [1.25]*5 + [1.50]*5 + [1.75]*5 + [2.00]*5 + [2.50]*5
}
df = pd.DataFrame(data)

# Streamlit app
st.title("Velocity vs. Pressure for AngleOn™ Brush")
st.write("Use the sliders below to adjust reference lines on the chart.")

# Sliders for reference lines
x_ref = st.slider("Reference Pressure (lbs/in²)", min_value=0.0, max_value=3.0, value=0.17, step=0.01)
y_ref = st.slider("Reference Velocity (in/sec)", min_value=0.0, max_value=5.0, value=2.5, step=0.01)

# Plotting
fig, ax = plt.subplots()
ax.scatter(df["Pressure (lbs/in²)"], df["Velocity (in/sec)"], alpha=0.7, label="Data")
ax.axvline(x=x_ref, color='red', linestyle='--', label=f'Pressure = {x_ref}')
ax.axhline(y=y_ref, color='blue', linestyle='--', label=f'Velocity = {y_ref}')
ax.set_xlabel("Pressure (lbs/in²)")
ax.set_ylabel("Velocity (in/sec)")
ax.set_title("Velocity vs. Pressure with Adjustable Reference Lines")
ax.legend()
ax.grid(True)

st.pyplot(fig)
