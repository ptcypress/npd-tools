import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.colors import qualitative

# Set up Streamlit page
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Velocity vs Pressure — AngleOn™")

st.write("""
Use the sliders in the sidebar to adjust **horizontal (velocity)** and **vertical (pressure)** reference lines.
""")

# Sample data
data = pd.DataFrame({
    "Pressure": [
        0.05, 0.05, 0.05, 0.05, 0.05,
        0.17, 0.17, 0.17, 0.17, 0.17,
        0.25, 0.25, 0.25, 0.25, 0.25,
        0.50, 0.50, 0.50, 0.50, 0.50,
        0.75, 0.75, 0.75, 0.75, 0.75,
        1.00, 1.00, 1.00, 1.00, 1.00,
        1.25, 1.25, 1.25, 1.25, 1.25,
        1.50, 1.50, 1.50, 1.50, 1.50,
        1.75, 1.75, 1.75, 1.75, 1.75,
        2.00, 2.00, 2.00, 2.00, 2.00,
        2.50, 2.50, 2.50, 2.50, 2.50
    ],
    "Velocity": [
        4.56, 4.56, 4.55, 4.54, 4.54,
        4.57, 4.56, 4.56, 4.55, 4.55,
        3.17, 3.15, 3.09, 3.10, 3.11,
        2.75, 2.78, 2.75, 2.76, 2.77,
        2.48, 2.50, 2.38, 2.41, 2.42,
        2.19, 2.22, 2.16, 2.23, 2.22,
        2.00, 2.03, 2.05, 2.04, 2.02,
        1.85, 1.83, 1.86, 1.85, 1.83,
        1.60, 1.61, 1.59, 1.60, 1.61,
        1.32, 1.34, 1.33, 1.33, 1.32,
        0.83, 0.85, 0.87, 0.85, 0.84
    ]
})

# Sidebar sliders
st.sidebar.header("Reference Lines")
ref_velocity = st.sidebar.slider("Velocity (in/sec)", min_value=0.5, max_value=5.0, value=2.5, step=0.01)
ref_pressure = st.sidebar.slider("Pressure (lbs/in²)", min_value=0.0, max_value=3.0, value=1.0, step=0.01)

# Create scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["Pressure"],
    y=data["Velocity"],
    mode="markers",
    marker=dict(size=8, color=qualitative.Plotly[0]),
    name="Data"
))

# Add reference lines
fig.add_shape(
    type="line", x0=data["Pressure"].min(), x1=data["Pressure"].max(),
    y0=ref_velocity, y1=ref_velocity,
    line=dict(color="gray", dash="dash"), name="Velocity Ref"
)

fig.add_shape(
    type="line", x0=ref_pressure, x1=ref_pressure,
    y0=data["Velocity"].min(), y1=data["Velocity"].max(),
    line=dict(color="gray", dash="dash"), name="Pressure Ref"
)

# Format plot
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    title="Velocity vs Pressure",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

# Show plot
st.plotly_chart(fig, use_container_width=True)
