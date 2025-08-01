import streamlit as st
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from plotly.colors import qualitative

# Data
pressure = np.array([
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
])
velocity = np.array([
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
])

# Polynomial regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(pressure.reshape(-1, 1))
model = LinearRegression()
model.fit(X_poly, velocity)
pressure_fit = np.linspace(0, 2.6, 300)
velocity_fit = model.predict(poly.transform(pressure_fit.reshape(-1, 1)))

# Streamlit app
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Velocity vs Pressure — Polynomial Fit")

# Plot
fig = go.Figure()

# Add measured data points
fig.add_trace(go.Scatter(
    x=pressure, y=velocity,
    mode='markers',
    name='Measured Data',
    marker=dict(color=qualitative.Plotly[0], size=8),
    hovertemplate='Pressure: %{x}<br>Velocity: %{y}<extra></extra>'
))

# Add regression line
fig.add_trace(go.Scatter(
    x=pressure_fit, y=velocity_fit,
    mode='lines',
    name='Polynomial Fit (Degree 3)',
    line=dict(color='red', width=2),
    hoverinfo='skip'
))

# Update layout
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    title="Velocity vs Pressure with Polynomial Fit",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79)
)

st.plotly_chart(fig, use_container_width=True)
