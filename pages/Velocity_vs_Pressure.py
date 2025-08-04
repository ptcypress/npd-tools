import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from plotly.colors import qualitative

# Load data
df = pd.read_csv("data/velocity_data.csv")

# Ensure necessary columns
required_cols = {"Brush", "Pressure", "Velocity"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain columns: Brush, Pressure, Velocity")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Velocity vs Pressure — Cubic Polynomial Fit")

# Plot setup
fig = go.Figure()
colors = qualitative.Plotly
brushes = ["AngleOn", "Competitor"]

for i, brush in enumerate(brushes):
    subset = df[df["Brush"] == brush]
    x = subset["Pressure"].values
    y = subset["Velocity"].values

    # Fit cubic polynomial
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = model.predict(poly.transform(x_fit.reshape(-1, 1)))

    # Raw data
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name=f'{brush} Data',
        marker=dict(color=colors[i], size=8),
        hovertemplate='Pressure: %{x}<br>Velocity: %{y}<extra></extra>'
    ))

    # Polynomial fit
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        name=f'{brush} Cubic Fit',
        line=dict(color=colors[i], width=2),
        hoverinfo='skip'
    ))

# Final layout
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    title="Cubic Polynomial Fit: AngleOn vs Competitor",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79)
)

st.plotly_chart(fig, use_container_width=True)
