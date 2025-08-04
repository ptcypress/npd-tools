import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from plotly.colors import qualitative
import os

# Load CSV from local data folder
file_path = os.path.join("data", "velocity_data.csv")

if not os.path.exists(file_path):
    st.error(f"CSV file not found at: {file_path}")
    st.stop()

df = pd.read_csv(file_path)

# Validate required columns
required_cols = {"Brush", "Pressure", "Velocity"}
if not required_cols.issubset(df.columns):
    st.error("CSV must contain the following columns: Brush, Pressure, Velocity")
    st.stop()

# Standardize brush names to lowercase
df["Brush"] = df["Brush"].str.strip().str.lower()

# Streamlit page setup
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Velocity vs Pressure — Cubic Polynomial Fit")

# Define target brushes (case-insensitive)
target_brushes = ["AngleOn™", "Competitor"]
colors = qualitative.Plotly
fig = go.Figure()

# Iterate and plot
for i, brush in enumerate(target_brushes):
    subset = df[df["Brush"] == brush]

    if subset.empty:
        st.warning(f"No data found for brush: {brush}")
        continue

    x = subset["Pressure"].values
    y = subset["Velocity"].values

    if len(x) < 4:
        st.warning(f"Not enough data points to fit a cubic model for brush: {brush}")
        continue

    # Fit cubic polynomial
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, y)
    x_fit = np.linspace(x.min(), x.max(), 300)
    y_fit = model.predict(poly.transform(x_fit.reshape(-1, 1)))

    # Plot raw data
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        name=f'{brush.title()} Data',
        marker=dict(color=colors[i], size=8),
        hovertemplate='Pressure: %{x}<br>Velocity: %{y}<extra></extra>'
    ))

    # Plot fit
    fig.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        name=f'{brush.title()} Cubic Fit',
        line=dict(color=colors[i], width=2),
        hoverinfo='skip'
    ))

# Final plot layout
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    title="Cubic Polynomial Fit: AngleOn vs Competitor",
    height=600,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79)
)

st.plotly_chart(fig, use_container_width=True)
