import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve

# Load durability data
csv_path = "data/durability_data.csv"
df = pd.read_csv(csv_path)

# Ensure proper column names are used
df = df.dropna(subset=['Belt Speed', 'Rate of Material Loss'])
x = df['Belt Speed'].values.reshape(-1, 1)
y = df['Rate of Material Loss'].values

# Polynomial regression (cubic)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)
model = LinearRegression().fit(X_poly, y)

# Predict over a range of belt speeds
x_range = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))

# Streamlit layout
st.set_page_config(page_title="Durability vs Belt Speed", layout="wide")
st.title("Durability Characterization")
st.subheader("Rate of Material Loss vs Belt Speed")

# Plot
fig = go.Figure()

# Add regression line
fig.add_trace(go.Scatter(
    x=x_range.flatten(), y=y_pred,
    mode='lines',
    name='Cubic Fit',
    line=dict(color='green', width=3)
))

# Update layout
fig.update_layout(
    xaxis_title="Belt Speed (units?)",
    yaxis_title="Rate of Material Loss (units?)",
    height=650,
    hovermode='x',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    xaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=1
    ),
    yaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=1
    ),
    hoverlabel=dict(bgcolor="rgba(0,0,0,0)", font_size=12, font_family="Arial")
)

# Caption
st.markdown("""
This chart visualizes how the **rate of material loss** varies with **belt speed**. The cubic regression line illustrates the overall trend, helping to identify potential thresholds where wear accelerates.
""")

st.plotly_chart(fig, use_container_width=True)
