import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve

# Load durability data
csv_path = "data/durability_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Extract data for each material
x = df['Belt Speed'].values.reshape(-1, 1)
y_angleon = df['AngleOn'].values
y_competitor = df['Competitor'].values

# Fit cubic polynomial models
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)
model_angleon = LinearRegression().fit(X_poly, y_angleon)
model_competitor = LinearRegression().fit(X_poly, y_competitor)

# Define functions for integration
def f(x_val):
    x_val = float(np.squeeze(x_val))
    return model_angleon.predict(poly.transform([[x_val]]))[0]

def g(x_val):
    x_val = float(np.squeeze(x_val))
    return model_competitor.predict(poly.transform([[x_val]]))[0]

# Find intersection point
x_intersect = fsolve(lambda x_val: f(x_val) - g(x_val), x0=1.0)[0]

# Calculate area between curves from x_intersect to 50
area, _ = quad(lambda x_val: abs(f(x_val) - g(x_val)), x_intersect, 50)

# Predict over a range for plotting
x_range = np.linspace(x.min(), 50, 300).reshape(-1, 1)
y_pred_angleon = model_angleon.predict(poly.transform(x_range))
y_pred_competitor = model_competitor.predict(poly.transform(x_range))

# Streamlit layout
st.set_page_config(page_title="Durability vs Belt Speed", layout="wide")
st.title("Durability Characterization")
st.subheader("Rate of Material Loss vs Belt Speed")

# Plot
fig = go.Figure()

# AngleOn line
fig.add_trace(go.Scatter(
    x=x_range.flatten(), y=y_pred_angleon,
    mode='lines',
    name='AngleOn',
    line=dict(color='blue', width=3)
))

# Competitor line
fig.add_trace(go.Scatter(
    x=x_range.flatten(), y=y_pred_competitor,
    mode='lines',
    name='Competitor',
    line=dict(color='red', width=3)
))

# Shaded area between curves
fig.add_trace(go.Scatter(
    x=np.concatenate([x_range.flatten(), x_range.flatten()[::-1]]),
    y=np.concatenate([y_pred_angleon, y_pred_competitor[::-1]]),
    fill='toself',
    fillcolor='rgba(100,100,100,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    name='Difference Area'
))

# Update layout
fig.update_layout(
    xaxis_title="Belt Speed",
    yaxis_title="Rate of Material Loss",
    height=650,
    hovermode='x',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    xaxis=dict(
        type='log',
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=1
    ),
    yaxis=dict(
        showspikes=True, spikemode="across", spikesnap="cursor",
        showline=True, spikecolor="lightgray", spikethickness=1
    ),
    hoverlabel=dict(bgcolor="rgba(0,0,0,0)", font_size=12, font_family="Arial")
)

# Annotation
fig.add_annotation(
    x=40,
    y=max(y_pred_angleon.max(), y_pred_competitor.max()),
    text=f"Shaded area = {area:.3f} (units²)",
    showarrow=False,
    font=dict(size=13)
)

# Caption
st.markdown("""
This chart compares the **rate of material loss** across belt speeds for two materials.
The shaded area between the curves, from their intersection to a belt speed of 50,
represents the **cumulative durability advantage** of one material over the other.
""")

st.plotly_chart(fig, use_container_width=True)
