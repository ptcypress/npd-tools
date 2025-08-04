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

# Apply log transformation for modeling
log_x = np.log(x)

# Fit cubic polynomial models on log(x)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(log_x)
model_angleon = LinearRegression().fit(X_poly, y_angleon)
model_competitor = LinearRegression().fit(X_poly, y_competitor)

# Define prediction functions
poly_transform = lambda val: poly.transform(np.log(np.array([[val]])))

def f(x_val):
    x_val = float(np.squeeze(x_val))
    return model_angleon.predict(poly_transform(x_val))[0]

def g(x_val):
    x_val = float(np.squeeze(x_val))
    return model_competitor.predict(poly_transform(x_val))[0]

# Area between curves from x = 6.3 to 50
area, _ = quad(lambda x_val: abs(f(x_val) - g(x_val)), 6.3, 50)

# Predict over a restricted range for plotting
x_range = np.linspace(6.3, 50, 300).reshape(-1, 1)
log_x_range = np.log(x_range)
y_pred_angleon = model_angleon.predict(poly.transform(log_x_range))
y_pred_competitor = model_competitor.predict(poly.transform(log_x_range))

# Determine max y for setting fixed axis range with buffer
y_max = max(y_pred_angleon.max(), y_pred_competitor.max())
y_max_buffered = y_max * 1.05

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
    xaxis_title="Belt Speed (in/sec)",
    yaxis_title="Rate of Material Loss (in/min)",
    height=650,
    hovermode='x',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    xaxis=dict(
        type='linear',
        range=[6.3, 50],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot"
    ),
    yaxis=dict(
        range=[0, y_max_buffered],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot",
        tickformat=".4f"  # disables µ-style formatting
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0)",
        font_size=12,
        font_family="Arial"
    )
)

# Annotation
fig.add_annotation(
    x=40,
    y=y_max_buffered * 0.95,
    text=f"Shaded area = {area:.6f} (in·min⁻¹·in·sec⁻¹)",
    showarrow=False,
    font=dict(size=13)
)

# Caption
st.markdown("""
This chart compares the **rate of material loss** across belt speeds for two materials.
The shaded area between the curves, from **6.3 to 50 in/sec**, represents the **cumulative durability advantage** of one material over the other.
Lower material loss indicates superior wear resistance under dynamic conditions.
""")

# Show plot
st.plotly_chart(fig, use_container_width=True)
