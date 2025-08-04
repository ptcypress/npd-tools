import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad
from scipy.optimize import fsolve
import streamlit as st

# Load CSV from correct relative path
file_path = "data/velocity_data.csv"
df = pd.read_csv(file_path)

# Filter for the two brushes
angleon = df[df["Brush"] == "AngleOn™"]
competitor = df[df["Brush"] == "Competitor"]

x1 = angleon["Pressure"].values
y1 = angleon["Velocity"].values
x2 = competitor["Pressure"].values
y2 = competitor["Velocity"].values

# Fit cubic polynomial models
poly = PolynomialFeatures(degree=3)
X1_poly = poly.fit_transform(x1.reshape(-1, 1))
X2_poly = poly.fit_transform(x2.reshape(-1, 1))

model1 = LinearRegression().fit(X1_poly, y1)
model2 = LinearRegression().fit(X2_poly, y2)

# Define model functions
def f(x): return model1.predict(poly.transform(np.array(x).reshape(-1, 1))).flatten()
def g(x): return model2.predict(poly.transform(np.array(x).reshape(-1, 1))).flatten()
def diff(x): return f(x) - g(x)

# Find intersection point
x_intersect = fsolve(diff, x0=1.0)[0]
area_between, _ = quad(lambda x: abs(f(x) - g(x)), 0, x_intersect)

# Generate fitted data
x_fit = np.linspace(min(x1.min(), x2.min()), max(x1.max(), x2.max()), 300)
y1_fit = [f(xi) for xi in x_fit]
y2_fit = [g(xi) for xi in x_fit]

# Shaded region
x_shade = np.linspace(0, x_intersect, 300)
y_upper = np.maximum([f(xi) for xi in x_shade], [g(xi) for xi in x_shade])
y_lower = np.minimum([f(xi) for xi in x_shade], [g(xi) for xi in x_shade])

# Build interactive plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_fit, y=y1_fit, mode='lines', name="AngleOn™ Cubic Fit",
    line=dict(color="blue", width=3)
))

fig.add_trace(go.Scatter(
    x=x_fit, y=y2_fit, mode='lines', name="Competitor Cubic Fit",
    line=dict(color="red", width=3)
))

fig.add_trace(go.Scatter(
    x=np.concatenate([x_shade, x_shade[::-1]]),
    y=np.concatenate([y_upper, y_lower[::-1]]),
    fill="toself",
    fillcolor="rgba(100,100,100,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    hoverinfo="skip",
    name="Performance Advantage Area"
))

# Annotate chart
fig.add_annotation(
    x=x_intersect, y=f(x_intersect),
    text=f"Intersection @ {x_intersect:.2f} psi",
    showarrow=True, arrowhead=1, ax=0, ay=-40
)

fig.add_annotation(
    x=0.5 * x_intersect, y=np.max(y_upper),
    text=f"Total Advantage Area = {area_between:.3f} in/sec·psi",
    showarrow=False,
    font=dict(size=13, color="black"),
    bgcolor="rgba(255,255,255,0.7)",
    bordercolor="gray",
    borderwidth=1
)

# Cursor reference lines
fig.update_layout(
    xaxis=dict(
        title="Pressure (lbs/in²)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="grey",
        spikethickness=1,
        spikedash="dot"
    ),
    yaxis=dict(
        title="Velocity (in/sec)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="grey",
        spikethickness=1,
        spikedash="dot"
    ),
    title="Velocity vs Pressure — AngleOn™ vs Competitor",
    height=600,
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.79)
)

# Display in Streamlit
st.set_page_config(page_title="Velocity Comparison", layout="wide")
st.title("Cubic Regression Comparison: AngleOn™ vs Competitor")
st.plotly_chart(fig, use_container_width=True)
