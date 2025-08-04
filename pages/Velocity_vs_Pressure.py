import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve

# Load data
csv_path = "data/velocity_data.csv"
df = pd.read_csv(csv_path)

# Filter data
angleon = df[df['Brush'] == 'AngleOn']
competitor = df[df['Brush'] == 'Competitor']

x_angleon = angleon['Pressure'].values.reshape(-1, 1)
y_angleon = angleon['Velocity'].values

x_comp = competitor['Pressure'].values.reshape(-1, 1)
y_comp = competitor['Velocity'].values

# Polynomial model (cubic)
poly = PolynomialFeatures(degree=3)
X_angleon_poly = poly.fit_transform(x_angleon)
X_comp_poly = poly.fit_transform(x_comp)

model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
model2 = LinearRegression().fit(X_comp_poly, y_comp)

# Define fit functions
def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]

def diff(x): return f(x) - g(x)

# Solve for intersection
x_intersect = fsolve(diff, x0=1.0)[0]

# Integrate area between functions up to intersection
area, _ = quad(diff, 0, x_intersect)

# Prepare line plot data
pressure_range = np.linspace(0, x_intersect + 0.5, 300)
angleon_fit = [f(x) for x in pressure_range]
competitor_fit = [g(x) for x in pressure_range]

# Streamlit layout
st.set_page_config(page_title="Cubic Regression Comparison", layout="wide")
st.title("Cubic Regression Comparison: AngleOn\u2122 vs Competitor")
st.subheader("Velocity vs Pressure — AngleOn\u2122 vs Competitor")

# Plot
fig = go.Figure()

# Add shaded area between curves
fig.add_trace(go.Scatter(
    x=np.concatenate([pressure_range, pressure_range[::-1]]),
    y=np.concatenate([angleon_fit, competitor_fit[::-1]]),
    fill='toself',
    fillcolor='rgba(150,150,150,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    name='Performance Advantage Area'
))

# Add regression lines
fig.add_trace(go.Scatter(
    x=pressure_range, y=angleon_fit,
    mode='lines',
    name='AngleOn\u2122 Cubic Fit',
    line=dict(color='blue', width=3)
))

fig.add_trace(go.Scatter(
    x=pressure_range, y=competitor_fit,
    mode='lines',
    name='Competitor Cubic Fit',
    line=dict(color='red', width=3)
))

# Annotate intersection point and area
fig.add_annotation(
    x=x_intersect, y=g(x_intersect),
    text=f"Intersection @ {x_intersect:.2f} psi",
    showarrow=True, arrowhead=3, ax=40, ay=-40
)

fig.add_annotation(
    x=pressure_range[len(pressure_range)//2],
    y=(max(angleon_fit) + max(competitor_fit)) / 2,
    text=f"Total Advantage Area = {area:.3f} in/sec·psi",
    showarrow=False,
    font=dict(size=14)
)

# Add crosshairs and live cursor
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    hovermode='x',
    height=650,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True),
    yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

st.plotly_chart(fig, use_container_width=True)
