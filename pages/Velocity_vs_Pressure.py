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

# Pivot data so each brush has its own column
pivot_df = df.pivot_table(index='Pressure', columns='Brush', values='Velocity').reset_index()

# Extract x and y values
x = pivot_df['Pressure'].values.reshape(-1, 1)
y_angleon = pivot_df['AngleOn™'].values
y_competitor = pivot_df['Competitor'].values

# Polynomial model (cubic)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)

model1 = LinearRegression().fit(X_poly, y_angleon)
model2 = LinearRegression().fit(X_poly, y_competitor)

# Define fit functions
def f(x_val): return model1.predict(poly.transform(np.array(x_val).reshape(-1, 1))).flatten()
def g(x_val): return model2.predict(poly.transform(np.array(x_val).reshape(-1, 1))).flatten()

def diff(x_val): return f(x_val) - g(x_val)

# Solve for intersection
x_intersect = fsolve(diff, x0=1.0)[0]

# Integrate area between functions up to intersection
area, _ = quad(lambda x: f(x) - g(x), 0, x_intersect)

# Prepare line plot data
pressure_range = np.linspace(0, x_intersect + 0.5, 300)
angleon_fit = f(pressure_range)
competitor_fit = g(pressure_range)

# Streamlit layout
st.set_page_config(page_title="Cubic Regression Comparison", layout="wide")
st.title("Cubic Regression Comparison: AngleOn™ vs Competitor")
st.subheader("Velocity vs Pressure — AngleOn™ vs Competitor")

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
    name='AngleOn™ Cubic Fit',
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
    x=x_intersect, y=g([x_intersect])[0],
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
