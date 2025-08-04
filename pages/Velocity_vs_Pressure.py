import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve
import warnings

# --- Page Setup ---
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Cubic Regression Comparison: AngleOn™ vs Competitor")
st.subheader("Velocity vs Pressure")

# --- Load Data ---
csv_path = "/mnt/data/velocity_data.csv"
df = pd.read_csv(csv_path)

# --- Filter relevant data ---
df = df[df['Brush'].isin(['AngleOn™', 'Competitor'])]

angleon = df[df['Brush'] == 'AngleOn™']
competitor = df[df['Brush'] == 'Competitor']

x_angleon = angleon['Pressure (lbs/in²)'].values.reshape(-1, 1)
y_angleon = angleon['Velocity (in/sec)'].values

x_comp = competitor['Pressure (lbs/in²)'].values.reshape(-1, 1)
y_comp = competitor['Velocity (in/sec)'].values

# --- Polynomial Fit ---
poly = PolynomialFeatures(degree=3)
X_angleon_poly = poly.fit_transform(x_angleon)
X_comp_poly = poly.fit_transform(x_comp)

model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
model2 = LinearRegression().fit(X_comp_poly, y_comp)

# --- Functions ---
def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
def diff(x): return f(x) - g(x)

# --- Find Intersection ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        x_intersect = fsolve(diff, x0=1.0)[0]
        valid_intersection = True if 0 <= x_intersect <= df['Pressure (lbs/in²)'].max() else False
    except:
        x_intersect = None
        valid_intersection = False

# --- Area Between Curves ---
if valid_intersection:
    area, _ = quad(diff, 0, x_intersect)
else:
    area = None

# --- Build Plot ---
x_range = np.linspace(0, df['Pressure (lbs/in²)'].max() + 0.5, 300)
angleon_fit = [f(x) for x in x_range]
competitor_fit = [g(x) for x in x_range]

fig = go.Figure()

# --- Shaded Area Between Curves ---
if valid_intersection:
    x_fill = np.linspace(0, x_intersect, 300)
    y1 = np.array([f(x) for x in x_fill])
    y2 = np.array([g(x) for x in x_fill])

    fill_x = np.concatenate([x_fill, x_fill[::-1]])
    fill_y = np.concatenate([y1, y2[::-1]])

    fig.add_trace(go.Scatter(
        x=fill_x,
        y=fill_y,
        mode='lines',
        line=dict(width=0),
        fill='toself',
        fillcolor='rgba(150,150,150,0.3)',
        name='Advantage Area (0 to Intersection)'
    ))

# --- Regression Lines ---
fig.add_trace(go.Scatter(x=x_range, y=angleon_fit, mode='lines',
                         name='AngleOn™ Cubic Fit', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range, y=competitor_fit, mode='lines',
                         name='Competitor Cubic Fit', line=dict(color='red', width=3)))

# --- Annotations ---
if valid_intersection:
    fig.add_annotation(
        x=x_intersect,
        y=g(x_intersect),
        text=f"Intersection @ {x_intersect:.2f} psi",
        showarrow=True,
        arrowhead=3,
        ax=40,
        ay=-40
    )
    fig.add_annotation(
        x=x_intersect / 2,
        y=(np.max(y1) + np.max(y2)) / 2,
        text=f"Shaded Area = {area:.3f} in/sec·psi",
        showarrow=False,
        font=dict(size=14)
    )

# --- Layout ---
fig.update_layout(
    xaxis_title="Pressure (lbs/in²)",
    yaxis_title="Velocity (in/sec)",
    hovermode='x',
    height=650,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor"),
    yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor"),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
)

st.plotly_chart(fig, use_container_width=True)
