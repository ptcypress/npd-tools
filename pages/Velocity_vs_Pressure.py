import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.integrate import quad
from scipy.optimize import fsolve
import warnings

# --- Page Configuration ---
st.set_page_config(page_title="Velocity vs Pressure", layout="wide")
st.title("Cubic Regression Comparison: AngleOn™ vs Competitor")
st.subheader("Velocity vs Pressure")

# --- Load Data ---
csv_path = "data/velocity_data.csv"
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"CSV file not found at path: {csv_path}")
    st.stop()

required_columns = {'Brush', 'Pressure', 'Velocity'}
if not required_columns.issubset(df.columns):
    st.error(f"CSV must contain columns: {required_columns}")
    st.stop()

# --- Filter Data ---
angleon = df[df['Brush'].str.lower() == 'angleon']
competitor = df[df['Brush'].str.lower() == 'competitor']

if angleon.empty or competitor.empty:
    st.error("AngleOn™ or Competitor data is missing from the CSV.")
    st.dataframe(df)
    st.stop()

x_angleon = angleon['Pressure'].values.reshape(-1, 1)
y_angleon = angleon['Velocity'].values

x_comp = competitor['Pressure'].values.reshape(-1, 1)
y_comp = competitor['Velocity'].values

# --- Polynomial Regression (Cubic) ---
poly = PolynomialFeatures(degree=3)
X_angleon_poly = poly.fit_transform(x_angleon)
X_comp_poly = poly.fit_transform(x_comp)

model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
model2 = LinearRegression().fit(X_comp_poly, y_comp)

# --- Prediction Functions ---
def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
def diff(x): return f(x) - g(x)

# --- Solve for Intersection ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        x_intersect = fsolve(diff, x0=1.0)[0]
        if not (0 <= x_intersect <= max(df['Pressure'])):
            raise ValueError("Intersection out of range.")
        valid_intersection = True
    except:
        x_intersect = None
        valid_intersection = False

# --- Area Between Curves ---
if valid_intersection:
    area, _ = quad(diff, 0, x_intersect)
else:
    area = None

# --- Prepare Plot Data ---
x_range = np.linspace(0, max(df['Pressure']) + 0.5, 300)
angleon_fit = [f(x) for x in x_range]
competitor_fit = [g(x) for x in x_range]

# --- Plotly Visualization ---
fig = go.Figure()

# Shaded Area Between Curves
if valid_intersection:
    fill_x = np.concatenate([x_range[x_range <= x_intersect], x_range[x_range <= x_intersect][::-1]])
    fill_y = np.concatenate([[f(x) for x in fill_x[:len(fill_x)//2]],
                             [g(x) for x in fill_x[len(fill_x)//2:][::-1]]])
    fig.add_trace(go.Scatter(
        x=fill_x, y=fill_y,
        fill='toself',
        fillcolor='rgba(150,150,150,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Performance Advantage Area'
    ))

# Fitted Curves
fig.add_trace(go.Scatter(x=x_range, y=angleon_fit, mode='lines',
                         name='AngleOn™ Cubic Fit', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range, y=competitor_fit, mode='lines',
                         name='Competitor Cubic Fit', line=dict(color='red', width=3)))

# Raw Data
fig.add_trace(go.Scatter(x=angleon['Pressure'], y=angleon['Velocity'],
                         mode='markers', name='AngleOn™ Data',
                         marker=dict(color='blue', size=6)))
fig.add_trace(go.Scatter(x=competitor['Pressure'], y=competitor['Velocity'],
                         mode='markers', name='Competitor Data',
                         marker=dict(color='red', size=6)))

# Annotations
if valid_intersection:
    fig.add_annotation(x=x_intersect, y=g(x_intersect),
                       text=f"Intersection @ {x_intersect:.2f} psi",
                       showarrow=True, arrowhead=3, ax=40, ay=-40)

    fig.add_annotation(
        x=x_range[len(x_range) // 2],
        y=(max(angleon_fit) + max(competitor_fit)) / 2,
        text=f"Total Advantage Area = {area:.3f} in/sec·psi",
        showarrow=False,
        font=dict(size=14)
    )

# Final Layout
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
