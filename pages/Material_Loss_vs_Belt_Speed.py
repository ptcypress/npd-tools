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
csv_path = "data/velocity_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # clean up just in case

# --- Dynamically find column names ---
pressure_col = [col for col in df.columns if "Pressure" in col][0]
velocity_col = [col for col in df.columns if "Velocity" in col][0]

# --- Filter only AngleOn™ and Competitor data ---
df = df[df['Brush'].isin(['AngleOn™', 'Competitor'])]

angleon = df[df['Brush'] == 'AngleOn™']
competitor = df[df['Brush'] == 'Competitor']

# --- Prepare data for regression ---
x_angleon = angleon[pressure_col].values.reshape(-1, 1)
y_angleon = angleon[velocity_col].values

x_comp = competitor[pressure_col].values.reshape(-1, 1)
y_comp = competitor[velocity_col].values

# --- Polynomial Regression (Cubic) ---
poly = PolynomialFeatures(degree=3)
X_angleon_poly = poly.fit_transform(x_angleon)
X_comp_poly = poly.fit_transform(x_comp)

model1 = LinearRegression().fit(X_angleon_poly, y_angleon)
model2 = LinearRegression().fit(X_comp_poly, y_comp)

# --- Fit functions ---
def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
def diff(x): return f(x) - g(x)

# --- Find intersection ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        x_intersect = fsolve(diff, x0=1.0)[0]
        valid_intersection = True if 0 <= x_intersect <= df[pressure_col].max() else False
    except:
        x_intersect = None
        valid_intersection = False

# --- Area Between Curves ---
if valid_intersection:
    area, _ = quad(diff, 0, x_intersect)
else:
    area = None

# --- Generate shared x-range for smooth plotting & shading ---
x_range = np.linspace(0, df[pressure_col].max() + 0.5, 300).reshape(-1, 1)
X_range_poly = poly.transform(x_range)
y_pred_angleon = model1.predict(X_range_poly)
y_pred_competitor = model2.predict(X_range_poly)

# Subset only the values up to x_intersect for shading
x_fill = np.linspace(0, x_intersect, 300).reshape(-1, 1)
X_fill_poly = poly.transform(x_fill)
x_vals = x_fill.flatten()
y1_fill = model1.predict(X_fill_poly)
y2_fill = model2.predict(X_fill_poly)

# --- Create plot ---
fig = go.Figure()

# --- Shaded area between curves from 0 to intersection ---
if valid_intersection:
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_vals, x_vals[::-1]]),
        y=np.concatenate([y1_fill, y2_fill[::-1]]),
        fill='toself',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(150,150,150,0.3)',
        name='Shaded Area (0 to Intersection)'
    ))

# --- Add regression lines ---
fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred_angleon, mode='lines',
                         name='AngleOn™ Cubic Fit', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred_competitor, mode='lines',
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
        y=(np.max(y1_fill) + np.max(y2_fill)) / 2,
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

# --- Show plot ---
st.plotly_chart(fig, use_container_width=True)
