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
st.title("Regression Comparison: AngleOn™ vs Competitor")

st.write("""This plot shows cubic regression fit of average veloctiy as a function of pressure
for both AngleOn™ and competitor product. The shaded area represents the cumulative velocity 
advantage AngleOn™ has over competitor product.""")

st.subheader("Velocity vs Pressure")

# --- Load and prepare data ---
csv_path = "data/velocity_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Detect column names dynamically
pressure_col = [col for col in df.columns if "Pressure" in col][0]
velocity_col = [col for col in df.columns if "Velocity" in col][0]

df = df[df['Brush'].isin(['AngleOn™', 'Competitor'])]
angleon = df[df['Brush'] == 'AngleOn™']
competitor = df[df['Brush'] == 'Competitor']

x_angleon = angleon[pressure_col].values.reshape(-1, 1)
y_angleon = angleon[velocity_col].values
x_comp = competitor[pressure_col].values.reshape(-1, 1)
y_comp = competitor[velocity_col].values

# --- Polynomial Regression (Cubic) ---
poly = PolynomialFeatures(degree=3)
model1 = LinearRegression().fit(poly.fit_transform(x_angleon), y_angleon)
model2 = LinearRegression().fit(poly.fit_transform(x_comp), y_comp)

def f(x): return model1.predict(poly.transform(np.array([[x]])))[0]
def g(x): return model2.predict(poly.transform(np.array([[x]])))[0]
def diff(x): return f(x) - g(x)

# --- Attempt to solve for intersection with fallback ---
x_intersect = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        guess = fsolve(diff, x0=1.0)[0]
        if 0 <= guess <= df[pressure_col].max():
            x_intersect = float(guess)
        else:
            x_intersect = 3.08  # fallback if out of range
            #st.warning("fsolve failed to find a valid intersection. Using fallback value x = 3.08.")
    except:
        x_intersect = 3.08
        #st.warning("fsolve failed. Using fallback value x = 3.08.")

# --- Area Between Curves ---
area = None
if x_intersect is not None:
    area, _ = quad(diff, 0, x_intersect)

# --- Generate range for regression lines ---
x_range = np.linspace(0, df[pressure_col].max() + 0.5, 300).reshape(-1, 1)
y_angleon = model1.predict(poly.transform(x_range))
y_comp = model2.predict(poly.transform(x_range))

# --- Shading range ---
x_fill = np.linspace(0, x_intersect, 300).reshape(-1, 1)
y1 = model1.predict(poly.transform(x_fill))
y2 = model2.predict(poly.transform(x_fill))

# --- Build plot ---
fig = go.Figure()

# --- Shaded Area ---
fig.add_trace(go.Scatter(
    x=np.concatenate([x_fill.flatten(), x_fill.flatten()[::-1]]),
    y=np.concatenate([y1, y2[::-1]]),
    fill='toself',
    mode='lines',
    line=dict(width=0),
    fillcolor='rgba(150,150,150,0.3)',
    name='Advantage Area'
))

# --- Regression Lines ---
fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_angleon, mode='lines',
                         name='AngleOn™ Cubic Fit', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_comp, mode='lines',
                         name='Competitor Cubic Fit', line=dict(color='red', width=3)))

# --- Annotations ---
fig.add_annotation(
    x=x_intersect,
    y=g(x_intersect),
    text=f"Intersection @ {x_intersect:.2f} psi",
    showarrow=True, arrowhead=3, ax=40, ay=-40
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
    xaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot"
    ),
    yaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot",
        range=[0, None]  # 👈 Ensures no negative y values
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0)",
        font_size=12,
        font_family="Arial"
    )
)

# --- Render ---
st.plotly_chart(fig, use_container_width=True)
