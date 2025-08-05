import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.integrate import quad

# Load external dataset
csv_path = "data/durability_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

x = df['Belt Speed'].values.reshape(-1, 1)
y_angleon = df['AngleOn'].values
y_competitor = df['Competitor'].values
log_x = np.log(x)

# Polynomial regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(log_x)
model_angleon = LinearRegression().fit(X_poly, y_angleon)
model_competitor = LinearRegression().fit(X_poly, y_competitor)

# Define function handles
predict_angleon = lambda x_val: model_angleon.predict(poly.transform(np.log(np.array([[x_val]]))))[0]
predict_competitor = lambda x_val: model_competitor.predict(poly.transform(np.log(np.array([[x_val]]))))[0]

# Range and integration
x_start, x_end = 6.3, 50
area_between, _ = quad(lambda x_val: predict_angleon(x_val) - predict_competitor(x_val), x_start, x_end)
area_under_competitor, _ = quad(predict_competitor, x_start, x_end)
percent_improvement = (area_between / area_under_competitor) * 100

# Generate plot range
x_range = np.linspace(4, 55, 300)
y_pred_angleon = [predict_angleon(val) for val in x_range]
y_pred_competitor = [predict_competitor(val) for val in x_range]

# Streamlit UI
st.set_page_config(page_title="Material Loss vs Belt Speed", layout="wide")
st.title("Durability Comparison")
st.subheader("Material Loss vs Belt Speed (in/min)")

# Plotly figure
fig = go.Figure()

# Add regression lines
fig.add_trace(go.Scatter(x=x_range, y=y_pred_angleon, mode='lines', name='AngleOn™', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=x_range, y=y_pred_competitor, mode='lines', name='Competitor', line=dict(color='red', width=3)))

# Shaded region between curves
x_fill = np.linspace(x_start, x_end, 300)
y_fill_upper = [predict_competitor(xi) for xi in x_fill]
y_fill_lower = [predict_angleon(xi) for xi in x_fill]
fig.add_trace(go.Scatter(
    x=np.concatenate([x_fill, x_fill[::-1]]),
    y=np.concatenate([y_fill_upper, y_fill_lower[::-1]]),
    fill='toself',
    fillcolor='rgba(150,150,150,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Advantage Area',
    hoverinfo='skip'
))

# Annotations
fig.add_annotation(
    x=45,
    y=max(max(y_pred_angleon), max(y_pred_competitor)) * 0.95,
    text=f"Area = {abs(area_between):.6f} in/min·in/sec<br>Relative Advantage = {abs(percent_improvement):.2f}%",
    showarrow=False,
    font=dict(size=13)
)

# Layout
fig.update_layout(
    xaxis_title="Belt Speed (in/sec)",
    yaxis_title="Material Loss (in/min)",
    height=650,
    hovermode='x',
    xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True),
    yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", showline=True),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.75),
    hoverlabel=dict(bgcolor="rgba(0,0,0,0)", font_size=12, font_family="Arial")
)

st.plotly_chart(fig, use_container_width=True)

st.caption("Shaded region shows cumulative durability advantage of AngleOn™ over the competitor from 6.3 to 50 in/sec.")
