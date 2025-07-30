import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from plotly.colors import qualitative

# Sample data
data = {
    "Weight (lbs)": [0.167]*5 + [0.246]*5 + [1.875]*5 + [3.750]*5 + [5.625]*5 + [7.5]*5 + [9.375]*5 + [11.25]*5 + [13.125]*5 + [15.0]*5 + [18.75]*5,
    "Velocity (in/sec)": [4.56,4.56,4.55,4.54,4.54,4.57,4.56,4.56,4.55,4.55,3.17,3.15,3.09,3.10,3.11,2.75,2.78,2.75,2.76,2.77,
                           2.48,2.5,2.38,2.41,2.42,2.19,2.22,2.16,2.23,2.22,2.0,2.03,2.05,2.04,2.02,1.85,1.83,1.86,1.85,1.83,
                           1.6,1.61,1.59,1.6,1.61,1.32,1.34,1.33,1.33,1.32,0.83,0.85,0.87,0.85,0.84],
    "Pressure (lbs/in²)": [0.05]*5 + [0.17]*5 + [0.25]*5 + [0.5]*5 + [0.75]*5 + [1.0]*5 + [1.25]*5 + [1.5]*5 + [1.75]*5 + [2.0]*5 + [2.5]*5
}

df = pd.DataFrame(data)

# Streamlit UI
st.set_page_config(layout="wide")
left_col, right_col = st.columns([1, 4])

with left_col:
    st.header("Reference Sliders")
    ref_velocity = st.slider("Reference Velocity (in/sec)", min_value=0.0, max_value=5.0, value=2.5, step=0.05)
    ref_pressure = st.slider("Reference Pressure (lbs/in²)", min_value=0.0, max_value=3.0, value=1.0, step=0.05)

with right_col:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Pressure (lbs/in²)"],
        y=df["Velocity (in/sec)"],
        mode='markers',
        marker=dict(color=qualitative.Plotly[0]),
        name='Data'
    ))

    # Add reference lines
    fig.add_shape(
        type="line",
        x0=df["Pressure (lbs/in²)"].min(),
        x1=df["Pressure (lbs/in²)"].max(),
        y0=ref_velocity,
        y1=ref_velocity,
        line=dict(color="red", dash="dash"),
        name="Ref Velocity"
    )
    fig.add_shape(
        type="line",
        x0=ref_pressure,
        x1=ref_pressure,
        y0=df["Velocity (in/sec)"].min(),
        y1=df["Velocity (in/sec)"].max(),
        line=dict(color="blue", dash="dash"),
        name="Ref Pressure"
    )

    fig.update_layout(
        title="Velocity vs Pressure with Reference Lines",
        xaxis_title="Pressure (lbs/in²)",
        yaxis_title="Velocity (in/sec)",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
