#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.colors import qualitative

# App setup
st.set_page_config(page_title="Monofilament Coverage", layout="wide")
st.title("Monofilament Coverage: Percent Area vs EPI² & Filament Diameter")

st.write("""
Use the sliders in the sidebar to explore how different EPI² values affect theoretical coverage
over a range of filament diameters. A horizontal dashed line shows the hexagonal packing limit (~90.69%).
""")

# Sidebar controls
st.sidebar.header("EPI² Values (ends per inch squared)")
epi_inputs = {
    "Brushlon": st.sidebar.slider("Brushlon", min_value=500, max_value=15_000, value=9_750, step=10),
    "AngleOn™": st.sidebar.slider("AngleOn™", min_value=500, max_value=15_000, value=6_910, step=10),
    "XT10": st.sidebar.slider("XT10", min_value=500, max_value=15_000, value=2_250, step=10),
    "XT16": st.sidebar.slider("XT16", min_value=500, max_value=15_000, value=1_125, step=10),
}

diameter_min, diameter_max = st.sidebar.slider("Diameter Range (inches)", 0.002, 0.022, (0.002, 0.022), step=0.0005)

# Calculate diameters to sample
diameters = np.arange(diameter_min, diameter_max + 1e-9, 0.0001)

def calculate_area_coverage(epi_squared, diameters):
    filament_area = np.pi * (diameters / 2) ** 2
    return epi_squared * filament_area * 100

# Build plot
fig = go.Figure()
colors = qualitative.Plotly

# Plot each material line
for idx, (label, epi_val) in enumerate(epi_inputs.items()):
    coverage = calculate_area_coverage(epi_val, diameters)
    fig.add_trace(go.Scatter(
        x=diameters,
        y=coverage,
        mode='lines',
        name=f"{label} ({epi_val})",
        line=dict(color=colors[idx % len(colors)], width=2)
    ))

# Theoretical max limit
fig.add_trace(go.Scatter(
    x=[diameters[0], diameters[-1]],
    y=[90.69, 90.69],
    mode='lines',
    name='Hexagonal Packing Limit (~90.69%)',
    line=dict(color='gray', width=3, dash='dash')
))

# Reference vertical lines
ref_diams = [0.0045, 0.006, 0.010, 0.016]
for d in ref_diams:
    fig.add_shape(
        type='line',
        x0=d, x1=d, y0=0, y1=100,
        line=dict(color='darkgray', width=2, dash='dot'),
        xref='x', yref='y'
    )

fig.update_layout(
    xaxis_title='Filament Diameter (in)',
    yaxis_title='Percent Area Coverage',
    yaxis=dict(range=[0, 100]),
    title='Coverage vs Diameter for Various Materials',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

