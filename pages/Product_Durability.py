import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.write("File Exists:". os.path.exists(csv_path))
st.write("Files in /data:", os.listdir("data"))

# Load CSV from predefined path
csv_path = "data/durability_data.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Extract data
x = df['Hours']
y_angleon = df['AngleOn Run Time']
y_competitor = df['Competitor Run Time']

# Page layout
st.set_page_config(page_title="Material Loss Over Time", layout="wide")
st.title("Material Loss Over Time")
st.subheader("Cumulative Material Loss (in/min) vs Runtime (Hours)")

# Plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y_angleon,
    mode='lines+markers',
    name='AngleOn™',
    line=dict(color='blue', width=3)
))

fig.add_trace(go.Scatter(
    x=x, y=y_competitor,
    mode='lines+markers',
    name='Competitor',
    line=dict(color='red', width=3)
))

fig.update_layout(
    xaxis_title="Runtime (Hours)",
    yaxis_title="Cumulative Material Loss (in/min)",
    height=650,
    hovermode='x',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot"
    ),
    yaxis=dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        showline=True,
        spikecolor="lightgray",
        spikethickness=0.7,
        spikedash="dot",
        tickformat="e"  # scientific notation for clarity
    ),
    hoverlabel=dict(
        bgcolor="rgba(0,0,0,0)",
        font_size=12,
        font_family="Arial"
    )
)

# Description
st.markdown("This plot shows the accumulated material loss for both brushes over runtime. Lower values indicate superior durability.")

# Display plot
st.plotly_chart(fig, use_container_width=True)
