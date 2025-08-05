import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import os

# Streamlit page settings
st.set_page_config(page_title="Material Loss Over Time", layout="wide")
st.title("Material Loss Over Time")
st.subheader("Cumulative Material Loss (%) vs Runtime (Hours)")

# File path
csv_path = "data/runtime_data.csv"

# Load data
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = [
        "Mat'l Loss (%)",
        "AngleOn™ Run Time (hrs)",
        "Competitor Product Run Time (hrs)"
    ]

    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns.")
        st.write("Available columns:", df.columns.tolist())
    else:
        # Extract columns
        y_loss = df["Mat'l Loss (%)"]
        x_angleon = df["AngleOn™ Run Time (hrs)"]
        x_comp = df["Competitor Product Run Time (hrs)"]

        # Interpolate for smooth curves
        angleon_interp = interp1d(x_angleon, y_loss, kind='linear', fill_value='extrapolate')
        competitor_interp = interp1d(x_comp, y_loss, kind='linear', fill_value='extrapolate')

        x_smooth = np.linspace(
            min(x_angleon.min(), x_comp.min()),
            max(x_angleon.max(), x_comp.max()),
            500
        )

        y_angleon_smooth = angleon_interp(x_smooth)
        y_comp_smooth = competitor_interp(x_smooth)

        # Plot setup
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_angleon_smooth,
            mode='lines',
            name='AngleOn™',
            line=dict(color='blue', width=3),
            hovertemplate='Hour: %{x:.2f}<br>Loss: %{y:.2f}%'
        ))

        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_comp_smooth,
            mode='lines',
            name='Competitor',
            line=dict(color='red', width=3),
            hovertemplate='Hour: %{x:.2f}<br>Loss: %{y:.2f}%'
        ))

        fig.update_layout(
            xaxis_title="Runtime (Hours)",
            yaxis_title="Cumulative Material Loss (%)",
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
                tickformat=".1f"
            ),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",
                font_size=12,
                font_family="Arial"
            )
        )

        st.markdown("This chart shows the cumulative **percent material loss** over time for both brushes. Lower values indicate superior durability.")
        st.plotly_chart(fig, use_container_width=True)
