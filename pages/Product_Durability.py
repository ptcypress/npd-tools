import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Local file path
csv_path = "data/product_durability.csv"

# Page setup
st.set_page_config(page_title="Material Loss Over Time", layout="wide")
st.title("Material Loss Over Time")
st.subheader("Cumulative Material Loss (in) vs Runtime (Hours)")

# Check file existence
if not os.path.exists(csv_path):
    st.error(f"File not found: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = ["Mat'l Loss (in)", "AngleOn™ Run Time (hrs)", "Competitor Product Run Time (hrs)"]
    if not all(col in df.columns for col in required_cols):
        st.error("Required columns missing from CSV.")
        st.write("Available columns:", df.columns.tolist())
    else:
        x_angleon = df["AngleOn™ Run Time (hrs)"]
        x_comp = df["Competitor Product Run Time (hrs)"]
        y_loss = df["Mat'l Loss (%)"]

        # Plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_angleon, y=y_loss,
            mode='lines+markers',
            name='AngleOn™',
            line=dict(color='blue', width=3)
        ))

        fig.add_trace(go.Scatter(
            x=x_comp, y=y_loss,
            mode='lines+markers',
            name='Competitor',
            line=dict(color='red', width=3)
        ))

        fig.update_layout(
            xaxis_title="Runtime (Hours)",
            yaxis_title="Cumulative Material Loss (in)",
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
                tickformat="e"
            ),
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",
                font_size=12,
                font_family="Arial"
            )
        )

        st.markdown("This chart shows the cumulative material loss of two brushes as a function of their runtime. Lower material loss implies greater durability.")
        st.plotly_chart(fig, use_container_width=True)
