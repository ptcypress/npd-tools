import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load CSV
csv_path = "data/decibel_measurements.csv"
df = pd.read_csv(csv_path)

# Select relevant columns
columns_to_plot = ['Ambient', 'Empty Feeder', 'Full Feeder', 'AngleOn™']
filtered_df = df[columns_to_plot]

# Melt into long format for Plotly
melted_df = filtered_df.melt(var_name='Condition', value_name='dBA')

# Create Plotly boxplot figure
fig = go.Figure()

# Add each box
for condition in columns_to_plot:
    fig.add_trace(go.Box(
        y=melted_df[melted_df['Condition'] == condition]['dBA'],
        name=condition,
        boxpoints=False,  # Hide outlier dots
        line=dict(width=0),  # Hide whiskers ("fences")
        marker_color='blue',
        fillcolor='lightblue',
        width=0.6
    ))

# Add OSHA 85 dBA reference line
fig.add_shape(
    type='line',
    x0=-0.5, x1=len(columns_to_plot)-0.5,
    y0=85, y1=85,
    line=dict(color='red', dash='dash', width=2)
)

fig.add_annotation(
    x=len(columns_to_plot) - 0.5,
    y=85,
    text="85 (dBA) OSHA TWA Action Level",
    showarrow=False,
    yshift=10,
    font=dict(color="red", size=12)
)

# Dynamic range with buffer
y_min = melted_df['dBA'].min()
y_max = melted_df['dBA'].max()
buffer = (y_max - y_min) * 0.2

# Update layout for spacing and size
fig.update_layout(
    title="Decibel Levels by Condition",
    yaxis_title="Sound Level (dBA)",
    xaxis_title="Condition",
    yaxis=dict(range=[y_min - buffer, y_max + buffer]),
    template='plotly_white',
    height=800,  # Taller plot for more detail
    margin=dict(t=50, b=50, l=50, r=50)
)

# Streamlit app display
st.title("Decibel Measurement Visualization")
st.plotly_chart(fig, use_container_width=True)
