import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load CSV
csv_path = "data/decibel_measurements.csv"
df = pd.read_csv(csv_path)

# Select relevant columns
columns_to_plot = ['Ambient', 'Empty Feeder', 'Full Feeder', 'Full Feeder w/ AngleOn™']
filtered_df = df[columns_to_plot]

# Melt to long format for Plotly
melted_df = filtered_df.melt(var_name='Condition', value_name='dBA')

# Create boxplot
fig = go.Figure()

for condition in columns_to_plot:
    fig.add_trace(go.Box(
        y=melted_df[melted_df['Condition'] == condition]['dBA'],
        name=condition,
        boxpoints=False,        # Still hiding individual outlier dots
        marker_color='blue',
        fillcolor='lightblue',
        line=dict(width=1.5),   # Whisker line width
        width=0.6
    ))

# OSHA reference line at 85 dBA
fig.add_shape(
    type='line',
    x0=-0.5,
    x1=len(columns_to_plot) - 0.5,
    y0=85,
    y1=85,
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

# Adjust y-axis range
y_min = melted_df['dBA'].min()
y_max = melted_df['dBA'].max()
buffer = (y_max - y_min) * 0.2

# Layout and styling
fig.update_layout(
    title="Decibel Levels by Condition",
    yaxis_title="Sound Level (dBA)",
    xaxis_title="Condition",
    yaxis=dict(range=[y_min - buffer, y_max + buffer]),
    template='plotly_white',
    height=800,
    margin=dict(t=50, b=50, l=50, r=50),
    showlegend=False
)

# Streamlit display
st.title("Decibel Measurement Visualization")
st.markdown("Ambient and Empty Feeder represents sound pressure measurements with feeder off/on. Full Feeder and Full feeder w/ AngleOn™ used (25) 1/2-13 nuts as media tested.")
st.plotly_chart(fig, use_container_width=True)

