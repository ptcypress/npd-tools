import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load CSV
csv_path = "data/decibel_measurements.csv"
df = pd.read_csv(csv_path)

# Select columns to plot
columns_to_plot = ['Ambient', 'Empty Feeder', 'Full Feeder', 'AngleOn™']
filtered_df = df[columns_to_plot]

# Melt dataframe for Plotly boxplot
melted_df = filtered_df.melt(var_name='Condition', value_name='dBA')

# Create boxplot
fig = go.Figure()

# Add boxplots
for condition in columns_to_plot:
    fig.add_trace(go.Box(
        y=melted_df[melted_df['Condition'] == condition]['dBA'],
        name=condition,
        boxmean='sd',
        marker_color='blue'
    ))

# Add OSHA reference line at 85 dBA
fig.add_shape(
    type='line',
    x0=-0.5, x1=len(columns_to_plot)-0.5,
    y0=85, y1=85,
    line=dict(color='red', dash='dash')
)

fig.add_annotation(
    x=len(columns_to_plot)-1,
    y=85,
    text="85 (dBA) OSHA TWA Action Level",
    showarrow=False,
    yshift=10,
    font=dict(color="red", size=12)
)

# Dynamic y-axis range with buffer
y_min = melted_df['dBA'].min()
y_max = melted_df['dBA'].max()
buffer = (y_max - y_min) * 0.2  # 20% buffer
fig.update_layout(
    title="Decibel Levels by Condition",
    yaxis_title="Sound Level (dBA)",
    xaxis_title="Condition",
    yaxis=dict(range=[y_min - buffer, y_max + buffer]),
    template='plotly_white',
    height=600  # Increased plot height
)

# Streamlit app
st.title("Decibel Measurement Visualization")
st.plotly_chart(fig, use_container_width=True)
