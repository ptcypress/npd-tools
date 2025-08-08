import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load CSV
csv_path = "data/decibel_measurements.csv"  # Adjust path as needed
df = pd.read_csv(csv_path)

# Select columns to plot
columns_to_plot = ['Ambient', 'Empty Feeder', 'Full Feeder', 'AngleOn™']
filtered_df = df[columns_to_plot]

# Melt dataframe for Plotly boxplot
melted_df = filtered_df.melt(var_name='Condition', value_name='dBA')

# Create boxplot
fig = go.Figure()

# Add boxplots for each condition
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
    line=dict(color='red', dash='dash'),
)

fig.add_annotation(
    x=len(columns_to_plot)-1,
    y=85,
    text="85 (dBA) OSHA TWA Action Level",
    showarrow=False,
    yshift=10,
    font=dict(color="red", size=12)
)

# Layout formatting
fig.update_layout(
    title="Decibel Levels by Condition",
    yaxis_title="Sound Level (dBA)",
    xaxis_title="Condition",
    yaxis=dict(range=[filtered_df.min().min() - 5, filtered_df.max().max() + 5]),
    template='plotly_white'
)

# Streamlit app
st.title("Decibel Measurement Visualization")
st.plotly_chart(fig, use_container_width=True)
