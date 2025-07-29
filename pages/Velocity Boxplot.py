import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Velocity Boxplots", layout="wide")
st.title("Velocity Distributions by Material")

# Load data
csv_path = "data/Velocity_Boxplots.csv"  # Adjust path if needed
df = pd.read_csv(csv_path)

# Drop rows with all NaNs (if any)
df.dropna(how='all', inplace=True)

# Melt the DataFrame to long format for Plotly boxplot
df_melted = df.melt(var_name="Material", value_name="Velocity (in/sec)")
df_melted.dropna(inplace=True)

# Plot
fig = px.box(
    df_melted,
    x="Material",
    y="Velocity (in/sec)",
    points="all",  # Show all points
    title="Distribution of Seed Velocities by Material",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
