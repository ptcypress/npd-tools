import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Seed Velocity Boxplots", layout="wide")
st.title("Seed Velocity Distribution")

# Read from a local CSV file within the repo
df = pd.read_csv("data/seed_velocity_data.csv")

# Melt and clean
df_long = df.melt(var_name="Material", value_name="Velocity (in/sec)")
df_long = df_long.dropna()

# Plot
fig = px.box(
    df_long,
    x="Material",
    y="Velocity (in/sec)",
    points="all",
    title="Velocity Distribution by Material",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)
