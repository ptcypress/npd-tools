import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(page_title="Velocity Boxplots", layout="wide")
st.title("Velocity Distribution by Brush Type")

# Load data
df = pd.read_csv("data/Velocity_Boxplots.csv")

# Drop completely empty rows (common in Excel exports)
df.dropna(how='all', inplace=True)

# Melt the wide-format DataFrame to long-format
df_long = df.melt(var_name="Brush Type", value_name="Velocity (in/sec)")
df_long.dropna(inplace=True)  # drop any rows with NaN velocity

# Create boxplot
fig = px.box(
    df_long,
    x="Brush Type",
    y="Velocity (in/sec)",
    points="all",  # Shows all the raw data points
    title="Seed Ejection Velocity by Brush Type",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
