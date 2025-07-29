import streamlit as st
import pandas as pd
import plotly.express as px

# Page setup
st.set_page_config(page_title="Velocity Boxplots", layout="wide")
st.title("Velocity Distribution")
st.write("This boxplot visualization compares velocity distributions for different brush types.")

# File upload
uploaded_file = st.file_uploader("Upload your velocity data CSV", type=["csv"])
if uploaded_file:
    # Read and display data
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Melt data to long format for boxplotting
    df_long = df.melt(var_name="Material", value_name="Velocity (in/sec)")

    # Drop NaNs if any
    df_long = df_long.dropna()

    # Plot
    fig = px.box(
        df_long,
        x="Material",
        y="Velocity (in/sec)",
        points="all",  # show all data points
        title="Velocity Distribution by Material",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file with velocity data.")
