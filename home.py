import streamlit as st

st.set_page_config(
    page_title="New Product Development",
    page_icon="🧰",
    layout="wide"
)

st.title("🧰 New Product Development - AngleOn™ & AngleOn XT")
st.markdown("---")

st.markdown("""
This is a collection of interactive engineering analyses specific to angle brush development.  
Use the sidebar on the left to navigate between apps:

-  **Product Durability:**
    -  Material Loss vs Belt Speed (Accelerated Wear Testing)
    -  Product Durability (Calculated Material Life Expectancy)
-  **Product Physical Characteristics:**
    -  Monofilament Coverage (Comparison of Coverage Area Across Different Brush Types)
    -  Monofilament Density (Area Covered by Monofilament per Square Inch of Brush)
-  **Product Functional Characteristics:**  
    -  Velocity Boxplot
    -  Velocity vs Pressure

    
---

### 📌 How to Use
- Choose an app from the sidebar.
- Each app will load in its own view.


---
""")





