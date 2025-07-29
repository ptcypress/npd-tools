import streamlit as st

st.set_page_config(
    page_title="New Product Development",
    page_icon="🧰",
    layout="wide"
)

st.title("🧰 New Product Development Toolbox")
st.markdown("---")

st.markdown("""
This is a collection of interactive engineering tools specific to brush development.  
Use the sidebar on the left to navigate between apps:

- 📏 **Monofilament Coverage Calculator**  
- 📊 **Velocity Boxplot Viewer**

---

### 📌 How to Use
- Choose an app from the sidebar.
- Each app will load in its own view.
- You can add new tools anytime by placing a `.py` file in the `pages/` folder.

---
""")
