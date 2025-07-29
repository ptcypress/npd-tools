import streamlit as st

st.set_page_config(
    page_title="Streamlit Tool Suite",
    page_icon="🧰",
    layout="wide"
)

st.title("🧰 Streamlit Tool Suite")
st.markdown("---")

st.markdown("""
Welcome to your collection of interactive engineering tools.  
Use the sidebar on the left to navigate between apps:

- 📏 **Monofilament Coverage Calculator**  
- 📊 **Seed Velocity Boxplot Viewer**

---

### 📌 How to Use
- Choose an app from the sidebar.
- Each app will load in its own view.
- You can add new tools anytime by placing a `.py` file in the `pages/` folder.

---
""")
