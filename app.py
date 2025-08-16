import streamlit as st

st.set_page_config(page_title="Codex Stats Lab", page_icon="📊")

st.title("Codex Stats Lab")
st.subheader("Exploring Statistical Distributions with AI Collaboration")


st.markdown("""
Welcome to **Codex Stats Lab**, a data science playground where I explored interesting properties of statistical distributions using Agentic AI, Python, Statistics and Streamlit.

This project was designed to:
- Visualize and compare statistical distributions (Poisson, Beta, Log-Normal)
- Demonstrate key statistical properties interactively
- Make statistical concepts accessible to all from analysts to executives

🤖 **Built with help from OpenAI's Codex AI** to create reusable, modular plotting and dashboard components.

---  

**Choose a page from the sidebar to get started:**
- 🧪 `Distribution Explorer` for hands-on plots and insights  
- 🧠 `Understand the Stats` for simple, real-world explanations  
""")


st.markdown("""
<div style='text-align: right; color: gray; font-size: 14px;'>
    Created by Prathamesh Kulkarni • 2025
</div>
""", unsafe_allow_html=True)

