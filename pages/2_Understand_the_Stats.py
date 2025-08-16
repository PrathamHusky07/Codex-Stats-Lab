import streamlit as st

st.set_page_config(page_title="Understand the Stats", page_icon="📘")
st.title("📘 Understand the Stats")
st.subheader("Explaining Statistical Distributions for Everyone")

st.markdown("""
This section is designed for anyone without a stats background. For Stakeholders, Product managers, Marketers to understand what these distributions really mean in the real world.
""")

# -------------------------
st.markdown("### 📌 Poisson Distribution")
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/480px-Poisson_pmf.svg.png", caption="Poisson Distribution")

st.markdown("""
Imagine a **call center**. You're counting how many calls arrive in a 1-minute window.

- Some minutes have 2 calls, some have 5, some have 0, it varies.
- The **Poisson distribution** helps us model these rare, count-based events.
- It's used in traffic modeling, insurance claims, and customer service demand.

**Key Insight**: Poisson is perfect for **event frequency in fixed intervals** (e.g. time, space).
""")

# -------------------------
st.markdown("### 📌 Beta Distribution")
st.image("assets/beta_dist.png", caption="Beta Distribution")

st.markdown("""
Suppose you're testing a new ad. Out of 100 visitors, 10 click on it. Is that good?

- You're not 100% sure, there's uncertainty.
- The **Beta distribution** helps us model **probabilities and uncertainty**, especially in **A/B testing** or **Bayesian inference**.
- It's very flexible: it can model flat, peaked, or U-shaped curves depending on the parameters.

**Key Insight**: Beta is used to **model beliefs** about **probabilities** (e.g. conversion rates).
""")

# -------------------------
st.markdown("### 📌 Log-Normal Distribution")
st.image("assets/lognormal_dist.png", caption="Log-Normal Distribution")

st.markdown("""
Think about **income distribution**:

- Most people earn in a certain range, but a few earn a lot more, resulting in a **long tail**.
- The **Log-Normal distribution** models this kind of **positively skewed** data.
- It’s often used in **finance**, **biology**, and **environmental modeling**.

**Key Insight**: Log-Normal is great for variables where **values grow multiplicatively**, like income, population growth, or stock returns.
""")

st.markdown("---")
st.markdown("🔙 Use the sidebar to return to the home page.")
