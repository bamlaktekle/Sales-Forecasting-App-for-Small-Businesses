import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Data Visualization & Insights")

df = pd.read_csv("data/Advertising_Data.csv").dropna()

st.markdown("### Scatterplots (Sales vs. Channel Spend)")

feature_cols = ["TV","Billboards","Google_Ads","Social_Media","Influencer_Marketing","Affiliate_Marketing"]

for col in feature_cols:
    fig, ax = plt.subplots()
    sns.regplot(data=df, x=col, y="Product_Sold", ax=ax)
    ax.set_title(f"Product_Sold vs {col}")
    st.pyplot(fig)

st.markdown("### Correlation heatmap")
fig, ax = plt.subplots()
sns.heatmap(df[feature_cols + ["Product_Sold"]].corr(numeric_only=True), annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# quick bullets that you can tweak after you look at the heatmap
st.markdown("""
**Takeaways (fill after looking at your heatmap):**
- Channels with the strongest positive correlation to `Product_Sold`: _e.g., TV, Google_Ads_.
- Channels with weaker correlation: _e.g., Affiliate_Marketing or Billboards if coefficients are small_.
- The relationships look roughly linear → **linear regression is appropriate** for a first pass.
""")
