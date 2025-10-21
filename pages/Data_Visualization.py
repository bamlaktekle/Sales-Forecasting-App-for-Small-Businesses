import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

st.title("📊 Data Visualization & Insights")

DATA_PATH = Path("data/Advertising_Data.csv")
FEATURES = ["TV","Billboards","Google_Ads","Social_Media","Influencer_Marketing","Affiliate_Marketing"]
TARGET = "Product_Sold"

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cols = FEATURES + [TARGET]
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df

if not DATA_PATH.exists():
    st.error(f"CSV not found at `{DATA_PATH.as_posix()}`.")
    st.stop()

df = load_data(DATA_PATH)

st.markdown("### Scatterplots (Sales vs. Channel Spend)")
for col in FEATURES:
    fig, ax = plt.subplots()
    sns.regplot(data=df, x=col, y=TARGET, ax=ax, scatter_kws={'alpha':0.6}, line_kws={'linewidth':2})
    ax.set_title(f"{TARGET} vs {col}")
    ax.set_xlabel(f"{col} (spend)")
    ax.set_ylabel(TARGET)
    st.pyplot(fig, clear_figure=True)

st.markdown("### Correlation heatmap")
fig, ax = plt.subplots()
corr_df = df[FEATURES + [TARGET]].corr(numeric_only=True)
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix")
st.pyplot(fig, clear_figure=True)

# Correlation bar chart (feature → target)
st.markdown("### Feature–Target correlation (absolute)")
corr_to_y = corr_df[TARGET].drop(TARGET).reindex(FEATURES).sort_values(key=lambda s: s.abs(), ascending=False)
fig, ax = plt.subplots()
corr_to_y.plot(kind="bar", ax=ax)
ax.set_ylabel("Correlation with Product_Sold")
ax.set_ylim(-1, 1)
ax.set_title("Strength of linear relationship")
st.pyplot(fig, clear_figure=True)

# Auto takeaways from correlations
top3 = corr_to_y.abs().nlargest(3).index.tolist()
weak = corr_to_y.abs().nsmallest(2).index.tolist()

st.markdown("### Key takeaways")
st.success(
    "- Strongest drivers (by correlation): **{}**.\n"
    "- Weaker drivers: **{}**.\n"
    "- Relationships are broadly linear → a **linear regression** baseline is appropriate.\n"
    "- Consider multicollinearity if two channels are highly correlated with each other."
    .format(", ".join(top3), ", ".join(weak))
)
