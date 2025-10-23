import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Sales Forecasting", page_icon="📈", layout="wide")

st.title("📈 2Sales Forecasting for Small Businesses")

st.markdown(
    """
**🎯 Objectives**

Help a small business decide **how to allocate ad spend** to maximize **`Product_Sold`**.
We’ll describe the dataset, surface quick facts, and prepare for modeling on the next pages.

**🧮 Target & Features**

- **Target:** `Product_Sold`  
- **Features (ad spend):** `TV`, `Billboards`, `Google_Ads`, `Social_Media`, `Influencer_Marketing`, `Affiliate_Marketing`
    """
)

DATA_PATH = Path("data/Advertising_Data.csv")
EXPECTED_COLS = {
    "TV",
    "Billboards",
    "Google_Ads",
    "Social_Media",
    "Influencer_Marketing",
    "Affiliate_Marketing",
    "Product_Sold",
}

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip whitespace from headers just in case
    df.columns = [c.strip() for c in df.columns]
    # drop fully empty rows
    df = df.dropna(how="all")
    return df.reset_index(drop=True)

# ===== Dataset section =====
st.subheader("🗃️ Dataset")

if not DATA_PATH.exists():
    st.error(f"CSV not found at `{DATA_PATH.as_posix()}`. Make sure the file is in the `data/` folder.")
    st.stop()

df = load_data(DATA_PATH)

# Validate expected columns (helps avoid silent errors later)
missing = sorted(list(EXPECTED_COLS - set(df.columns)))
extra = sorted(list(set(df.columns) - EXPECTED_COLS))
if missing:
    st.warning(
        "Some expected columns are missing. "
        f"**Missing:** {missing}\n\n"
        "Update column names in the CSV or adjust downstream pages to match."
    )
if extra:
    with st.expander("Columns present that are not used by the app (OK to keep):"):
        st.write(extra)

# Preview
st.markdown("**Preview (first 10 rows)**")
st.dataframe(df.head(10), use_container_width=True)

# Quick stats block
st.subheader("📊 Quick stats")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing values", int(df.isna().sum().sum()))
c4.metric("Numerical cols", df.select_dtypes("number").shape[1])

with st.expander("Descriptive statistics"):
    st.write(df.describe(include="all").T)

st.caption(
    "All spend columns are in currency-like units; the target is the number of products sold. "
    "Proceed to **Data Visualization** for insights, then to **Model Prediction** to train and evaluate."
)
