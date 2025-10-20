import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sales Forecasting", page_icon="📈")

st.title("📈 Sales Forecasting for Small Businesses")
st.markdown("""
**Goal:** Predict **`Product_Sold`** from six marketing channels:
`TV`, `Billboards`, `Google_Ads`, `Social_Media`, `Influencer_Marketing`, `Affiliate_Marketing`.

**Business problem:** Help a small business decide **how to allocate ad spend** to maximize sales.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("data/Advertising_Data.csv")
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

st.subheader("Dataset preview")
st.dataframe(df.head())

st.subheader("Quick stats")
st.write(df.describe())

st.caption("All spend columns appear to be in currency units; target is the number of products sold.")
