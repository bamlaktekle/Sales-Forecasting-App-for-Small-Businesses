import streamlit as st, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

st.title("📊 Data Visualization & Insights")
df = pd.read_csv("data/Advertising_Data.csv")
st.dataframe(df.head())