import streamlit as st, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

st.title("Sales Forecasting App for Small Business")
df = pd.read_csv("data/Advertising_Data.csv")
st.dataframe(df.head())