# pages/2_Model_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🤖 Model & Prediction")

# ---------- Load & prepare data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Advertising_Data.csv")
    # Ensure numeric types and drop any rows with missing/invalid values
    cols = ["TV","Billboards","Google_Ads","Social_Media","Influencer_Marketing","Affiliate_Marketing","Product_Sold"]
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df

df = load_data()

FEATURES = ["TV","Billboards","Google_Ads","Social_Media","Influencer_Marketing","Affiliate_Marketing"]
TARGET = "Product_Sold"

X = df[FEATURES]
y = df[TARGET]

# ---------- Train / test split + model ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(Xtr, ytr)

# ---------- Metrics ----------
pred = model.predict(Xte)
r2   = r2_score(yte, pred)
mae  = mean_absolute_error(yte, pred)
mse  = mean_squared_error(yte, pred)       # version-agnostic
rmse = float(mse ** 0.5)

st.subheader("Performance")
st.write(f"**R²:** {r2:.3f}   |   **MAE:** {mae:.2f}   |   **RMSE:** {rmse:.2f}")

# ---------- Coefficients (feature impact) ----------
st.subheader("Channel impact (coefficients)")
coef_df = (
    pd.Series(model.coef_, index=FEATURES)
      .sort_values(ascending=False)
      .to_frame("Coefficient")
)
st.dataframe(coef_df.style.format(precision=3))
st.caption("Interpretation: holding other channels constant, +1 unit of spend in a channel changes predicted units sold by the coefficient amount.")

# ---------- What-if Prediction ----------
st.subheader("What-if prediction")
cols = st.columns(2)
inputs = {}

for i, col in enumerate(FEATURES):
    c = cols[i % 2]
    # Use observed min/max/median to set sensible slider bounds
    cmin = float(X[col].min())
    cmax = float(X[col].max())
    default = float(X[col].median())
    step = max((cmax - cmin) / 100.0, 1e-6)  # avoid zero step
    inputs[col] = c.slider(f"{col} Spend", cmin, cmax, default, step=step)

row = np.array([[inputs[f] for f in FEATURES]])
y_hat = model.predict(row)[0]
st.success(f"**Predicted Products Sold: {y_hat:.0f} units**")

# ---------- Optional: Quick scenario comparison ----------
with st.expander("Compare an alternative budget plan"):
    cols2 = st.columns(2)
    inputs2 = {}
    for i, col in enumerate(FEATURES):
        c = cols2[i % 2]
        cmin = float(X[col].min()); cmax = float(X[col].max())
        default = float(X[col].median())
        step = max((cmax - cmin) / 100.0, 1e-6)
        inputs2[col] = c.slider(f"[Alt] {col} Spend", cmin, cmax, default, step=step, key=f"alt_{col}")
    row2 = np.array([[inputs2[f] for f in FEATURES]])
    y_hat2 = model.predict(row2)[0]
    delta = y_hat2 - y_hat
    st.info(f"Alt plan predicts **{y_hat2:.0f}** units  (Δ **{delta:+.0f}** vs current).")
