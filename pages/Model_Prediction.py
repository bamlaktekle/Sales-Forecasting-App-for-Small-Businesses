# pages/Model_Prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🔮 Model & Prediction")

DATA_PATH = Path("data/Advertising_Data.csv")
ALL_FEATURES = ["TV","Billboards","Google_Ads","Social_Media","Influencer_Marketing","Affiliate_Marketing"]
TARGET = "Product_Sold"

# ---------- Load & prepare data ----------
@st.cache_data(show_spinner=False)
def load_data(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    cols = ALL_FEATURES + [TARGET]
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df

if not DATA_PATH.exists():
    st.error(f"CSV not found at `{DATA_PATH.as_posix()}`.")
    st.stop()

df = load_data(DATA_PATH)

# ---------- Controls ----------
c1, c2 = st.columns(2)
with c1:
    test_size = st.slider("Test size (holdout %)", 10, 40, 20, step=5) / 100.0
with c2:
    seed = st.number_input("Random state", min_value=0, value=42, step=1)

chosen_features = st.multiselect(
    "Select channels to include",
    ALL_FEATURES,
    default=ALL_FEATURES
)

if len(chosen_features) == 0:
    st.warning("Select at least one feature to train the model.")
    st.stop()

X = df[chosen_features]
y = df[TARGET]

# ---------- Train / test split + model ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LinearRegression().fit(Xtr, ytr)

# ---------- Metrics ----------
pred = model.predict(Xte)
r2   = r2_score(yte, pred)
mae  = mean_absolute_error(yte, pred)
rmse = mean_squared_error(yte, pred) ** 0.5

st.subheader("Performance")
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("R²", f"{r2:.3f}")
mcol2.metric("MAE", f"{mae:,.2f}")
mcol3.metric("RMSE", f"{rmse:,.2f}")

# Optional: quick 5-fold CV MAE (on full data with chosen features)
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
cv_mae = -cross_val_score(LinearRegression(), X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
st.caption(f"5-fold CV MAE: **{cv_mae:,.2f}** (lower is better)")

# ---------- Evaluate model predictions (test set) ----------
st.markdown("### Evaluate model predictions (holdout)")

test_plot = pd.DataFrame({
    "idx": np.arange(len(yte)),
    "Actual": yte.reset_index(drop=True),
    "Predicted": pd.Series(pred)
})

# Make the two series easy to distinguish: Predicted as line, Actual as markers
fig, ax = plt.subplots()
ax.plot(test_plot["idx"], test_plot["Predicted"], label="Predicted", linewidth=2)
ax.scatter(test_plot["idx"], test_plot["Actual"], label="Actual", s=26, alpha=0.9)
ax.set_xlabel("Test sample index")
ax.set_ylabel("Products Sold")
ax.set_title("Actual vs Predicted (Test Set)")
ax.legend()
st.pyplot(fig, clear_figure=True)

# ---- Extra evaluation visuals your professor will expect ----
col_a, col_b = st.columns(2)

# Parity plot (y_true vs y_pred)
with col_a:
    fig, ax = plt.subplots()
    ax.scatter(yte, pred, s=24, alpha=0.85)
    lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Parity plot (ideal = dashed line)")
    st.pyplot(fig, clear_figure=True)

# Residuals vs Predicted
with col_b:
    residuals = yte.reset_index(drop=True) - pred
    fig, ax = plt.subplots()
    ax.scatter(pred, residuals, s=24, alpha=0.85)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")
    st.pyplot(fig, clear_figure=True)

# ---------- Coefficients (feature impact) ----------
st.markdown("### Channel impact (coefficients)")
coef_series = pd.Series(model.coef_, index=chosen_features).sort_values(ascending=True)
fig, ax = plt.subplots()
coef_series.plot(kind="barh", ax=ax)
ax.set_xlabel("Coefficient (Δ predicted units per 1 spend unit)")
ax.set_title("Linear regression coefficients")
st.pyplot(fig, clear_figure=True)
st.caption("Interpretation: holding other channels constant, +1 unit of spend in a channel changes predicted units sold by the coefficient amount.")

# ---------- What-if Prediction ----------
st.markdown("### What-if prediction")
cols = st.columns(2)
inputs = {}
for i, col in enumerate(chosen_features):
    c = cols[i % 2]
    cmin = float(X[col].min()); cmax = float(X[col].max())
    default = float(X[col].median())
    step = max((cmax - cmin) / 100.0, 1e-6)
    inputs[col] = c.slider(f"{col} Spend", cmin, cmax, default, step=step)

row = np.array([[inputs[f] for f in chosen_features]])
y_hat = model.predict(row)[0]
st.success(f"**Predicted Products Sold: {y_hat:.0f} units**")

# ---------- Optional: simple budget allocation helper ----------
with st.expander("Budget allocation helper (toy)"):
    total_budget = st.number_input(
        "Total ad budget (same units as data)",
        min_value=0.0,
        value=float(sum(inputs.values())),
        step=100.0
    )
    coefs = pd.Series(model.coef_, index=chosen_features)
    pos = coefs.clip(lower=0)
    if pos.sum() > 0:
        weights = pos / pos.sum()
        alloc = (weights * total_budget).round(2)
        st.write("Suggested allocation (proportional to positive impact):")
        st.dataframe(alloc.to_frame("Suggested_Spend"))
        st.caption("Use this as a starting point—not a final plan.")
    else:
        st.info("All coefficients are non-positive; unable to produce a weighted allocation.")
