import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

# Columns a user must provide (the original dataset columns, minus the ID columns and the target)
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]

FRIENDLY = {
    "CreditScore": "Credit score", "Age": "Age", "Tenure": "Years with bank",
    "Balance": "Account balance", "EstimatedSalary": "Estimated salary",
    "NumOfProducts": "Number of products", "Geography_Germany": "Lives in Germany",
    "Geography_Spain": "Lives in Spain", "Gender_Male": "Male customer",
    "HasCrCard": "Has credit card", "IsActiveMember": "Is an active member",
    "ZeroBalance": "Zero account balance",
}


@st.cache_resource
def load_model():
    pipe = joblib.load("churn_model.joblib")
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    return pipe, explainer


pipe, explainer = load_model()


def make_features(df):
    """Same feature engineering as in the notebook."""
    df = df[FEATURES].copy()
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)
    return df


def score(df):
    return pipe.predict_proba(make_features(df))[:, 1]


def explain(df):
    """SHAP contributions for a one-row DataFrame, sorted by size."""
    prep = pipe.named_steps["prep"]
    names = [n.split("__")[1] for n in prep.get_feature_names_out()]
    transformed = pd.DataFrame(prep.transform(make_features(df)), columns=names)
    values = explainer(transformed).values[0]
    return pd.Series(values, index=names).sort_values(key=np.abs, ascending=False)


# ---------------------------------------------------------------- sidebar
st.title("🏦 Bank Customer Churn Predictor")
st.caption("Predicts how likely a customer is to leave the bank, and why.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.55, 0.01,
        help="Customers with a churn probability at or above this value are flagged as high risk. "
             "Lower = flag more customers (catch more churners, more false alarms).",
    )
    st.markdown(
        "**Model:** XGBoost\n\n"
        "**Default threshold 0.55** was chosen so that about 1 in 4 customers is flagged, "
        "which reached roughly 70% of churners in testing."
    )

tab_single, tab_batch = st.tabs(["Single customer", "Batch (CSV upload)"])

# ---------------------------------------------------------- single customer
with tab_single:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.number_input("Credit score", 300, 850, 650)
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with c2:
        age = st.number_input("Age", 18, 100, 45)
        tenure = st.number_input("Tenure (years with bank)", 0, 10, 3)
        balance = st.number_input("Account balance", 0.0, 300000.0, 80000.0, step=1000.0)
    with c3:
        products = st.number_input("Number of products", 1, 4, 1)
        has_card = st.selectbox("Has credit card?", ["Yes", "No"])
        active = st.selectbox("Active member?", ["Yes", "No"])
        salary = st.number_input("Estimated salary", 0.0, 250000.0, 60000.0, step=1000.0)

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "CreditScore": credit_score, "Geography": geography, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": products, "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(active == "Yes"), "EstimatedSalary": salary,
        }])
        prob = float(score(row)[0])
        high_risk = prob >= threshold

        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Risk level", "HIGH RISK" if high_risk else "Lower risk")
        if high_risk:
            st.error("This customer is likely to leave. Consider a retention action.")
        else:
            st.success("This customer is currently below the risk threshold.")

        contrib = explain(row)
        up = contrib[contrib > 0].head(3)
        down = contrib[contrib < 0].head(3)

        left, right = st.columns(2)
        with left:
            st.markdown("**Pushing towards leaving**")
            for k in up.index:
                st.write(f"🔺 {FRIENDLY.get(k, k)}")
            if up.empty:
                st.write("None")
        with right:
            st.markdown("**Pushing towards staying**")
            for k in down.index:
                st.write(f"🔻 {FRIENDLY.get(k, k)}")
            if down.empty:
                st.write("None")

        st.markdown("**Contribution of each factor** (positive = raises churn risk)")
        chart = contrib.head(8).rename(index=lambda k: FRIENDLY.get(k, k))
        st.bar_chart(chart)

# ------------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Score many customers at once")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            data["ChurnProbability"] = score(data)
            data["HighRisk"] = data["ChurnProbability"] >= threshold
            data = data.sort_values("ChurnProbability", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Customers", f"{len(data):,}")
            k2.metric("Flagged high risk", f"{int(data['HighRisk'].sum()):,}")
            k3.metric("Share flagged", f"{data['HighRisk'].mean():.1%}")

            st.dataframe(data, use_container_width=True)
            st.download_button(
                "Download results as CSV",
                data.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
