"""Streamlit interface for the tourism purchase propensity model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Visit with Us - Package Prediction", layout="centered")


@st.cache_resource
def load_model():
    model_path = ROOT / "models" / "best_model.joblib"
    metadata_path = ROOT / "models" / "model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Complete the training workflow first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


st.title("Visit with Us")
st.subheader("Travel package purchase prediction")
st.write(
    "Enter a customer profile to estimate the probability of a historical "
    "travel package purchase. The training data does not separately identify "
    "Wellness package purchases."
)

try:
    model, metadata = load_model()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

categories = metadata["categories"]

with st.form("customer_profile"):
    st.markdown("### Customer details")
    left, right = st.columns(2)

    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=38)
        city_tier = st.selectbox("City tier", options=[1, 2, 3])
        occupation = st.selectbox("Occupation", categories["Occupation"])
        gender = st.selectbox("Gender", categories["Gender"])
        people_visiting = st.number_input(
            "Number of travelers", min_value=1, max_value=15, value=3
        )
        hotel_rating = st.selectbox("Preferred hotel stars", [3, 4, 5])
        marital_status = st.selectbox(
            "Marital status", categories["MaritalStatus"]
        )

    with right:
        annual_trips = st.number_input(
            "Trips per year", min_value=0, max_value=30, value=2
        )
        passport = st.selectbox("Valid passport", ["No", "Yes"])
        owns_car = st.selectbox("Owns a car", ["No", "Yes"])
        children_visiting = st.number_input(
            "Children under five", min_value=0, max_value=12, value=0
        )
        designation = st.selectbox("Designation", categories["Designation"])
        income = st.number_input(
            "Monthly income (dataset units)",
            min_value=0.0,
            max_value=2_000_000.0,
            value=22_000.0,
            step=1_000.0,
        )

    submitted = st.form_submit_button(
        "Predict purchase probability", use_container_width=True
    )

if submitted:
    customer = {
        "Age": age,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": people_visiting,
        "PreferredPropertyStar": hotel_rating,
        "MaritalStatus": marital_status,
        "NumberOfTrips": annual_trips,
        "Passport": int(passport == "Yes"),
        "OwnCar": int(owns_car == "Yes"),
        "NumberOfChildrenVisiting": children_visiting,
        "Designation": designation,
        "MonthlyIncome": income,
    }
    customer_data = pd.DataFrame([customer], columns=metadata["features"])
    probability = float(model.predict_proba(customer_data)[0, 1])

    st.divider()
    st.metric("Predicted purchase probability", f"{probability:.1%}")
    st.progress(int(round(probability * 100)))
    st.caption(
        "A 50% cutoff is shown for reference. Marketing teams should set "
        "their outreach threshold using campaign costs and capacity."
    )
    with st.expander("Customer inputs"):
        st.dataframe(customer_data, hide_index=True, use_container_width=True)

metrics_path = ROOT / "reports" / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with st.expander("Model performance on the test set"):
        st.write("Selected model:", metrics["selected_algorithm"])
        st.write("ROC-AUC:", f"{metrics['holdout_roc_auc']:.3f}")
        st.write(
            "Average precision:", f"{metrics['holdout_average_precision']:.3f}"
        )
        st.write("Top 20% ranking lift:", f"{metrics['top_20pct_lift']:.2f}x")