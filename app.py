import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Restaurant Revenue Predictor",
    page_icon="🍽️",
    layout="centered",
)

# ------------------------------------------------------------------
# Load model artifacts (produced by the notebook)
#   - restaurant_model.pkl  -> trained RandomForestClassifier
#   - encoders.pkl          -> dict of fitted LabelEncoders for
#                              ["Location", "Cuisine", "Parking Availability"]
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("restaurant_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders


try:
    model, encoders = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `restaurant_model.pkl` and/or `encoders.pkl`.\n\n"
        "Run the training notebook first (it saves these two files with "
        "`pickle.dump`), then place them in the same folder as this app.py "
        "before launching Streamlit."
    )
    st.stop()

# The exact column order the model was trained on
FEATURE_ORDER = [
    "Location",
    "Cuisine",
    "Rating",
    "Seating Capacity",
    "Average Meal Price",
    "Marketing Budget",
    "Social Media Followers",
    "Chef Experience Years",
    "Number of Reviews",
    "Avg Review Length",
    "Ambience Score",
    "Service Quality Score",
    "Parking Availability",
    "Weekend Reservations",
    "Weekday Reservations",
]

CATEGORICAL_COLS = ["Location", "Cuisine", "Parking Availability"]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🍽️ Restaurant Revenue Predictor")
st.write(
    "Predicts whether a restaurant's revenue is likely to be **High** or "
    "**Low** (relative to the median revenue in the training data), based "
    "on restaurant attributes. Powered by the Random Forest model from the "
    "training notebook."
)

st.divider()

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Restaurant Details")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", options=list(encoders["Location"].classes_))
        cuisine = st.selectbox("Cuisine", options=list(encoders["Cuisine"].classes_))
        parking = st.selectbox(
            "Parking Availability",
            options=list(encoders["Parking Availability"].classes_),
        )
        rating = st.slider("Rating", 1.0, 5.0, 3.5, 0.1)
        seating_capacity = st.number_input("Seating Capacity", min_value=1, value=50)
        avg_meal_price = st.number_input(
            "Average Meal Price ($)", min_value=0.0, value=45.0, step=0.5
        )
        marketing_budget = st.number_input(
            "Marketing Budget ($)", min_value=0, value=2500
        )
        social_media_followers = st.number_input(
            "Social Media Followers", min_value=0, value=20000
        )

    with col2:
        chef_experience = st.number_input(
            "Chef Experience (Years)", min_value=0, value=10
        )
        num_reviews = st.number_input("Number of Reviews", min_value=0, value=200)
        avg_review_length = st.number_input(
            "Avg Review Length (characters)", min_value=0.0, value=150.0
        )
        ambience_score = st.slider("Ambience Score", 0.0, 10.0, 5.0, 0.1)
        service_quality = st.slider("Service Quality Score", 0.0, 10.0, 5.0, 0.1)
        weekend_reservations = st.number_input(
            "Weekend Reservations", min_value=0, value=25
        )
        weekday_reservations = st.number_input(
            "Weekday Reservations", min_value=0, value=15
        )

    submitted = st.form_submit_button("Predict Revenue Class", use_container_width=True)

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
if submitted:
    raw_input = {
        "Location": location,
        "Cuisine": cuisine,
        "Rating": rating,
        "Seating Capacity": seating_capacity,
        "Average Meal Price": avg_meal_price,
        "Marketing Budget": marketing_budget,
        "Social Media Followers": social_media_followers,
        "Chef Experience Years": chef_experience,
        "Number of Reviews": num_reviews,
        "Avg Review Length": avg_review_length,
        "Ambience Score": ambience_score,
        "Service Quality Score": service_quality,
        "Parking Availability": parking,
        "Weekend Reservations": weekend_reservations,
        "Weekday Reservations": weekday_reservations,
    }

    input_df = pd.DataFrame([raw_input])

    # Apply the same LabelEncoders used at training time
    for col in CATEGORICAL_COLS:
        input_df[col] = encoders[col].transform(input_df[col])

    # Ensure correct column order for the model
    input_df = input_df[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success(f"📈 Predicted: **High Revenue** (confidence: {proba[1]*100:.1f}%)")
    else:
        st.warning(f"📉 Predicted: **Low Revenue** (confidence: {proba[0]*100:.1f}%)")

    with st.expander("Show class probabilities"):
        st.write(
            pd.DataFrame(
                {"Class": ["Low Revenue (0)", "High Revenue (1)"], "Probability": proba}
            )
        )

    with st.expander("Show input sent to model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "Model: Random Forest Classifier trained to predict whether revenue is "
    "above or below the dataset median. This is a classification of revenue "
    "tier, not an exact dollar-amount forecast."
)
