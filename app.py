import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("titanic_svm_probability_model.joblib")

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🚢 Titanic Survival Predictor")

st.caption(
    "A machine learning application using SVM "
    "with calibrated probability estimates."
)

st.divider()

# --------------------------------------------------
# Passenger Information
# --------------------------------------------------

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Sex",
        ["female", "male"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=80.0,
        step=1.0
    )

with col2:

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["C", "Q", "S"]
    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Predict Survival",
    type="primary",
    use_container_width=True
):

    passenger = pd.DataFrame([{
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }])

    prediction = model.predict(passenger)[0]

    probability = model.predict_proba(passenger)[0]

    survival_probability = probability[1] * 100
    death_probability = probability[0] * 100

    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.subheader("Prediction")

    if prediction == 1:

        st.success(
            "Prediction: Survived"
        )

    else:

        st.error(
            "Prediction: Did not survive"
        )

    # --------------------------------------------------
    # Probability Metrics
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Survival Probability",
            f"{survival_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Non-Survival Probability",
            f"{death_probability:.2f}%"
        )

    # --------------------------------------------------
    # Probability Visualization
    # --------------------------------------------------

    st.subheader("Probability")

    st.progress(
        survival_probability / 100,
        text=f"Survival: {survival_probability:.2f}%"
    )

    st.progress(
        death_probability / 100,
        text=f"Non-Survival: {death_probability:.2f}%"
    )

    # --------------------------------------------------
    # Passenger Summary
    # --------------------------------------------------

    st.subheader("Passenger Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Class",
            str(pclass)
        )

    with summary_col2:

        st.metric(
            "Sex",
            sex.title()
        )

    with summary_col3:

        st.metric(
            "Age",
            f"{age:.0f}"
        )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Fare",
            f"{fare:.2f}"
        )

    with summary_col2:

        st.metric(
            "Siblings / Spouses",
            str(sibsp)
        )

    with summary_col3:

        st.metric(
            "Parents / Children",
            str(parch)
        )

    # --------------------------------------------------
    # Model Information
    # --------------------------------------------------

    st.subheader("Model Information")

    st.write("**Algorithm:** Support Vector Machine (SVM)")
    st.write("**Probability:** CalibratedClassifierCV")
    st.write("**Features:** 7 passenger features")
    st.write("**Test Accuracy:** 81.56%")
    st.write("**ROC-AUC:** 0.8401")

    # --------------------------------------------------
    # Disclaimer
    # --------------------------------------------------

    st.warning(
        "This prediction is generated by a machine learning "
        "model trained on historical Titanic passenger data. "
        "The probabilities are estimates, not guarantees. "
        "This application is for educational and demonstration "
        "purposes only."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

