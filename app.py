import streamlit as st
import pandas as pd
import joblib
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.joblib")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered"
)

# Title
st.title("Student Performance Predictor")
st.write("Enter the student's information to predict the final score.")

st.divider()

# Inputs
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0,
    step=1.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

st.divider()

# Prediction
if st.button("Predict Final Score", use_container_width=True):

    new_student = pd.DataFrame([
        {
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignment_score": assignment_score
        }
    ])

    prediction = model.predict(new_student)[0]

    st.success(f"Predicted Final Score: {prediction:.2f} / 100")

    st.info(
        "This prediction is based on the trained machine learning model. "
        "It should be treated as an estimate, not a guaranteed result."
    )

st.divider()

st.caption("Student Performance Predictor • Machine Learning Project")
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

