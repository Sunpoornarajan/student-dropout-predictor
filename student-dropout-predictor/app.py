import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap

# ---------------- LOAD MODEL ----------------
with open("data/model/dropout_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 AI-Based Student Dropout Predictor")
st.write("Enter student details to predict dropout risk")

# ---------------- USER INPUT ----------------
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal = st.slider("Internal Marks", 0, 100, 60)
gpa = st.slider("Previous GPA", 0.0, 10.0, 6.5)
assignments = st.slider("Assignments Completed", 0, 10, 5)
behavior = st.slider("Behavior Score", 0, 10, 7)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_df = pd.DataFrame(
        [[attendance, internal, gpa, assignments, behavior]],
        columns=[
            "attendance",
            "internal_marks",
            "previous_gpa",
            "assignments",
            "behavior_score"
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.3:
        risk = "🟢 Low Risk"
    elif probability < 0.6:
        risk = "🟡 Medium Risk"
    else:
        risk = "🔴 High Risk"

    st.success(f"Dropout Risk: {risk}")
    st.info(f"Dropout Probability: {probability:.2f}")

    # ---------------- SHAP EXPLANATION ----------------
    st.subheader("🔍 Student-Specific Risk Explanation")
    st.caption("How each feature influenced THIS student's prediction")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Normalize SHAP output shape
    shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]
    elif shap_array.ndim == 2:
        shap_array = shap_array

    shap_df = pd.DataFrame(
        shap_array,
        columns=input_df.columns
    )

    st.bar_chart(shap_df.iloc[0])

# ---------------- GLOBAL FEATURE IMPORTANCE ----------------
st.subheader("📊 Global Feature Importance (Training Data)")
st.caption("Overall impact of features across all students")

importance_df = pd.DataFrame({
    "Feature": [
        "attendance",
        "internal_marks",
        "previous_gpa",
        "assignments",
        "behavior_score"
    ],
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
