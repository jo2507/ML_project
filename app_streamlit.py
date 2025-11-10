import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

scaler = joblib.load("models/scaler.joblib")

models = {
    "KNN": joblib.load("models/knn.joblib"),
    "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
    "Naive Bayes": joblib.load("models/naive_bayes.joblib"),
    "Decision Tree": joblib.load("models/decision_tree.joblib"),
    "Random Forest": joblib.load("models/random_forest.joblib"),
    "AdaBoost": joblib.load("models/adaboost.joblib"),
    "Gradient Boosting": joblib.load("models/gradient_boosting.joblib"),
    "XGBoost": joblib.load("models/xgboost.joblib"),
    "SVM": joblib.load("models/svm.joblib"),
}

results = joblib.load("models/results.joblib")

import streamlit as st
import numpy as np

st.title("💖 Heart Disease Prediction App")

model_options = [f"{name} – Accuracy: {results[name]*100:.2f}%" for name in models.keys()]

selected_option = st.selectbox(
    "Select a Machine Learning Model:",
    model_options
)

model_choice = selected_option.split(" –")[0]

st.write(f"🔍 You selected: **{model_choice}** model")

st.header("Enter patient details")

age = st.number_input("Age", 1, 120, 45, key="age_input")
sex = st.selectbox("Sex", ("Male", "Female"), key="sex_input")
cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 0, key="cp_input")
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120, key="trestbps_input")
chol = st.number_input("Cholesterol", 100, 400, 200, key="chol_input")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"), key="fbs_input")
restecg = st.number_input("Rest ECG (0–2)", 0, 2, 1, key="restecg_input")
thalach = st.number_input("Max Heart Rate", 60, 220, 150, key="thalach_input")
exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"), key="exang_input")
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, key="oldpeak_input")
slope = st.number_input("Slope (0–2)", 0, 2, 1, key="slope_input")
ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0, key="ca_input")
thal = st.number_input("Thal (0–3)", 0, 3, 1, key="thal_input")


sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

features_scaled = scaler.transform(features)

if st.button("Predict"):
    model = models[model_choice]
    prediction = model.predict(features_scaled)[0]
    if prediction == 1:
        st.error("🚨 The patient is likely to have heart disease.")
    else:
        st.success("💚 The patient is unlikely to have heart disease.")

st.title("💖 Heart Disease Prediction App")
st.write("Enter patient details below to check the likelihood of heart disease.")

age = st.number_input("Age", 1, 120, 40)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [1, 0])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

model_choice = st.selectbox("Choose a model", list(models.keys()))

if st.button("🔍 Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    features_scaled = scaler.transform(features)
    model = models[model_choice]
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts that this patient is likely to have heart disease.")
    else:
        st.success("✅ The model predicts that this patient is unlikely to have heart disease.")
# --- Model Comparison Chart ---
import pandas as pd
import matplotlib.pyplot as plt

st.subheader("📊 Model Accuracy Comparison")

# Create a DataFrame from results
results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
results_df["Accuracy"] = results_df["Accuracy"] * 100  # convert to %

# Create a bar chart
fig, ax = plt.subplots()
ax.barh(results_df["Model"], results_df["Accuracy"])
ax.set_xlabel("Accuracy (%)")
ax.set_ylabel("Model")
ax.set_title("Model Performance Comparison")

st.pyplot(fig)

st.subheader("📋 Model Accuracy Table")

st.dataframe(results_df.style.format({"Accuracy": "{:.2f}"}))
