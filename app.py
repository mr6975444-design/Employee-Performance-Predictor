
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "employee_performance_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_DIR = BASE_DIR / "images"

st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Performance Predictor")
st.write(
    "An educational machine-learning dashboard using synthetic employee data."
)

st.warning(
    "Educational demonstration only. Predictions are based on invented "
    "synthetic labels and must not be used for real hiring, promotion, "
    "pay, disciplinary, or other employment decisions."
)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python -m src.train_model")
    st.stop()

model = joblib.load(MODEL_PATH)

# Dashboard navigation
page = st.sidebar.radio(
    "Navigate",
    ["Performance Prediction", "Model Evaluation", "Project Information"],
)

if page == "Performance Prediction":
    st.header("Predict a Synthetic Performance Category")
    st.write("Enter sample information below.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 21, 60, 30)
            department = st.selectbox(
                "Department",
                ["IT", "HR", "Sales", "Finance", "Marketing", "Operations"],
            )
            education = st.selectbox(
                "Education",
                ["Bachelor", "Master", "PhD"],
            )
            job_level = st.selectbox(
                "Job Level",
                ["Junior", "Mid", "Senior", "Lead"],
            )
            experience_years = st.slider(
                "Years of Experience", 0, 30, 5
            )
            training_hours = st.slider(
                "Training Hours", 0, 100, 30
            )
            projects_count = st.slider(
                "Number of Projects", 1, 15, 5
            )

        with col2:
            on_time_delivery_rate = st.slider(
                "On-Time Delivery Rate",
                min_value=0.50,
                max_value=1.00,
                value=0.80,
                step=0.01,
            )
            avg_task_delay_days = st.slider(
                "Average Task Delay (Days)",
                0.0, 10.0, 2.0, step=0.1
            )
            unplanned_absences = st.slider(
                "Unplanned Absences", 0, 15, 2
            )
            peer_feedback_score = st.slider(
                "Peer Feedback Score", 1.0, 5.0, 3.5, step=0.1
            )
            manager_score = st.slider(
                "Manager Score", 1.0, 5.0, 3.5, step=0.1
            )
            previous_performance_score = st.slider(
                "Previous Performance Score",
                40.0, 100.0, 70.0, step=0.5
            )

        submitted = st.form_submit_button("Predict Category")

    if submitted:
        input_data = pd.DataFrame([{
            "age": age,
            "department": department,
            "education": education,
            "job_level": job_level,
            "experience_years": experience_years,
            "training_hours": training_hours,
            "projects_count": projects_count,
            "on_time_delivery_rate": on_time_delivery_rate,
            "avg_task_delay_days": avg_task_delay_days,
            "unplanned_absences": unplanned_absences,
            "peer_feedback_score": peer_feedback_score,
            "manager_score": manager_score,
            "previous_performance_score": previous_performance_score,
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        st.metric("Predicted Synthetic Category", str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.named_steps["classifier"].classes_

            probability_df = pd.DataFrame({
                "Category": classes,
                "Model Probability": probabilities,
            }).sort_values("Model Probability", ascending=False)

            st.write("Model-estimated class probabilities")
            st.bar_chart(
                probability_df.set_index("Category")
            )

        st.caption(
            "Probabilities represent the model's estimates for this "
            "synthetic dataset; they are not validated measures of "
            "a person's actual performance."
        )

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    metrics_path = OUTPUT_DIR / "metrics.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Macro-F1 Score", f"{metrics['macro_f1']:.3f}")
        col3.metric("Test Records", int(metrics["testing_records"]))
    else:
        st.info("Run the training script to generate evaluation metrics.")

    for image_name, title in [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("feature_importance.png", "Feature Importance"),
        ("performance_distribution.png", "Synthetic Category Distribution"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("training_hours_by_category.png", "Training Hours by Category"),
    ]:
        image_path = IMAGE_DIR / image_name

        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)

    report_path = OUTPUT_DIR / "classification_report.csv"

    if report_path.exists():
        st.subheader("Classification Report")
        report = pd.read_csv(report_path, index_col=0)
        st.dataframe(report, use_container_width=True)

    importance_path = OUTPUT_DIR / "feature_importance.csv"

    if importance_path.exists():
        st.subheader("Feature Importance Table")
        importance = pd.read_csv(importance_path)
        st.dataframe(importance, use_container_width=True)

elif page == "Project Information":
    st.header("About This Project")

    st.markdown("""
    **Project goals**
    - Generate a synthetic employee dataset.
    - Inspect data quality and missing values.
    - Explore distributions and numeric correlations.
    - Train a Random Forest classification model.
    - Evaluate the model using accuracy, macro-F1, and a confusion matrix.
    - Present predictions and evaluation charts in a dashboard.

    **Technology stack**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Matplotlib and Seaborn
    - Streamlit
    - Joblib

    **Limitations**

    The target categories were created using an invented scoring formula.
    Model results measure how well the classifier learns patterns in this
    synthetic dataset. They do not establish valid or fair measures of
    real-world employee performance.
    """)
