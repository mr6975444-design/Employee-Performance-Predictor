
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/employee_data.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
IMAGE_DIR = Path("images")


def train_and_evaluate():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    target = "performance_band"

    # Do not use employee_id or the target as input features.
    X = df.drop(columns=[target, "employee_id"])
    y = df[target]

    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Handle missing numeric and categorical values
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    # Build the complete machine-learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    # Keep the test set separate from model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate on unseen test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nMODEL EVALUATION")
    print("=" * 50)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro-F1 score: {macro_f1:.3f}")

    report = classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0,
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"],
        zero_division=0,
    ))

    # Save evaluation metrics
    pd.DataFrame([{
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # Save predictions for inspection
    predictions = X_test.copy()
    predictions["actual_performance_band"] = y_test
    predictions["predicted_performance_band"] = y_pred
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Plot confusion matrix
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save feature importance
    fitted_preprocessor = model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False
    )

    top_features = importance_df.head(15).sort_values(
        "importance", ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # Save the complete pipeline, including preprocessing
    joblib.dump(model, MODEL_DIR / "employee_performance_model.joblib")

    print("\nModel and reports saved successfully.")
    print(f"Model: {MODEL_DIR / 'employee_performance_model.joblib'}")
    print(f"Metrics: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"Charts: {IMAGE_DIR.resolve()}")


if __name__ == "__main__":
    train_and_evaluate()
