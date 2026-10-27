
from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    departments = ["IT", "HR", "Sales", "Finance", "Marketing", "Operations"]
    education_levels = ["Bachelor", "Master", "PhD"]
    job_levels = ["Junior", "Mid", "Senior", "Lead"]

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(1, n_samples + 1)],
        "age": rng.integers(21, 61, n_samples),
        "department": rng.choice(departments, n_samples),
        "education": rng.choice(
            education_levels, n_samples, p=[0.55, 0.35, 0.10]
        ),
        "job_level": rng.choice(
            job_levels, n_samples, p=[0.30, 0.35, 0.25, 0.10]
        ),
        "experience_years": rng.integers(0, 31, n_samples),
        "training_hours": rng.integers(0, 101, n_samples),
        "projects_count": rng.integers(1, 16, n_samples),
        "on_time_delivery_rate": rng.uniform(0.50, 1.00, n_samples).round(2),
        "avg_task_delay_days": rng.uniform(0, 10, n_samples).round(2),
        "unplanned_absences": rng.integers(0, 16, n_samples),
        "peer_feedback_score": rng.uniform(1, 5, n_samples).round(1),
        "manager_score": rng.uniform(1, 5, n_samples).round(1),
        "previous_performance_score": rng.uniform(40, 100, n_samples).round(1),
    })

    # Create an illustrative score from synthetic features.
    score = (
        25 * df["on_time_delivery_rate"]
        + 5 * df["peer_feedback_score"]
        + 5 * df["manager_score"]
        + 0.15 * df["training_hours"]
        + 0.10 * df["projects_count"]
        + 0.15 * df["previous_performance_score"]
        - 0.8 * df["avg_task_delay_days"]
        - 0.3 * df["unplanned_absences"]
        + rng.normal(0, 5, n_samples)
    )

    # Convert the score into three illustrative categories.
    df["performance_band"] = pd.cut(
        score,
        bins=[-np.inf, 55, 70, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # Add a small amount of missing data for cleaning practice.
    for column in [
        "training_hours",
        "peer_feedback_score",
        "avg_task_delay_days",
    ]:
        missing_indices = rng.choice(
            df.index, size=max(1, int(0.02 * n_samples)), replace=False
        )
        df.loc[missing_indices, column] = np.nan

    output_path = Path("data") / "employee_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nPerformance category counts:")
    print(df["performance_band"].value_counts())


if __name__ == "__main__":
    generate_data()
