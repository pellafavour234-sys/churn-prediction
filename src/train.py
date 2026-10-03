"""
Customer Churn Prediction - Training Pipeline
---------------------------------------------
Loads data, cleans it, builds a preprocessing + SMOTE + model pipeline,
compares models by F1-score, tunes the best one, evaluates it, and
saves the final trained pipeline to disk.

Run from the project root:
    python src/train.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_model.pkl"
OUTPUTS_DIR = "outputs"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_and_clean_data(url):
    df = pd.read_csv(url)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def plot_eda(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.countplot(data=df, x="Churn", ax=axes[0])
    axes[0].set_title("Churn Count")

    sns.barplot(data=df, x="Contract", y="Churn", ax=axes[1])
    axes[1].set_title("Churn Rate by Contract")
    axes[1].tick_params(axis="x", rotation=20)

    sns.boxplot(data=df, x="Churn", y="tenure", ax=axes[2])
    axes[2].set_title("Tenure vs Churn")

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/churn_eda.png")
    plt.close()


def build_preprocessor(categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, categorical_features),
    ])


def compare_models(preprocessor, X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
    }

    print("\n--- Cross-validation (F1, since classes are imbalanced) ---")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        print(f"{name}: mean F1={scores.mean():.4f}, std={scores.std():.4f}")


def tune_model(preprocessor, X_train, y_train):
    # imblearn's Pipeline (not sklearn's) so SMOTE can be a step --
    # it only runs during .fit() on training folds, never during .predict()
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nBest params:", grid_search.best_params_)
    print("Best CV F1 score:", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\n--- Final Test Results ---")
    print(classification_report(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"]).plot()
    plt.title("Confusion Matrix - Best Model")
    plt.savefig(f"{OUTPUTS_DIR}/churn_confusion_matrix.png")
    plt.close()


def main():
    df = load_and_clean_data(DATA_URL)
    plot_eda(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    categorical_features = [col for col in X.columns if col not in NUMERIC_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(categorical_features)
    compare_models(preprocessor, X_train, y_train)

    best_model = tune_model(preprocessor, X_train, y_train)
    evaluate_model(best_model, X_test, y_test)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
