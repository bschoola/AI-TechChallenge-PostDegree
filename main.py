import seaborn as sns

from src.data_loader import load_data
from src.evaluation import cross_validate_models, evaluate_model
from src.explainability import feature_importance, shap_analysis
from src.models import build_models
from src.preprocessing import prepare_features, split_data

sns.set(style="whitegrid")


def main() -> None:
    print("=== Cancer Prediction Pipeline ===")

    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models()

    print("\n=== Model Evaluation ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test)

    cross_validate_models(models, X, y)

    print("\n=== SHAP Analysis (Logistic Regression) ===")
    pipe_lr = models["Logistic Regression"]
    shap_analysis(pipe_lr, X_train, X_test)
    feature_importance(pipe_lr, X.columns.tolist())


if __name__ == "__main__":
    main()
