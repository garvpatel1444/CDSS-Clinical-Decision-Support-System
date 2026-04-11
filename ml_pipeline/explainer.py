import shap


def generate_shap_explanation(model, X_sample):
    """
    Generates SHAP explanation depending on model type.
    """

    # If model is Pipeline, extract final estimator
    if hasattr(model, "named_steps"):
        base_model = model.named_steps["model"]
        preprocessor = model.named_steps["scaler"]
        X_sample = preprocessor.transform(X_sample)
    else:
        base_model = model

    # Tree-based models
    if "XGB" in type(base_model).__name__ or "Forest" in type(base_model).__name__:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X_sample)

    # Linear models
    else:
        explainer = shap.LinearExplainer(base_model, X_sample)
        shap_values = explainer.shap_values(X_sample)

    return shap_values
