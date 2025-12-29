import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig = shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    return fig
