import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
)
from sklearn.inspection import permutation_importance
import altair as alt


############################ Binary Case ####################################
@st.cache_resource(experimental_allow_widgets=True)
def binary_roc(y_pred, y_test):
    y_pred1 = [y[1] for y in y_pred]
    fpr, tpr, _ = roc_curve(y_test, y_pred1)
    roc_auc = auc(fpr, tpr)

    roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

    roc_curve_chart = (
        alt.Chart(roc_data)
        .mark_line(color="darkorange")
        .encode(
            x=alt.X("False Positive Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("True Positive Rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=["False Positive Rate", "True Positive Rate"],
        )
        .properties(title=f"ROC Curve (AUC = {roc_auc:.2f})", height=500, width=500)
    )

    diagonal = pd.DataFrame(
        {"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]}
    )
    diagonal_line = (
        alt.Chart(diagonal)
        .mark_line(color="navy", strokeDash=[5, 5])
        .encode(x="False Positive Rate", y="True Positive Rate")
    )

    final_chart = roc_curve_chart + diagonal_line
    return final_chart


@st.cache_resource(experimental_allow_widgets=True)
def binary_prec_recall(y_pred, y_test):
    y_pred1 = [y[1] for y in y_pred]
    precision, recall, _ = precision_recall_curve(y_test, y_pred1)
    pr_data = pd.DataFrame({"Recall": recall, "Precision": precision})
    pr_curve_chart = (
        alt.Chart(pr_data)
        .mark_line(color="blue")
        .encode(
            x=alt.X(
                "Recall",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
            ),
            y=alt.Y(
                "Precision",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
            ),
            tooltip=["Recall", "Precision"],
        )
        .properties(
            title=alt.TitleParams(text="Precision-Recall Curve", fontSize=20),
            width=500,
            height=500,
        )
    )
    return pr_curve_chart


@st.cache_resource(experimental_allow_widgets=True)
def binary_cm(y_pred_binary, y_test):
    cm = confusion_matrix(y_test, y_pred_binary)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    cm_df_melt = cm_df.reset_index().melt(id_vars="index")
    cm_df_melt.columns = ["Actual", "Predicted", "Count"]

    cm_chart = (
        alt.Chart(cm_df_melt)
        .mark_rect()
        .encode(
            x=alt.X(
                "Predicted:O",
                title="Predicted",
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
            ),
            y=alt.Y(
                "Actual:O",
                title="Actual",
                axis=alt.Axis(labelFontSize=14, titleFontSize=16),
            ),
            color=alt.Color(
                "Count:Q", legend=alt.Legend(labelFontSize=14, titleFontSize=16)
            ),
            tooltip=["Actual", "Predicted", "Count"],
        )
        .properties(
            title=alt.TitleParams(text="Confusion Matrix", fontSize=20),
            width=500,
            height=500,
        )
    )
    threshold = cm_df_melt["Count"].max() / 2.0
    cm_text = cm_chart.mark_text(baseline="middle", fontSize=40).encode(
        text="Count:Q",
        color=alt.condition(
            alt.datum.Count > threshold, alt.value("white"), alt.value("black")
        ),
    )

    return cm_chart + cm_text


@st.cache_data
def binary_metric_table(y_pred_binary, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0
    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred_binary)
    mse = mean_squared_error(y_test, y_pred_binary)

    metric_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
                "Sensitivity / Recall",
                "Specificity",
                "Precision",
                "Negative Predictive Value",
                "F1 score",
                "MSE",
            ],
            "Value": [
                accuracy,
                balanced_accuracy,
                recall,
                specificity,
                precision,
                negative_predictive_value,
                f1,
                mse,
            ],
        }
    )
    return metric_df


################################## Multiclass Case ####################################
@st.cache_resource(experimental_allow_widgets=True)
def multiclass_cm(y_pred_binary, y_test, _le):
    cm = confusion_matrix(y_test, y_pred_binary)
    cm_df = pd.DataFrame(
        cm,
        index=[
            f"True {i}" for i in _le.inverse_transform([i for i in range(cm.shape[0])])
        ],
        columns=[
            f"Pred {i}" for i in _le.inverse_transform([i for i in range(cm.shape[1])])
        ],
    )
    cm_long_df = cm_df.reset_index().melt(id_vars="index")
    cm_long_df.columns = ["True Label", "Predicted Label", "Count"]
    heatmap = (
        alt.Chart(cm_long_df)
        .mark_rect()
        .encode(
            x="Predicted Label:O",
            y="True Label:O",
            color="Count:Q",
            tooltip=["True Label", "Predicted Label", "Count"],
        )
        .properties(title="Confusion Matrix Heatmap", width=500, height=500)
    )
    return heatmap


@st.cache_resource(experimental_allow_widgets=True)
def multiclass_roc(y_pred, y_test, _le):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(_le.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    roc_data = pd.DataFrame(
        columns=["False Positive Rate", "True Positive Rate", "Class", "AUC"]
    )

    for i in range(len(_le.classes_)):
        temp_df = pd.DataFrame(
            {
                "False Positive Rate": fpr[i],
                "True Positive Rate": tpr[i],
                "Class": f"Class {_le.inverse_transform([i])[0]}",
                "AUC": roc_auc[i],
            }
        )
        roc_data = pd.concat([roc_data, temp_df])

    roc_data.reset_index(drop=True, inplace=True)

    roc_plot = (
        alt.Chart(roc_data)
        .mark_line()
        .encode(
            x="False Positive Rate:Q",
            y="True Positive Rate:Q",
            color="Class:N",
            tooltip=["Class", "AUC"],
        )
        .properties(title="One-vs-All ROC Curves", width=500, height=500)
    )

    diagonal = pd.DataFrame(
        {"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]}
    )
    diagonal_line = (
        alt.Chart(diagonal)
        .mark_line(color="navy", strokeDash=[5, 5])
        .encode(x="False Positive Rate", y="True Positive Rate")
    )

    return roc_plot + diagonal_line


def calculate_multiclass_metrics(y_true, y_pred, class_label):
    """
    Calculate one-vs-all metrics for a specific class.
    """
    y_true_ova = np.where(y_true == class_label, 1, 0)
    y_pred_ova = np.where(y_pred == class_label, 1, 0)

    tn, fp, fn, tp = confusion_matrix(y_true_ova, y_pred_ova).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0
    accuracy = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
    balanced_accuracy = balanced_accuracy_score(y_true_ova, y_pred_ova)
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true_ova, y_pred_ova)

    return {
        "Class": class_label,
        "Accuracy": accuracy,
        "Balanced accuracy": balanced_accuracy,
        "Recall": recall,
        "Specificity": specificity,
        "Precision": precision,
        "Negative Predictive Value": negative_predictive_value,
        "F1 score": f1,
    }


@st.cache_data
def create_multiclass_metric_table(y_test, y_pred, y_train):
    classes = np.unique(y_train)
    metrics = [
        calculate_multiclass_metrics(y_test, y_pred, class_label)
        for class_label in classes
    ]
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


############################# Regression case ######################################
@st.cache_data
def reg_table(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    metric_df = pd.DataFrame({"Metric": ["Mean Squared Error"], "Value": [mse]})
    return metric_df


@st.cache_resource(experimental_allow_widgets=True)
def reg_residuals(y_pred, y_test):
    residuals = list(y_test.iloc[:, 0].to_numpy()) - y_pred
    plot_df = pd.DataFrame(
        {
            "Actual": list(y_test.iloc[:, 0].to_numpy()),
            "Predicted": y_pred,
            "Residuals": list(residuals),
        }
    )
    residual_plot = (
        alt.Chart(plot_df.reset_index())
        .mark_circle(size=60)
        .encode(
            x="Predicted", y="Residuals", tooltip=["Actual", "Predicted", "Residuals"]
        )
        .properties(title="Residual Plot", width=500, height=500)
    )
    return residual_plot


########################## Same for all 3 cases ####################################
@st.cache_resource(experimental_allow_widgets=True)
def feature_importance(_model, X_test, y_test):
    imp = permutation_importance(_model, X_test, y_test, n_repeats=2, random_state=42)
    importances = pd.Series(imp.importances_mean)
    feature_importance_df = pd.DataFrame(
        {"Feature": X_test.columns, "Importance": importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )
    importance_chart = (
        alt.Chart(feature_importance_df)
        .mark_bar()
        .encode(
            x=alt.X("Importance", title="Feature Importance"),
            y=alt.Y("Feature", sort="-x", title="Feature"),
            tooltip=["Feature", "Importance"],
        )
        .properties(title="Feature Importance", height=500, width=500)
    )
    return importance_chart
def ass():
    pass