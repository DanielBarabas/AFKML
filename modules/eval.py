import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from modules.data_wrangling import (
    create_model_df,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.preprocessing import LabelEncoder
import altair as alt


# This won't be needed in the final version
def preproc(problem_type):
    if problem_type == "Binary classification":
        df = pd.read_csv(
            "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
        )
        res_df = pd.DataFrame({"Variable": ["sex"], "Encoding": ["One-Hot"]})
        X = create_model_df(res_df, df, "DRK_YN", "Categorical")
        le = LabelEncoder()
        y = le.fit_transform(df["DRK_YN"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBClassifier(n_estimators=30, max_depth=2)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred_binary = model.predict(X_test)

    elif problem_type == "Multiclass classification":
        df = pd.read_csv(
            "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
        )
        res_df = pd.DataFrame(
            {"Variable": ["sex", "DRK_YN"], "Encoding": ["One-Hot", "One-Hot"]}
        )
        X = create_model_df(res_df, df, "weight", "Categorical")

        df["weight"] = df["weight"].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(df["weight"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBClassifier(n_estimators=30, max_depth=2)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred_binary = model.predict(X_test)

    else:
        df = pd.read_csv(
            "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
        )
        res_df = pd.DataFrame(
            {"Variable": ["sex", "DRK_YN"], "Encoding": ["One-Hot", "One-Hot"]}
        )
        X = create_model_df(res_df, df, "hemoglobin", "Numeric")

        y = df["hemoglobin"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBRegressor(n_estimators=30, max_depth=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_binary = y_pred

    return (model, y_pred, y_pred_binary, X_test, y_test, le)


# added leading underscore to model param, since streamlit couldn't hash it
@st.cache_data
def predict1(_model, X_test, problem_type):
    if problem_type == "Regression":
        y_pred = _model.predict(X_test)
        y_pred_binary = "anyad"
    else:
        y_pred = _model.predict_proba(X_test)
        y_pred_binary = _model.predict(X_test)

    return y_pred, y_pred_binary


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


@st.cache_resource(experimental_allow_widgets=True)
def binary_metric_table(y_pred_binary, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    specificity = tn / (tn + fp)
    precision = tp / (fp + tp)
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred_binary)
    negative_predictive_value = tn / (tn + fn)
    recall = tp / (tp + fn)
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


### TODO Kell majd egy multiclass precison-recall, és metrikák!


############################# Regression case ######################################
@st.cache_resource(experimental_allow_widgets=True)
def reg_table(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    metric_df = pd.DataFrame({"Metric": ["Mean Squared Error"], "Value": [mse]})
    return metric_df


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
