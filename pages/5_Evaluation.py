import streamlit as st
from modules.eval import (
    binary_roc,
    binary_prec_recall,
    binary_cm,
    binary_metric_table,
    feature_importance,
    multiclass_cm,
    multiclass_roc,
    reg_table,
    reg_residuals,
    create_multiclass_metric_table,
)


st.set_page_config(page_title="Model evaluation", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()
elif "model" not in st.session_state:
    st.write("Train a model first on the modelling page")
    st.stop()


st.title("Model evaluation")

# Biclass
if st.session_state["problem_type"] == "Binary classification":
    st.header("ROC curve")
    if st.toggle(label="Create ROC Curve", key="bi_roc"):
        st.altair_chart(
            binary_roc(st.session_state["y_pred"], st.session_state["y_test"])
        )

    st.header("Precision-Recall Chart")
    if st.toggle(label="Create Precision-Recall Chart", key="bi_prec_rec"):
        st.altair_chart(
            binary_prec_recall(st.session_state["y_pred"], st.session_state["y_test"])
        )

    st.header("Confusion matrix")
    if st.toggle(label="Create Confusion Matrix", key="conf_matrix"):
        st.altair_chart(
            binary_cm(st.session_state["y_pred_binary"], st.session_state["y_test"])
        )

    st.header("Metric table")
    if st.toggle(label="Create Metric Table", key="bi_metric_tab"):
        st.write(
            binary_metric_table(
                st.session_state["y_pred_binary"], st.session_state["y_test"]
            )
        )

    st.header("Feature importance")
    if st.toggle(label="Create Feature Importance Chart", key="bi_imp"):
        st.altair_chart(
            feature_importance(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["y_test"],
            )
        )

# Multiclass
elif st.session_state["problem_type"] == "Multiclass classification":
    st.header("Feature importance")
    if st.toggle(label="Create Feature Importance Chart", key="m_imp"):
        st.altair_chart(
            feature_importance(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["y_test"],
            )
        )

    st.header("Roc curves")
    if st.toggle(label="Create ROC Curves", key="m_roc"):
        st.altair_chart(
            multiclass_roc(
                st.session_state["y_pred"],
                st.session_state["y_test"],
                st.session_state["le"],
            )
        )

    st.header("Multiclass confusion matrices")
    if st.toggle(label="Create Multiclass Confusion Matrices", key="m_conf_matrix"):
        st.altair_chart(
            multiclass_cm(
                st.session_state["y_pred_binary"],
                st.session_state["y_test"],
                st.session_state["le"],
            )
        )
    st.header("Multiclass metrics")
    if st.toggle(label="Create Multiclass Metric Table"):
        st.write(
            create_multiclass_metric_table(
                st.session_state["y_test"],
                st.session_state["y_pred_binary"],
                st.session_state["y_train"],
            )
        )

# Regression
else:
    st.header("Residuals")
    if st.toggle(label="Create Residual Plot", key="reg_resids"):
        st.altair_chart(
            reg_residuals(st.session_state["y_pred"], st.session_state["y_test"])
        )

    st.header("Metric table")
    if st.toggle(label="Create Metric Table", key="reg_met_tab"):
        st.write(reg_table(st.session_state["y_pred"], st.session_state["y_test"]))

    st.header("Feature importance")
    if st.toggle(label="Create Feature Importance Chart", key="reg_imp"):
        st.altair_chart(
            feature_importance(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["y_test"],
            )
        )
