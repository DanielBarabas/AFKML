import streamlit as st
from streamlit_extras.stateful_button import button
from modules.eval import (
    binary_roc,
    binary_prec_recall,
    binary_cm,
    binary_metric_table,
    feature_importance,
    multiclass_cm,
    multiclass_roc,
    reg_table,
    predict1,
)


y_pred, y_pred_binary = predict1(
    st.session_state["model"],
    st.session_state["X_test"],
    st.session_state["problem_type"],
)


if st.session_state["problem_type"] == "Binary classification":
    with st.expander(label="ROC Curve"):
        if button(label="Create ROC Curve", key="bi_roc"):
            st.altair_chart(binary_roc(y_pred, st.session_state["y_test"]))

    with st.expander(label="Precision-Recall Chart"):
        if button(label="Create Precision-Recall Chart", key="bi_prec_rec"):
            st.altair_chart(binary_prec_recall(y_pred, st.session_state["y_test"]))

    with st.expander(label="Confusion Matrix"):
        if button(label="Create Confusion Matrix", key="conf_matrix"):
            st.altair_chart(binary_cm(y_pred_binary, st.session_state["y_test"]))

    with st.expander(label="Metric Table"):
        if button(label="Create Metric Table", key="bi_metric_tab"):
            st.write(binary_metric_table(y_pred_binary, st.session_state["y_test"]))

    with st.expander(label="Feature Importance"):
        if button(label="Create Feature Importance Chart", key="bi_imp"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )

elif st.session_state["problem_type"] == "Multiclass classification":
    with st.expander(label="Feature Importance"):
        if button(label="Create Feature Importance Chart", key="m_imp"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )
    with st.expander(label="ROC Curves"):
        if button(label="Create ROC Curves", key="m_roc"):
            st.altair_chart(
                multiclass_roc(
                    y_pred, st.session_state["y_test"], st.session_state["le"]
                )
            )
    with st.expander(label="Multiclass Confusion Matrices"):
        if button(label="Create Multiclass Confusion Matrices", key="m_conf_matrix"):
            st.altair_chart(
                multiclass_cm(
                    y_pred_binary, st.session_state["y_test"], st.session_state["le"]
                )
            )

else:
    with st.expander(label="Metric Table"):
        if button(label="Create Metric Table", key="reg_met_tab"):
            st.write(reg_table(y_pred, st.session_state["y_test"]))
    with st.expander(label="Feature Importance"):
        if button(label="Create Feature Importance Chart", key="reg_imp"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )
