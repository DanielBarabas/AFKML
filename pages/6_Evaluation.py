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
    predict1,
)


y_pred, y_pred_binary = predict1(
    st.session_state["model"],
    st.session_state["X_test"],
    st.session_state["problem_type"],
)


if st.session_state["problem_type"] == "Binary classification":
    with st.expander(label="ROC Curve"):
        if st.button("Create ROC Curve"):
            st.altair_chart(binary_roc(y_pred, st.session_state["y_test"]))

    with st.expander(label="Precision-Recall Chart"):
        if st.button("Create Precision-Recall Chart"):
            st.altair_chart(binary_prec_recall(y_pred, st.session_state["y_test"]))

    with st.expander(label="Confusion Matrix"):
        if st.button("Create Confusion Matrix"):
            st.altair_chart(binary_cm(y_pred_binary, st.session_state["y_test"]))

    with st.expander(label="Metric Table"):
        if st.button("Create Metric Table"):
            st.write(binary_metric_table(y_pred_binary, st.session_state["y_test"]))

    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )

elif st.session_state["problem_type"] == "Multiclass classification":
    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )
    with st.expander(label="ROC Curves"):
        if st.button("Create ROC Curves"):
            st.altair_chart(
                multiclass_roc(
                    y_pred, st.session_state["y_test"], st.session_state["le"]
                )
            )
    with st.expander(label="Multiclass Confusion Matrices"):
        if st.button("Create Multiclass Confusion Matrices"):
            st.altair_chart(
                multiclass_cm(
                    y_pred_binary, st.session_state["y_test"], st.session_state["le"]
                )
            )

else:
    with st.expander(label="Metric Table"):
        if st.button("Create Metric Table"):
            st.write(reg_table(y_pred, st.session_state["y_test"]))
    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(
                feature_importance(
                    st.session_state["model"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            )
