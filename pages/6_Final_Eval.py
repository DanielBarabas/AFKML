import streamlit as st
from modules.eval import preproc, binary_roc, binary_prec_recall, binary_cm, binary_metric_table, feature_importance, multiclass_cm, multiclass_roc,reg_table, predict1

import altair as alt

#This will be replaced with session state variables
#problem_type="Multiclass classification"
@st.cache_data
def do_shit(problem_type):
    return preproc(problem_type=problem_type)

#model,y_pred,y_pred_binary,X_test,y_test,le = do_shit(problem_type)

# Load session state variables
problem_type = st.session_state["problem_type"]
model = st.session_state["model"]
le = st.session_state["le"]
(X_train, X_test, y_train, y_test) = (
        st.session_state["X_train"],
        st.session_state["X_test"],
        st.session_state["y_train"],
        st.session_state["y_test"],
    )

y_pred, y_pred_binary = predict1(model,X_test,problem_type)




if problem_type == "Binary classification":
    with st.expander(label="ROC Curve"):
        if st.button("Create ROC Curve"):
            st.altair_chart(binary_roc(y_pred,y_test))

    with st.expander(label="Precision-Recall Chart"):
        if st.button("Create Precision-Recall Chart"):
            st.altair_chart(binary_prec_recall(y_pred,y_test))

    with st.expander(label="Confusion Matrix"):
        if st.button("Create Confusion Matrix"):
            st.altair_chart(binary_cm(y_pred_binary,y_test))

    with st.expander(label="Metric Table"):
        if st.button("Create Metric Table"):
            st.write(binary_metric_table(y_pred_binary,y_test))

    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(feature_importance(model,X_test,y_test))

elif problem_type == "Multiclass classification":
    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(feature_importance(model,X_test,y_test))
    with st.expander(label="ROC Curves"):
        if st.button("Create ROC Curves"):
            st.altair_chart(multiclass_roc(y_pred,y_test,le))
    with st.expander(label="Multiclass Confusion Matrices"):
        if st.button("Create Multiclass Confusion Matrices"):
            st.altair_chart(multiclass_cm(y_pred_binary,y_test,le))

else:
    with st.expander(label="Feature Importance"):
        if st.button("Create Feature Importance Chart"):
            st.altair_chart(feature_importance(model,X_test,y_test))
    with st.expander(label="Metric Table"):
        if st.button("Create Metric Table"):
            st.write(reg_table(y_pred,y_test))

