import streamlit as st
from modules.eval import (
    binary_roc,
    binary_prec_recall,
    binary_cm,
    binary_metric_table,
    feature_importance,
    multiclass_cm,
    multiclass_roc,
    reg_table
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

######### Cached functions ###########
@st.cache_data
def predict1(_model, X_test, problem_type):
    if problem_type == "Regression":
        y_pred = _model.predict(X_test)
        y_pred_binary = "anyad"
    else:
        y_pred = _model.predict_proba(X_test)
        y_pred_binary = _model.predict(X_test)

    return y_pred, y_pred_binary
@st.cache_resource(experimental_allow_widgets=True)
def show_bin_roc_chart(y_pred,y_test):
    st.altair_chart(binary_roc(y_pred, y_test))

@st.cache_resource(experimental_allow_widgets=True)
def show_prec_recall_chart(y_pred,y_test):
    st.altair_chart(binary_prec_recall(y_pred, y_test))

@st.cache_resource(experimental_allow_widgets=True)
def show_bin_cm_chart(y_pred_binary,y_test):
    st.altair_chart(binary_cm(y_pred_binary, y_test))
@st.cache_resource(experimental_allow_widgets=True)
def show_bin_metric_table(y_pred_binary,y_test):
    st.write(binary_metric_table(y_pred_binary, y_test))

@st.cache_resource(experimental_allow_widgets=True)
def show_feature_importance(_model,X_test,y_test):
    st.altair_chart(
            feature_importance(
                _model,
                X_test,
                y_test,
            )
        )

@st.cache_resource(experimental_allow_widgets=True)
def show_multi_roc(y_pred,y_test,_le):
    st.altair_chart(
                multiclass_roc(y_pred, y_test, _le)
            )
    
@st.cache_resource(experimental_allow_widgets=True)
def show_multi_cm(y_pred_binary,y_test,_le):
    st.altair_chart(
                multiclass_cm(y_pred_binary, y_test, _le)
            )

@st.cache_resource(experimental_allow_widgets=True)
def show_reg_table(y_pred,y_test):       
    st.write(reg_table(y_pred,y_test))

y_pred, y_pred_binary = predict1(
    st.session_state["model"],
    st.session_state["X_test"],
    st.session_state["problem_type"],
)


if st.session_state["problem_type"] == "Binary classification":
    st.header("ROC curve")
    roc_switch = st.toggle("Create ROC Curve")
    if roc_switch:
        show_bin_roc_chart(y_pred,st.session_state["y_test"]) 

    st.header("Precision-Recall Chart")
    pr_switch = st.toggle(label="Create Precision-Recall Chart")
    if pr_switch:
        show_prec_recall_chart(y_pred,st.session_state["y_test"])

    st.header("Confusion matrix")
    cm_switch = st.toggle(label="Create Confusion Matrix")
    if cm_switch:
        show_bin_cm_chart(y_pred_binary,st.session_state["y_test"]) 

    st.header("Metric table")
    mt_switch = st.toggle(label="Create Metric Table")
    if mt_switch:
        show_bin_metric_table(y_pred_binary,st.session_state["y_test"])
        

    st.header("Feature importance")
    fi_switch = st.toggle(label="Create Feature Importance Chart")
    if fi_switch:
        show_feature_importance(st.session_state["model"],st.session_state["X_test"],st.session_state["y_test"])
        

elif st.session_state["problem_type"] == "Multiclass classification":
    st.header("Feature importance")
    fi_switch = st.toggle(label="Create Feature Importance Chart")
    if fi_switch:
        show_feature_importance(st.session_state["model"],st.session_state["X_test"],st.session_state["y_test"])

    st.header("Roc curves")
    roc_switch = st.toggle(label="Create ROC Curves")
    if roc_switch:
        show_multi_roc(y_pred,st.session_state["y_test"],st.session_state["le"])
        

    st.header("Multiclass confusion matrices")
    cm_switch = st.toggle(label="Create Multiclass Confusion Matrices")
    if cm_switch:
        show_multi_cm(y_pred_binary,st.session_state["y_test"],st.session_state["le"])

else:
    st.header("Metric table")
    mt_switch = st.toggle(label="Create Metric Table")
    if mt_switch:
        show_reg_table(y_pred,st.session_state["y_test"])

    st.header("Feature importance")
    fi_switch = st.toggle(label="Create Feature Importance Chart")
    if fi_switch:
        show_feature_importance(st.session_state["model"],st.session_state["X_test"],st.session_state["y_test"])
