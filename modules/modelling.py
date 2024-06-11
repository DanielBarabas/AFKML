import streamlit as st


def rf_param_input(n_features) -> tuple:
    """Gives ui to specify the parameters of random forest

    Args:
        n_features (int): number of features

    Returns:
        tuple: specified hyperparameter values
    """
    st.write(
        "Read the scikit-learn documentation for more information on the parameters: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
    )
    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100)
    max_depth = st.radio(
        "max_depth", options=["Up to min_samples_split", "Input specific number"]
    )
    if max_depth == "Input specific number":
        max_depth = st.slider("max_depth", min_value=0, max_value=15, value=4)
    else:
        max_depth = None

    max_features = st.radio(
        "max_features", options=["All", "sqrt", "log2", "Input specific number"]
    )
    if max_features == "Input specific number":
        st.slider("max_features", min_value=1, max_value=n_features, value=1)
    elif max_features == "All":
        max_features = None

    min_samples_split = st.number_input(
        "min_samples_split", min_value=1, value=2, step=1
    )
    min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, value=1, step=1)
    bootstrap = st.checkbox("Boostrap", value=True)
    paralel = st.checkbox(
        "Do you want to build trees in paralel with multiple CPUs", value=True
    )
    if paralel:
        n_jobs = -1
    else:
        n_jobs = None

    return (
        n_estimators,
        max_depth,
        max_features,
        min_samples_split,
        min_samples_leaf,
        bootstrap,
        n_jobs,
    )


def xbg_param_input() -> tuple:
    """Gives ui to specify the parameters of xgboost

    Returns:
        tuple: specified parameter values
    """
    st.write(
        "Read the XGBoost documentation for more information on the parameters: https://xgboost.readthedocs.io/en/stable/parameter.html"
    )
    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100)
    max_depth = st.slider("max_depth", min_value=0, max_value=15, value=6)
    learning_rate = st.slider(
        "learning_rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01
    )
    gamma = st.number_input("gamma", min_value=0.0, value="min", step=0.05)
    subsample = st.slider(
        "subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.01
    )
    colsample_bytree = st.slider(
        "colsample_bytree", min_value=0.1, max_value=1.0, value=1.0, step=0.01
    )
    reg_lambda = st.number_input("lamdba", min_value=0.0, value=1.0, step=0.1)
    reg_alpha = st.number_input("alpha", min_value=0.0, value=1.0, step=0.1)
    min_child_weight = st.number_input(
        "min_child_weight", min_value=0.0, value=1.0, step=0.1
    )
    scale_pos_weight = st.number_input(
        "scale_pos_weight", min_value=0.0, value=1.0, step=0.1
    )

    return (
        n_estimators,
        max_depth,
        learning_rate,
        gamma,
        subsample,
        colsample_bytree,
        reg_lambda,
        reg_alpha,
        min_child_weight,
        scale_pos_weight,
    )


def log_reg_param_input() -> tuple:
    C = st.slider("C", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    penalty = st.radio("penalty", options=["l1", "l2"])
    solver = st.radio("solver", options=["liblinear", "saga"])

    return C, penalty, solver
