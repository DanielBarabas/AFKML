import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st


# TODO kill
# TODO return np arrays instead -> faster? (csak ebben az esetben van értelme ennek az fv-nek)
def train_test_X_y_split(df: pd.DataFrame, y_colname: str, test_ratio: float = 0.2):
    """Creates test/train and X/y split of a df

    Args:
        df (pd.DataFrame): your data frame
        y_colname (str): name of target variable
        test_ratio (float, optional): ratio of test set. Defaults to 0.2.

    Returns:
        X_train, X_test, y_train, y_test (pd.DataFrame): test/train split and X/y split dfs
    """
    retlist = train_test_split(
        df.drop(y_colname, axis=1),
        df[y_colname],
        test_size=test_ratio,
        random_state=72,
    )

    return [
        (
            pd.DataFrame(f, columns=[f for f in df.columns if not f == y_colname])
            if i < 2
            else pd.DataFrame(f, columns=[y_colname])
        )
        for i, f in enumerate(retlist)
    ]


# TODO mások a paraméterek regressziónál
def rf_param_input(n_features):
    """Gives ui to specify the parameters of random forest

    Args:
        n_features (int): number of features

    Returns:
        tuple: specified parameter values
    """
    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100)
    max_depth = st.radio(
        "max_depth", options=["Up to min_samples_split", "Input specific number"]
    )
    if max_depth == "Input specific number":
        st.slider("max_depth", min_value=0, max_value=15, value=4)
    else:
        max_depth = None

    max_features = st.radio(
        "max_features", options=["All", "Square root", "Log", "Input specific number"]
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

    return (
        n_estimators,
        max_depth,
        max_features,
        min_samples_split,
        min_samples_leaf,
        bootstrap,
        n_jobs,
    )


def xbg_param_input():
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
    reg_alpha = st.number_input("aplha", min_value=0.0, value=1.0, step=0.1)
    # TODO add step?
    min_child_weight = st.number_input("min_child_weight", min_value=0, value=1)
    # TODO add step?
    scale_pos_weight = st.number_input("scale_pos_weight", min_value=0, value=1)

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
