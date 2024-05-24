import streamlit as st
import pandas as pd
from modelling import train_test_X_y_split

# TODO töröl modelling a pages mappából

st.set_page_config(page_title="Modelling", layout="wide")
st.title("Modelling")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df[["sex", "DRK_YN"]] = df[["sex", "DRK_YN"]].astype("category")
    return df


df = load_data(
    "C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/smoking_drinking.csv"
)

st.write(df.head(10))

with st.expander(label="Preparation", expanded=True):
    test_ratio = st.slider(
        "Select what ratio you want the test set to be",
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        help="Normally test set ratio is between 0.1 and 0.3",
    )

    X_train, X_test, y_train, y_test = train_test_X_y_split(
        df=df, y_colname="DRK_YN", test_ratio=test_ratio
    )

    st.write(len(X_train), len(X_test), len(y_train), len(y_test))

    model_type = st.selectbox("Select model", options=["Random forest"])

    # not live ATM
    # TODO dynmically update loss func options based on model
    model_lossf_map = {}
    loss_func = st.selectbox("Select loss function", options=["RMSE", "MSE", "MAE"])

    use_all = st.checkbox("Use all features?", value=True)
    # TODO here the encoded cat variables might cause an issue
    if not use_all:
        feat_used = st.multiselect(
            "Choose the features to be included in modeling", options=X_train.columns
        )

model_params_map = {
    "random_forest": [
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_split",
        "min_samples_leaf",
        "bootstrap",
        "n_jobs",
    ],
    "xgboost": [],
    "decision_tree": [],
}

with st.expander(label="Parameters"):
    n_estimators = st.slider("n_estimators", min_value=10, max_value=1000, value=100)
    # max_depth = -> problem: defaults to None
    max_features = st.radio(
        "max_features", options=["All", "Square root", "Log", "Specify in integers"]
    )

    if max_features == "Specify in integers":
        st.slider("max_features", min_value=1, max_value=X_train.shape[1], value=1)

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
    # TODO write out: your model is in training + time passed
