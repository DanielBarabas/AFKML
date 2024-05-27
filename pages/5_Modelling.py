import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import modelling as m

# TODO töröl modelling a pages mappából
# TODO rewrite, so not everything is run automatically

st.set_page_config(page_title="Modelling", layout="wide")
st.title("Modelling")


### replace with session state df
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df[["sex", "DRK_YN"]] = df[["sex", "DRK_YN"]].astype("category")
    return df


df = load_data(
    "C:/Users/aronn/OneDrive/Asztali gép/Rajk/Prog 2/prog_machine_project/data/smoking_drinking.csv"
)

oh_enc = OneHotEncoder(drop="first", sparse_output=False, dtype=np.int8)

df_ = pd.DataFrame(oh_enc.fit_transform(df[["sex"]].values.reshape(-1, 1)))
df_colnames_oh = [str(f) for f in oh_enc.categories_[0][1:]]
df_.columns = df_colnames_oh
df = df_.join(df)

df_ = pd.DataFrame(oh_enc.fit_transform(df[["DRK_YN"]].values.reshape(-1, 1)))
df_colnames_oh = [str(f) for f in oh_enc.categories_[0][1:]]
df_.columns = df_colnames_oh
df = df_.join(df)

df = df.rename(columns={"Y": "is_drinking"})
df = df.drop(columns=["sex", "DRK_YN"])

st.write(df.head(10))
#### replace up to that point

with st.expander(label="Preparation", expanded=True):
    problem_type = st.selectbox(
        "Choose the type of your problem",
        options=["Regression", "Multi-classification", "Binary-classification"],
    )

    model_type = st.selectbox("Select model", options=["Random forest", "XGBoost"])

    test_ratio = st.slider(
        "Select what ratio you want the test set to be",
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        help="Normally test set ratio is between 0.1 and 0.3",
    )

    X_train, X_test, y_train, y_test = m.train_test_X_y_split(
        df=df, y_colname="triglyceride", test_ratio=test_ratio
    )

    st.write(len(X_train), len(X_test), len(y_train), len(y_test))

    use_all = st.checkbox("Use all features?", value=True)
    # TODO here the encoded cat variables might cause an issue
    if use_all:
        feat_used = X_train.columns
    else:
        feat_used = st.multiselect(
            "Choose the features to be included in modeling", options=X_train.columns
        )

    n_features = X_train.shape[1]


with st.expander(label="Hyper-parameters"):
    if model_type == "Random forest":
        (
            n_estimators,
            max_depth,
            max_features,
            min_samples_split,
            min_samples_leaf,
            bootstrap,
            n_jobs,
        ) = m.rf_param_input(n_features)

        if problem_type == "Regression":
            if st.button("Run model", key="rfr"):
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                ).fit(X_train.loc[:, feat_used], y_train)

                st.write(type(rf))
        else:
            if st.button("Run model", key="rfc"):
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                ).fit(X_train.loc[:, feat_used], y_train)

                st.write(type(rf))

    elif model_type == "XGBoost":
        (
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
        ) = m.xbg_param_input()

        if problem_type == "Regression":
            if st.button("Run model", key="xgbr"):
                xgb = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    min_child_weight=min_child_weight,
                    scale_pos_weight=scale_pos_weight,
                ).fit(X_train.loc[:, feat_used], y_train)

                st.write(type(xgb))
                st.write(xgb.evals_result)
                st.write(xgb.score)
        else:
            if st.button("Run model", key="xgbc"):
                xgb = XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    min_child_weight=min_child_weight,
                    scale_pos_weight=scale_pos_weight,
                ).fit(X_train.loc[:, feat_used], y_train)

                st.write(type(xgb))
                st.write(xgb.evals_result)
                st.write(xgb.score)

    # TODO write out: your model is in training + time passed
