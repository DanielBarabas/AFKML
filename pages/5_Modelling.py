import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import modelling as m

# TODO rewrite, so not everything is run automatically

n_features = st.session_state["X"].shape[1]


st.set_page_config(page_title="Modelling", layout="wide")
st.title("Modelling")


# Define own function only for caching
@st.cache_data
def my_train_test_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=72)


with st.expander(label="Preparation", expanded=True):
    st.session_state["problem_type"] = st.selectbox(
        "Choose the type of your problem",
        options=["Regression", "Multiclass classification", "Binary classification"],
    )

    model_type = st.selectbox("Select model", options=["Random forest", "XGBoost"])

    test_size = st.slider(
        "Select what ratio you want the test set to be",
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        help="Normally test set ratio is between 0.1 and 0.3",
    )

    (
        st.session_state["X_train"],
        st.session_state["X_test"],
        st.session_state["y_train"],
        st.session_state["y_test"],
    ) = my_train_test_split(
        st.session_state["X"], st.session_state["y"], test_size=test_size
    )

    st.write("Number of rows in training set", len(st.session_state["X_train"]))
    st.write("Number of rows in test set", len(st.session_state["X_test"]))

    use_all = st.checkbox("Use all features?", value=True)
    # TODO here the encoded cat variables might cause an issue
    if use_all:
        feat_used = st.session_state["X_train"].columns
    else:
        feat_used = st.multiselect(
            "Choose the features to be included in modeling",
            options=st.session_state["X_train"].columns,
        )


with st.expander(label="Hyperparameters"):
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

        if st.session_state["problem_type"] == "Regression":
            if st.button("Run model", key="rfr"):
                st.session_state["model"] = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                ).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )

                st.write(type(st.session_state["model"]))
        else:
            if st.button("Run model", key="rfc"):
                st.session_state["model"] = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap,
                    n_jobs=n_jobs,
                ).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )

                st.write(type(st.session_state["model"]))

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

        if st.session_state["problem_type"] == "Regression":
            if st.button("Run model", key="xgbr"):
                st.session_state["model"] = XGBRegressor(
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
                ).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )

                st.write(type(st.session_state["model"]))
                st.write(st.session_state["model"].evals_result)
                st.write(st.session_state["model"].score)
        else:
            if st.button("Run model", key="xgbc"):
                st.session_state["model"] = XGBClassifier(
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
                ).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )

                st.write(type(st.session_state["model"]))
                st.write(st.session_state["model"].evals_result)
                st.write(st.session_state["model"].score)

    # TODO write out: your model is in training + time passed
    # TODO cache the models?
