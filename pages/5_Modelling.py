import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
import modules.modelling as m


st.set_page_config(page_title="Modelling", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()
elif "X" not in st.session_state:
    st.write(
        "First choose the target variable and approriate encoding on the encoding page"
    )
    st.stop()

n_features = st.session_state["X"].shape[1]


st.title("Modelling")


# Define own function only for caching
@st.cache_data
def my_train_test_split(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=72)


st.header("Preparation")
st.write(
    f'You are going to do {st.session_state["problem_type"]} since the target variable, {st.session_state["y_colname"]} is {st.session_state["y_type"]}'
)


if st.session_state["problem_type"] == "Regression":
    model_options = ["Random forest", "XGBoost", "Linear regression"]
else:
    model_options = ["Random forest", "XGBoost", "Logistic regression"]

model_type = st.selectbox(
    "Select model",
    options=model_options,
)

test_size = st.slider(
    "Select what ratio you want the test set to be",
    min_value=0.1,
    max_value=0.5,
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
if use_all:
    feat_used = st.session_state["X_train"].columns
else:
    feat_used = st.multiselect(
        "Choose the features to be included in modeling",
        options=st.session_state["X_train"].columns,
    )


st.header("Hyperparameters")
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
            st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["problem_type"],
            )

            st.write(
                "Model training is complete go to evaluation page to see model diagnostics"
            )
            st.balloons()
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
            st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["problem_type"],
            )

            st.write(
                "Model training is complete go to evaluation page to see model diagnostics"
            )
            st.balloons()
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
            ).fit(
                st.session_state["X_train"].loc[:, feat_used],
                st.session_state["y_train"],
            )
            st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["problem_type"],
            )

            st.write(
                "Model training is complete go to evaluation page to see model diagnostics"
            )
            st.balloons()
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
            ).fit(
                st.session_state["X_train"].loc[:, feat_used],
                st.session_state["y_train"],
            )
            st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
                st.session_state["model"],
                st.session_state["X_test"],
                st.session_state["problem_type"],
            )

            st.write(
                "Model training is complete go to evaluation page to see model diagnostics"
            )

            st.text(f"Value of n_estimators: {n_estimators}")
            st.text(f"Value of max_depth: {max_depth}")
            st.text(f"Value of learning_rate: {learning_rate}")
            st.text(f"Value of subs: {subsample}")
            st.text(f"Value of cols: {colsample_bytree}")
            st.balloons()
elif model_type == "Linear regression":
    fit_intercept = st.checkbox("Fit intercept", value=True)
    paralel = st.checkbox(
        "Do you want to use multiple CPUs for the calculations?", value=True
    )
    if paralel:
        n_jobs = -1
    else:
        n_jobs = None

    if st.button("Run model", key="lr"):
        st.session_state["model"] = LinearRegression(
            fit_intercept=fit_intercept, n_jobs=n_jobs
        ).fit(
            st.session_state["X_train"].loc[:, feat_used],
            st.session_state["y_train"],
        )
        st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
            st.session_state["model"],
            st.session_state["X_test"],
            st.session_state["problem_type"],
        )

        st.write(
            "Model training is complete go to evaluation page to see model diagnostics"
        )
        st.balloons()
elif model_type == "Logistic regression":
    fit_intercept = st.checkbox("Fit intercept", value=True)
    paralel = st.checkbox(
        "Do you want to use multiple CPUs for the calculations?", value=True
    )
    if paralel:
        n_jobs = -1
    else:
        n_jobs = None

    if st.button("Run model", key="logr"):
        st.session_state["model"] = LogisticRegression(
            fit_intercept=fit_intercept, n_jobs=n_jobs
        ).fit(
            st.session_state["X_train"].loc[:, feat_used],
            st.session_state["y_train"],
        )
        st.session_state["y_pred"], st.session_state["y_pred_binary"] = m.predict1(
            st.session_state["model"],
            st.session_state["X_test"],
            st.session_state["problem_type"],
        )

        st.write(
            "Model training is complete go to evaluation page to see model diagnostics"
        )
        st.balloons()


# Drop unused features from session_state X_test
# Otherwise eval page will throw an error
st.session_state["X_test"] = st.session_state["X_test"].loc[:, feat_used]
