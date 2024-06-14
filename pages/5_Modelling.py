import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import modules.modelling as m

####### Definitions for hyperparameters optimiztation #######

trials = Trials()
strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

xgb_space = {
    "n_estimators": hp.choice("n_estimators", np.arange(10, 1000, 10, dtype="int")),
    "max_depth": hp.choice("max_depth", np.arange(1, 15, 1, dtype=int)),
    "learning_rate": hp.uniform("learning_rate", 0.01, 1),
    "gamma": hp.uniform("gamma", 0, 10e1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "reg_lambda": hp.uniform("reg_lambda", 0, 1),
    "reg_alpha": hp.uniform("reg_alpha", 10e-7, 10),
    "min_child_weight": hp.choice("min_child_weight", np.arange(1, 10, 1, dtype="int")),
}

rf_space = {
    "n_estimators": hp.choice("n_estimators", np.arange(10, 500, 10, dtype="int")),
    "max_depth": hp.choice("max_depth", np.arange(1, 10, 1, dtype=int)),
    "max_features": hp.choice("max_features", [None, "sqrt", "log2"]),
    "min_samples_split": hp.choice(
        "min_samples_split", np.arange(2, 20, 1, dtype="int")
    ),
    "min_samples_leaf": hp.choice("min_samples_leaf", np.arange(1, 20, 1, dtype="int")),
}


def rf_class_obj_func(params):
    rf = RandomForestClassifier(**params)
    score = cross_val_score(
        estimator=rf,
        X=st.session_state["X_train"],
        y=st.session_state["y_train"],
        cv=strat_kfold,
        scoring="accuracy",
        n_jobs=-1,
    ).mean()

    loss = -score

    return {"loss": loss, "params": params, "status": STATUS_OK}


def rf_reg_obj_func(params):
    rf = RandomForestRegressor(**params)
    score = cross_val_score(
        estimator=rf,
        X=st.session_state["X_train"],
        y=st.session_state["y_train"],
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).mean()

    loss = -score

    return {"loss": loss, "params": params, "status": STATUS_OK}


def xgb_class_obj_func(params):
    xgboost = XGBClassifier(**params)
    score = cross_val_score(
        estimator=xgboost,
        X=st.session_state["X_train"],
        y=st.session_state["y_train"],
        cv=strat_kfold,
        scoring="accuracy",
        n_jobs=-1,
    ).mean()

    loss = -score

    return {"loss": loss, "params": params, "status": STATUS_OK}


def xgb_reg_obj_func(params):
    xgboost = XGBRegressor(**params, objective="reg:squarederror")
    score = cross_val_score(
        estimator=xgboost,
        X=st.session_state["X_train"],
        y=st.session_state["y_train"],
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    ).mean()

    loss = -score

    return {"loss": loss, "params": params, "status": STATUS_OK}


####### Page content #######


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


# Preparation
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


# Hyperparameters
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

    # Rf regression
    if st.session_state["problem_type"] == "Regression":
        if st.button("Train model", key="rfr"):
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

        # Hyperparameter optimization
        if st.button("Run hyperparameter optimization", key="rfr_opt"):
            best = fmin(
                fn=rf_reg_obj_func,
                space=rf_space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
            )

            st.write("Best hyperparameters found:")
            for key, value in best.items():
                st.write(f"  - {key}: {value:.2f}")

            if st.button("Train model with best hyperparameters", key="rfr_best"):
                st.session_state["model"] = RandomForestRegressor(**best).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )
                st.session_state["y_pred"], st.session_state["y_pred_binary"] = (
                    m.predict1(
                        st.session_state["model"],
                        st.session_state["X_test"],
                        st.session_state["problem_type"],
                    )
                )

                st.write(
                    "Model training is complete go to evaluation page to see model diagnostics"
                )
                st.balloons()

    # Rf classification
    else:
        if st.button("Train model", key="rfc"):
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

        # Hyperparameter optimization
        if st.button("Run hyperparameter optimization", key="rfc_opt"):
            best = fmin(
                fn=rf_class_obj_func,
                space=rf_space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
            )

            st.write("Best hyperparameters found:")
            for key, value in best.items():
                st.write(f"  - {key}: {value:.2f}")

            # Train with best params
            if st.button("Train model with best hyperparameters", key="rfc_best"):
                st.session_state["model"] = RandomForestClassifier(**best).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )
                st.session_state["y_pred"], st.session_state["y_pred_binary"] = (
                    m.predict1(
                        st.session_state["model"],
                        st.session_state["X_test"],
                        st.session_state["problem_type"],
                    )
                )

                st.write(
                    "Model training is complete go to evaluation page to see model diagnostics"
                )
                st.balloons()

# Xgboost
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

    # Xgb regression
    if st.session_state["problem_type"] == "Regression":
        if st.button("Train model", key="xgbr"):
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

        # Hyperpameter optimization
        if st.button("Run hyperparameter optimization", key="xgbr_opt"):
            best = fmin(
                fn=xgb_reg_obj_func,
                space=xgb_space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
            )

            st.write("Best hyperparameters found:")
            for key, value in best.items():
                st.write(f"  - {key}: {value:.2f}")

            # Train with best params
            if st.button("Train model with best hyperparameters", key="xgbr_best"):
                st.session_state["model"] = XGBRegressor(**best).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )
                st.session_state["y_pred"], st.session_state["y_pred_binary"] = (
                    m.predict1(
                        st.session_state["model"],
                        st.session_state["X_test"],
                        st.session_state["problem_type"],
                    )
                )

                st.write(
                    "Model training is complete go to evaluation page to see model diagnostics"
                )
                st.balloons()

    # Xgb classification
    else:
        if st.button("Train model", key="xgbc"):
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
            st.balloons()

        # Hyperparameter optimization
        if st.button("Run hyperparameter optimization", key="xgbc_opt"):
            best = fmin(
                fn=xgb_class_obj_func,
                space=xgb_space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
            )

            st.write("Best hyperparameters found:")
            for key, value in best.items():
                st.write(f"  - {key}: {value:.2f}")

            # Train with best params
            if st.button("Train model with best hyperparameters", key="xgbc_best"):
                st.session_state["model"] = XGBClassifier(**best).fit(
                    st.session_state["X_train"].loc[:, feat_used],
                    st.session_state["y_train"],
                )
                st.session_state["y_pred"], st.session_state["y_pred_binary"] = (
                    m.predict1(
                        st.session_state["model"],
                        st.session_state["X_test"],
                        st.session_state["problem_type"],
                    )
                )

                st.write(
                    "Model training is complete go to evaluation page to see model diagnostics"
                )
                st.balloons()

# Linear regression
elif model_type == "Linear regression":
    fit_intercept = st.checkbox("Fit intercept", value=True)
    paralel = st.checkbox(
        "Do you want to use multiple CPUs for the calculations?", value=True
    )
    if paralel:
        n_jobs = -1
    else:
        n_jobs = None

    if st.button("Train model", key="lr"):
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

# Logistic regression
elif model_type == "Logistic regression":
    fit_intercept = st.checkbox("Fit intercept", value=True)
    paralel = st.checkbox(
        "Do you want to use multiple CPUs for the calculations?", value=True
    )
    if paralel:
        n_jobs = -1
    else:
        n_jobs = None

    if st.button("Train model", key="logr"):
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
