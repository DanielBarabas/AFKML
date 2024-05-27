import streamlit as st
import pandas as pd
import numpy as np
from modules_for_pages.data_wrangling import (
    find_valid_cols,
    find_label_type,
    create_model_df,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
    LabelEncoder,
)
import altair as alt


df = pd.read_csv(
    "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
)
res_df = pd.DataFrame(
    {"Variable": ["sex", "DRK_YN"], "Encoding": ["One-Hot", "One-Hot"]}
)
X = create_model_df(res_df, df, "hemoglobin", "Numeric")

y = df["hemoglobin"]
st.write(y.dtype)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Table for számok
mse = mean_squared_error(y_test, y_pred)
metric_df = pd.DataFrame({"Metric": ["Mean Squared Error"], "Value": [mse]})
st.write(metric_df)
