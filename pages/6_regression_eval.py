import streamlit as st
import pandas as pd
import numpy as np
from modules_for_pages.data_wrangling import create_model_df

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import altair as alt


df = pd.read_csv(
    "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
)
res_df = pd.DataFrame(
    {"Variable": ["sex", "DRK_YN"], "Encoding": ["One-Hot", "One-Hot"]}
)
X = create_model_df(res_df, df, "hemoglobin", "Numeric")

y = df["hemoglobin"]
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


# Feature importance

result = permutation_importance(
    model, X_test, y_test, n_repeats=2, random_state=42, 
    #n_jobs=4 This does not work for the regression case for some reason :D
)
importances = pd.Series(result.importances_mean)
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)
importance_chart = (
    alt.Chart(feature_importance_df)
    .mark_bar()
    .encode(
        x=alt.X("Importance", title="Feature Importance"),
        y=alt.Y("Feature", sort="-x", title="Feature"),
        tooltip=["Feature", "Importance"],
    )
    .properties(title="Feature Importance", height=500, width=500)
)
st.altair_chart(importance_chart)
