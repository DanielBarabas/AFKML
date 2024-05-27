import streamlit as st
import pandas as pd
import numpy as np
from modules_for_pages.data_wrangling import create_model_df
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
import altair as alt


df = pd.read_csv(
    "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
)
res_df = pd.DataFrame(
    {"Variable": ["sex", "DRK_YN"], "Encoding": ["One-Hot", "One-Hot"]}
)
X = create_model_df(res_df, df, "weight", "Categorical")

df["weight"] = df["weight"].astype(str)
le = LabelEncoder()
y = le.fit_transform(df["weight"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(max_depth=5,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
y_pred_binary = model.predict(X_test)


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

# Confusion matrix
cm = confusion_matrix(y_test,y_pred_binary)
cm_df = pd.DataFrame(cm, index=[f'True {i}' for i in le.inverse_transform([i for i in range(cm.shape[0])])], 
                     columns=[f'Pred {i}' for i in le.inverse_transform([i for i in range(cm.shape[1])])])
cm_long_df = cm_df.reset_index().melt(id_vars='index')
cm_long_df.columns = ['True Label', 'Predicted Label', 'Count']
heatmap = alt.Chart(cm_long_df).mark_rect().encode(
    x='Predicted Label:O',
    y='True Label:O',
    color='Count:Q',
    tooltip=['True Label', 'Predicted Label', 'Count']
).properties(
    title='Confusion Matrix Heatmap',
    width = 500,
    height = 500
)
st.altair_chart(heatmap)

#ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
roc_data = pd.DataFrame(columns=['False Positive Rate', 'True Positive Rate', 'Class', 'AUC'])

for i in range(len(le.classes_)):
    temp_df = pd.DataFrame({
        'False Positive Rate': fpr[i],
        'True Positive Rate': tpr[i],
        'Class': f'Class {le.inverse_transform([i])[0]}',
        'AUC': roc_auc[i]
    })
    roc_data = pd.concat([roc_data, temp_df])

roc_data.reset_index(drop=True, inplace=True)

roc_plot = alt.Chart(roc_data).mark_line().encode(
    x='False Positive Rate:Q',
    y='True Positive Rate:Q',
    color='Class:N',
    tooltip=['Class', 'AUC']
).properties(
    title='One-vs-All ROC Curves' ,
    width = 500,
    height = 500
)

diagonal = pd.DataFrame({
    'False Positive Rate': [0, 1],
    'True Positive Rate': [0, 1]
})
diagonal_line = alt.Chart(diagonal).mark_line(color='navy', strokeDash=[5,5]).encode(
    x='False Positive Rate',
    y='True Positive Rate'
)

roc_final = roc_plot + diagonal_line
st.altair_chart(roc_final)