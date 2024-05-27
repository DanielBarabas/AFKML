import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
import numpy as np
from modules_for_pages.data_wrangling import find_valid_cols, find_label_type, create_model_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve,  balanced_accuracy_score,  f1_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, TargetEncoder, LabelEncoder
import altair as alt

## Ezek a dolgok máshol történnek, de ide is kellenek a teszteléshez

df = pd.read_csv("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv")
res_df = pd.DataFrame({"Variable": ["sex"],
                       "Encoding":["One-Hot"]})
X = create_model_df(res_df,df,"DRK_YN","Categorical")
le = LabelEncoder()
y = le.fit_transform(df["DRK_YN"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
y_pred_binary = model.predict(X_test)

# ROC
y_pred1 = [y[1] for y in y_pred]
fpr, tpr, _ = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)

roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr
})

roc_curve_chart = alt.Chart(roc_data).mark_line(color='darkorange').encode(
    x=alt.X('False Positive Rate', scale=alt.Scale(domain=[0, 1])),
    y=alt.Y('True Positive Rate', scale=alt.Scale(domain=[0, 1])),
    tooltip=['False Positive Rate', 'True Positive Rate']
).properties(
    title=f'ROC Curve (AUC = {roc_auc:.2f})',
    height = 500,
    width = 500
)

diagonal = pd.DataFrame({
    'False Positive Rate': [0, 1],
    'True Positive Rate': [0, 1]
})
diagonal_line = alt.Chart(diagonal).mark_line(color='navy', strokeDash=[5,5]).encode(
    x='False Positive Rate',
    y='True Positive Rate'
)

final_chart = roc_curve_chart + diagonal_line
st.altair_chart(final_chart)

#Precision-Recall 
precision, recall, _ = precision_recall_curve(y_test, y_pred1)
print(len(precision),recall)
pr_data = pd.DataFrame({
    'Recall': recall,
    'Precision': precision
})
pr_curve_chart = alt.Chart(pr_data).mark_line(color='blue').encode(
    x=alt.X('Recall', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
    y=alt.Y('Precision', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
    tooltip=['Recall', 'Precision']
).properties(
    title=alt.TitleParams(text='Precision-Recall Curve', fontSize=20),
    width = 500,
    height = 500
)

st.altair_chart(pr_curve_chart)

#CM matrix
cm = confusion_matrix(y_test, y_pred_binary)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
cm_df_melt = cm_df.reset_index().melt(id_vars='index')
cm_df_melt.columns = ['Actual', 'Predicted', 'Count']

# Plot the confusion matrix using Altair
cm_chart = alt.Chart(cm_df_melt).mark_rect().encode(
    x=alt.X('Predicted:O', title='Predicted', axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
    y=alt.Y('Actual:O', title='Actual', axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
    color=alt.Color('Count:Q', legend=alt.Legend(labelFontSize=14, titleFontSize=16)),
    tooltip=['Actual', 'Predicted', 'Count']
).properties(
    title=alt.TitleParams(text='Confusion Matrix', fontSize=20),
    width = 500,
    height = 500
)
threshold = cm_df_melt['Count'].max() / 2.0
cm_text = cm_chart.mark_text(baseline='middle', fontSize=40).encode(
    text='Count:Q',
    color=alt.condition(
        alt.datum.Count > threshold,
        alt.value('white'),
        alt.value('black')
    )
)

cm_final_chart = cm_chart + cm_text
st.altair_chart(cm_final_chart)

# Table for numbers
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
specificity = tn / (tn+fp)
precision = tp / (fp + tp)
accuracy = (tn + tp) / (tn + fp + fn + tp)
balanced_accuracy = balanced_accuracy_score(y_test,y_pred_binary)
negative_predictive_value = tn / (tn + fn)
recall = tp / (tp + fn)
f1 = f1_score(y_test,y_pred_binary)
mse = mean_squared_error(y_test, y_pred_binary)

metric_df = pd.DataFrame({
    "Metric":["Accuracy","Balanced Accuracy","Sensitivity / Recall","Specificity","Precision","Negative Predictive Value",
              "F1 score","MSE"],
    "Value":[accuracy,balanced_accuracy,recall,specificity,precision,negative_predictive_value,f1,mse]
})

st.write(metric_df)
