import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid
import numpy as np
from modules.data_wrangling import  find_valid_cols, find_label_type, create_model_df




target_var = st.multiselect(
    "Choose the target variable:", st.session_state["df"].columns.to_list(), max_selections=1
)
if len(target_var) > 0:
    target_var = target_var[0]
else:
    target_var = st.session_state["df"].columns.to_list()[0]


encodings = ("One-Hot", "Target", "Ordinal")
valid_cols = find_valid_cols(st.session_state["df"],target_var)
y_type = find_label_type(st.session_state["df"],target_var)

st.write(target_var,y_type)

original_encodings = pd.DataFrame({"Variable": valid_cols, "Encoding": "One-Hot"})

gb = GridOptionsBuilder.from_dataframe(original_encodings)
gb.configure_column(
    "Encoding",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": encodings},
)

vgo = gb.build()
response = AgGrid(
    original_encodings,
    gridOptions=vgo,
)

model_df = create_model_df(response.data, st.session_state["df"], target_var,y_type)
st.write(model_df)
