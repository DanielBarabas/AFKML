import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, AgGridReturn

df = pd.read_csv(
    "C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv"
)

typelist = ("Numeric", "Categorical")

dtype_map = {"Numeric": float, "Categorical": "category"}
dtype_map_inverse = {'float64': "Numeric", "category": "Categorical","object":"Categorical",'int64': "Numeric"}

df1 = pd.DataFrame(
    {
    "Variable": df.columns.to_list(), 
    "Type": [dtype_map_inverse[a] for a in [str(typ) for typ in df.dtypes]]
    }
)

gb = GridOptionsBuilder.from_dataframe(df1)
gb.configure_column(
    "Type",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": typelist},
)

vgo = gb.build()
response = AgGrid(
    df1,
    gridOptions=vgo,
    # theme='blue'
)


st.write(response.data)
