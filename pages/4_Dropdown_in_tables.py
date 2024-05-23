import streamlit as st
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data("C:\Projects\Rajk\prog_2\project\prog_machine_project\data\drinking.csv")

typelist = ("Numeric", "Categorical")

dtype_map = {"Numeric": float, "Categorical": "category"}
dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}

# This can be cached:
original_types = pd.DataFrame(
    {
        "Variable": df.columns.to_list(),
        "Type": [dtype_map_inverse[a] for a in [str(typ) for typ in df.dtypes]],
    }
)
orig_dict = {}
for i in range(len(original_types)):
    orig_dict[original_types.iloc[i]["Variable"]] = original_types.iloc[i]["Type"]


gb = GridOptionsBuilder.from_dataframe(original_types)
gb.configure_column(
    "Type",
    editable=True,
    cellEditor="agSelectCellEditor",
    cellEditorParams={"values": typelist},
)

vgo = gb.build()
response = AgGrid(
    original_types,
    gridOptions=vgo,
)

res_dict = {}
for i in range(len(response.data)):
    res_dict[response.data.iloc[i]["Variable"]] = response.data.iloc[i]["Type"]
st.write(res_dict == orig_dict)
for key in res_dict.keys():
    if res_dict[key] != orig_dict[key]:
        # Cannot handle exceptions yet
        df[key] = df[key].astype(dtype_map[res_dict[key]])
orig_dict = res_dict
st.write(df.dtypes)

# Encoding part
