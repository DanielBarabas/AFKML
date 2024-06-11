import streamlit as st
import altair as alt
from modules.eda import (
    v_counts_bar_chart,
    histogram,
    boxplot,
    stacked_bar,
    scatter,
    cor_matrix,
    missing_value_plot,
)
from modules.data_wrangling import cast_date_to_timestamp


st.set_page_config(page_title="Exploratory data analysis", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()


st.title("Exploratory data analysis")
alt.data_transformers.disable_max_rows()
st.session_state["df"] = cast_date_to_timestamp(st.session_state["df"])

dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}


st.header("Descriptive Table")
percentiles = st.multiselect(
    label="Choose percentiles",
    options=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
    default=[0.25, 0.5, 0.75],
)
if st.toggle(label="Create descriptive statistics", key="desc_stat"):
    st.write(st.session_state["df"].describe(percentiles=percentiles))


st.header("Histogram")
num_options = st.session_state["df"].select_dtypes(include="number").columns.tolist()
selected_num = st.selectbox("Select numeric variable", num_options)
if st.toggle(label="Create histogram", key="hist"):
    fig = histogram(st.session_state["df"], selected_num)
    st.altair_chart(fig, use_container_width=True)

st.header("Value Counts Chart")
cat_options = st.session_state["df"].select_dtypes(include="object").columns.tolist()
selected_cat = st.selectbox("Select categorical variable", cat_options)

if st.toggle(label="Create value counts chart", key="v_counts"):
    fig = v_counts_bar_chart(st.session_state["df"], selected_cat)
    st.altair_chart(fig, use_container_width=True)


st.header("Association Figures")
options = st.multiselect(
    label="Choose the desired colums",
    options=st.session_state["df"].columns,
    default=st.session_state["df"].columns[[1, 2]].to_list(),
    max_selections=2,
)

if len(options) == 2 and st.toggle(label="Create association figure", key="ass"):
    # Both category:
    if (
        dtype_map_inverse[str(st.session_state["df"][options[0]].dtype)]
        == dtype_map_inverse[str(st.session_state["df"][options[1]].dtype)]
        == "Categorical"
    ):
        st.write("Both Cat")
        fig = stacked_bar(st.session_state["df"], options)
    # Both continous:
    elif (
        dtype_map_inverse[str(st.session_state["df"][options[0]].dtype)]
        == dtype_map_inverse[str(st.session_state["df"][options[1]].dtype)]
        == "Numeric"
    ):
        st.write("Both Cont")
        fig = scatter(st.session_state["df"], options)
    # One is continous, other is not:
    else:
        st.write("Mixed")
        fig = boxplot(st.session_state["df"], options)
    st.altair_chart(fig, use_container_width=True)


st.header("Correlation Matrix")
if st.toggle(label="Create correlation matrix", key="corr"):
    fig = cor_matrix(st.session_state["df"])
    st.altair_chart(fig, use_container_width=True)


st.header("Missing Values")
if st.toggle(label="Create missing values chart", key="missing"):
    fig = missing_value_plot(st.session_state["df"])
    st.pyplot(fig)
