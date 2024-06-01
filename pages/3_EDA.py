import streamlit as st
import altair as alt
import pandas as pd

from modules.eda import (
    v_counts_bar_chart,
    boxplot,
    stacked_bar,
    scatter,
    cor_matrix,
    missing_value_plot,
    desc_table,
)
from modules.data_wrangling import cast_date_to_timestamp


st.set_page_config(page_title="Exploratory data analysis", layout="wide")

# Don't run until no data is uploaded
if "df" not in st.session_state:
    st.write("First upload some data on the homepage")
    st.stop()


st.title("Exploratory data analysis")
alt.data_transformers.disable_max_rows()


### Cached functions


@st.cache_data
def get_cat_options(df):
    cat_options = df.select_dtypes(include="object").columns.tolist()
    return cat_options


@st.cache_data
def update_df(df):
    return cast_date_to_timestamp(df)


@st.cache_resource(experimental_allow_widgets=True)
def show_desc_table(df):
    st.write(desc_table(df))


@st.cache_resource(experimental_allow_widgets=True)
def show_vcount_bar_chart(df, selected_cat):
    st.altair_chart(
        v_counts_bar_chart(df, selected_cat, vc_switch), use_container_width=True
    )


@st.cache_resource(experimental_allow_widgets=True)
def show_ass_chart(df, options):
    if len(options) == 2:

        # Both category:
        if (
            dtype_map_inverse[str(df[options[0]].dtype)]
            == dtype_map_inverse[str(df[options[1]].dtype)]
            == "Categorical"
        ):
            st.write("Both Cat")
            fig = stacked_bar(df, options)
        # Both continous:
        elif (
            dtype_map_inverse[str(df[options[0]].dtype)]
            == dtype_map_inverse[str(df[options[1]].dtype)]
            == "Numeric"
        ):
            st.write("Both Cont")
            fig = scatter(df, options)
        # One is continous, other is not:
        else:
            st.write("Mixed")
            fig = boxplot(df, options)
        st.altair_chart(fig, use_container_width=True)


@st.cache_resource(experimental_allow_widgets=True)
def show_cor_matrix(df):
    st.altair_chart(cor_matrix(df), use_container_width=True)


@st.cache_resource(experimental_allow_widgets=True)
def show_missing_value_plot(df):
    st.pyplot(missing_value_plot(df))


st.session_state["df"] = update_df(st.session_state["df"])

dtype_map_inverse = {
    "float64": "Numeric",
    "category": "Categorical",
    "object": "Categorical",
    "int64": "Numeric",
}

######### Content ##########


# TODO add differing "key" param to buttons -> so that they can have the same name
st.header("Descriptive Table")
desc_switch = st.toggle(label="Create descriptive statistics")
if desc_switch:
    show_desc_table(st.session_state["df"])
# TODO not just object but category dtype as well!
st.header("Distribution of Categorical variables")

cat_options = get_cat_options(st.session_state["df"])
selected_cat = st.selectbox("Select categorical variable", cat_options)

vc_switch = st.toggle(label="Create value counts chart", value=False)
if vc_switch:
    show_vcount_bar_chart(st.session_state["df"], selected_cat)


st.header("Association Figures")
options = st.multiselect(
    label="Choose the desired colums",
    options=st.session_state["df"].columns,
    default=st.session_state["df"].columns[[1, 2]].to_list(),
    max_selections=2,
)
switch_ass = st.toggle(label="Create association figure")
if switch_ass:
    show_ass_chart(st.session_state["df"], options)


st.header("Correlation Matrix")
corr_switch = st.toggle(label="Create correlation matrix")
if corr_switch:
    show_cor_matrix(st.session_state["df"])

st.header("Missing Values")
missing_switch = st.toggle(label="Create missing values chart")
if missing_switch:
    show_missing_value_plot(st.session_state["df"])
