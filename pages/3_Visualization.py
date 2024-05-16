import streamlit as st
import pandas as pd
import altair as alt
from vizualization_functions import boxplot, stacked_bar, scatter

st.set_page_config(page_title="Visualization", layout="wide")
alt.data_transformers.disable_max_rows()
cat_list = ["object", "category"]
exp = st.expander(label="Association Figures")


small_df = st.session_state["df"].sample(n=10_000)
with exp:
    options = st.multiselect(
        label="Choose the desired colums",
        options=small_df.columns,
        default=small_df.columns[[1, 2]].to_list(),
        max_selections=2,
    )
    if len(options) == 2:
        st.write("Ok")
        # Both category:
        if (
            small_df[options[0]].dtype in cat_list
            and small_df[options[1]].dtype in cat_list
        ):
            fig = stacked_bar(small_df, options)
        # Both continous:
        elif (
            small_df[options[0]].dtype not in cat_list
            and small_df[options[1]].dtype not in cat_list
        ):
            fig = scatter(small_df, options)
        # One is continous, other is not:
        else:
            fig = boxplot(small_df, options)
        st.altair_chart(fig, use_container_width=True)
