import altair as alt
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st


# TODO plots take ages to create on data of ~1M rows -> look for better vis options?
# TODO docstring for all the functions, at least input/output description
@st.cache_data
def sample_check(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] > 10_000:
        return df.sample(n=10_000)
    else:
        return df



@st.cache_resource
def desc_table(df: pd.DataFrame):
    return df.describe()


@st.cache_resource
def v_counts_bar_chart(
    df: pd.DataFrame, selected_cat: str, vc_switch: bool
) -> alt.Chart:
    """Creates boxplot of value counts of a categorical variable

    Args:
        v_counts (pd.Series): Value counts for a categorical variable

    Returns:
        alt.Chart: Value counts bar plot
    """

    v_counts = df.value_counts(selected_cat)
    df = v_counts.reset_index()
    df.columns = ["category", "count"]

    fig = alt.Chart(df).mark_bar().encode(x="category:N", y="count:Q")
    return fig


@st.cache_resource
def stacked_bar(df, chosen_features):
    fig = (
        alt.Chart(
            sample_check(df).groupby(chosen_features).size().reset_index(name="Count")
        )
        .mark_bar()
        .encode(
            x=alt.X("Count", stack="normalize"),
            y=f"{chosen_features[0]}:N",
            color=chosen_features[1],
        )
    )
    return fig


@st.cache_resource
def boxplot(df, chosen_features):
    fig = (
        alt.Chart(sample_check(df))
        .mark_boxplot(extent="min-max")
        .encode(
            alt.X(f"{chosen_features[1]}:O", scale=alt.Scale(padding=0)),
            alt.Y(f"{chosen_features[0]}:Q", scale=alt.Scale(padding=0)),
        )
        .configure_boxplot(size=20)
    )
    return fig


@st.cache_resource
def scatter(df, chosen_features):
    fig = (
        alt.Chart(sample_check(df))
        .mark_circle(size=60)
        .encode(alt.X(f"{chosen_features[0]}:Q"), alt.Y(f"{chosen_features[1]}:Q"))
    )
    return fig

@st.cache_resource
def cor_matrix(df):
    n = len(df.select_dtypes(include=["int64", "float64"]).columns)
    df = (
        df.select_dtypes(include=["int64", "float64"])
        .corr()
        .stack()
        .reset_index()
        .rename(
            columns={0: "correlation", "level_0": "variable", "level_1": "variable2"}
        )
    )
    df["correlation_label"] = df["correlation"].map("{:.2f}".format)

    base = alt.Chart(df).encode(x="variable2:O", y="variable:O")

    cor_plot = base.mark_rect().encode(color="correlation:Q")

    text = base.mark_text().encode(
        text="correlation_label",
        color=alt.condition(
            alt.datum.correlation > 0.5, alt.value("white"), alt.value("black")
        ),
    )
    if n > 25:
        return cor_plot
    else:
        return cor_plot + text

@st.cache_resource
def missing_value_plot(df):
    fig, ax = plt.subplots()
    msno.matrix(
        df,
        sparkline=False,
        figsize=(16, 8),
        fontsize=11,
        color=(0.7, 0.57, 0.47),
        ax=ax,
    )
    gray_patch = mpatches.Patch(color="#B29177", label="Data present")
    white_patch = mpatches.Patch(color="white", label="Data absent")

    ax.legend(loc=[1.05, 0.7], handles=[gray_patch, white_patch], fontsize=16)
    return fig
