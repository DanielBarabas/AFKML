import altair as alt
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def stacked_bar(df, chosen_features):
    fig = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count()").stack("normalize"),
            y=chosen_features[0],
            color=chosen_features[1],
        )
    )
    return fig


def boxplot(df, chosen_features):
    fig = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            alt.X(f"{chosen_features[0]}:Q").scale(zero=False),
            alt.Y(f"{chosen_features[1]}:N"),
        )
    )
    return fig


def scatter(df, chosen_features):
    fig = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(alt.X(f"{chosen_features[0]}:Q"), alt.Y(f"{chosen_features[1]}:Q"))
    )
    return fig


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
