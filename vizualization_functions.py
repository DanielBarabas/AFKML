import altair as alt
import pandas as pd


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
