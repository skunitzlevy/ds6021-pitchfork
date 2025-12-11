import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

def summary_charts(df: pd.DataFrame) -> Dict[str, object]:

    figs = {}

    df1 = df.dropna(subset=['followers_count', 'score', 'length', 'review_release_difference']).copy()

    artist_df = df1.groupby('artist')[['followers_count', 'score', 'main_genre']].agg({
        'followers_count': 'mean',
        'score': 'mean',
        'main_genre': 'first'
    }).reset_index()
    artist_df = artist_df.dropna(subset=['followers_count', 'score'])
    artist_df['followers_count'] = artist_df['followers_count'].replace(0, 1)
    artist_df['log_followers'] = np.log10(artist_df['followers_count'])

    if len(artist_df) >= 2:
        coef_artist = np.polyfit(artist_df['log_followers'], artist_df['score'], 1)
        x_artist_line = np.linspace(artist_df['log_followers'].min(), artist_df['log_followers'].max(), 100)
        y_artist_line = coef_artist[0] * x_artist_line + coef_artist[1]
    else:
        x_artist_line, y_artist_line = [], []

    # Chart: followers_vs_score_by_artist
    followers_vs_score_by_artist = px.scatter(
        artist_df,
        x='log_followers',
        y='score',
        color='main_genre',
        title='Mean Score vs Log10(Followers Count) by Artist',
        height=500
    )

    if len(x_artist_line) > 0:
        trendline_artist = go.Scatter(
            x=x_artist_line,
            y=y_artist_line,
            mode='lines',
            line=dict(color='black', width=4),
            name='OLS Trendline',
            hoverinfo='skip'
        )
        followers_vs_score_by_artist.add_trace(trendline_artist)

    figs['followers_vs_score_by_artist'] = followers_vs_score_by_artist

    df1 = df1[df1['length'] > 0].copy()
    df1['log_length'] = np.log(df1['length'])
    df1['age_category'] = pd.cut(
        df1['review_release_difference'],
        bins=4,
        labels=['New', 'Recent', 'Old', 'Vintage']
    )

    if len(df1) >= 2:
        coef_length = np.polyfit(df1['log_length'], df1['score'], 1)
        x_length_line = np.linspace(df1['log_length'].min(), df1['log_length'].max(), 100)
        y_length_line = coef_length[0] * x_length_line + coef_length[1]
    else:
        x_length_line, y_length_line = [], []

    # Chart: score_vs_length
    score_vs_length = px.scatter(
        df1,
        x='log_length',
        y='score',
        color='age_category',
        title='Score vs Log Review Length',
        height=500
    )

    if len(x_length_line) > 0:
        trendline_length = go.Scatter(
            x=x_length_line,
            y=y_length_line,
            mode='lines',
            line=dict(color='black', width=4),
            name='OLS Trendline',
            hoverinfo='skip'
        )
        score_vs_length.add_trace(trendline_length)

    figs['score_vs_length'] = score_vs_length

    # Chart: pitchfork_score_distribution
    pitchfork_score_distribution = px.histogram(
        df,
        x='score',
        nbins=20,
        title='Distribution of Album Review Scores',
        labels={'score': 'Score'},
        height=450
    )
    figs['pitchfork_score_distribution'] = pitchfork_score_distribution

    # Chart: pitchfork_review_counts
    top5 = df['genre'].value_counts().head(5).sort_values(ascending=True)
    pitchfork_review_counts = px.bar(
        x=top5.values,
        y=top5.index,
        orientation='h',
        title='Top 5 Most Reviewed Genres',
        labels={'x': 'Number of Reviews', 'y': 'Genre'},
        height=450
    )
    figs['pitchfork_review_counts'] = pitchfork_review_counts

    # Chart: genre_score_boxplot
    temp = df.copy()

    valid_genres = (
        temp["main_genre"]
        .value_counts()
        .loc[lambda s: s >= 20]
        .index
    )

    filtered = temp[temp["main_genre"].isin(valid_genres)]

    genre_fig = px.box(
        filtered,
        x="main_genre",
        y="score",
        title="Score by Main Genre",
        labels={"main_genre": "Main Genre", "score": "Score"},
    )

    genre_fig.update_layout(
        xaxis_tickangle=45,
        margin=dict(l=20, r=20, t=50, b=80),
        template="plotly_white"
    )

    figs["genre_score_boxplot"] = genre_fig

    #Chart: top_artists_by_reviews
    filtered_df = df[df["artist"] != "Various Artists"]

    top_artists_by_n = (
        filtered_df.groupby("artist")
                .size()
                .sort_values(ascending=False)
                .head(10)
                .reset_index(name="n_reviews")
    )

    top_artists_by_n = top_artists_by_n.sort_values("n_reviews", ascending=True)

    top_artists_fig = px.bar(
        top_artists_by_n,
        x="n_reviews",
        y="artist",
        orientation="h",
        title="Top 10 Artists by Number of Reviews",
        labels={"n_reviews": "Number of Reviews", "artist": "Artist"}
    )

    top_artists_fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=40)
    )

    figs["top_artists_by_reviews"] = top_artists_fig

    return figs
