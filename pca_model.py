import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def run_pca(df):

    pca_df = df[df["followers_count"] > 0].copy()

    pca_df["log_followers"] = np.log10(pca_df["followers_count"])

    features = [
        "score",
        "log_followers",
        "length",
        "review_year",
        "review_release_difference",
        "reviewer_reviews",
        "artist_reviews",
        "genre_len"
    ]
    X = pca_df[features].dropna()

    genres = pca_df.loc[X.index, "main_genre"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    figs = {}

    # Variance explained
    figs["scree"] = go.Figure(
        data=[go.Scatter(x=list(range(1, len(explained)+1)), y=explained, mode="lines+markers")],
        layout=go.Layout(title="Variance Explained by Component", xaxis_title="Principal Component", yaxis_title="Variance Explained")
    )

    # Cumulative variance
    figs["cumulative"] = go.Figure(
        data=[go.Scatter(x=list(range(1, len(explained)+1)), y=np.cumsum(explained), mode="lines+markers")],
        layout=go.Layout(title="Cumulative Variance Explained", xaxis_title="Principal Component", yaxis_title="Cumulative Variance")
    )

    # PC1 vs PC2 scatter
    pc_df = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1], "Genre": genres})
    figs["scatter"] = px.scatter(
        pc_df, x="PC1", y="PC2", color="Genre",
        title="PCA: PC1 vs PC2",
        labels={"PC1":"PC1", "PC2":"PC2", "Genre":"Main Genre"},
        width=800, height=600
    )

    # Loadings table
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(features))],
        index=features
    )
    figs["loadings"] = loadings

    return figs