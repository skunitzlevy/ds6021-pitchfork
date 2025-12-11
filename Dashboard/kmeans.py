import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA # Import PCA

def cluster_and_plot_latent(df, numeric_cols, categorical_cols, 
                            true_label_col='main_genre', n_clusters=5):
    """
    Preprocesses data, reduces dimensions to 2 latent features using PCA,
    clusters the data, and plots the results on the latent axes.
    """
    
    # 1. Prepare Data
    features = numeric_cols + categorical_cols
    X = df[features].copy()
    true_labels = df[true_label_col]

    X = X.dropna()
    true_labels = true_labels.loc[X.index]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    X_processed = preprocessor.fit_transform(X)

    # 2. Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_processed)

    # Dimensionality Reduction (PCA)
    # Compress the processed data into 2 latent features
    pca = PCA(n_components=2)
    latent_components = pca.fit_transform(X_processed)
    
    explained_variance = pca.explained_variance_ratio_.sum() * 100

    # 3. Setup Plot Data
    plot_df = X.copy().reset_index(drop=True)
    plot_df['True Genre'] = true_labels.reset_index(drop=True)
    plot_df['Cluster'] = cluster_labels.astype(str)
    
    plot_df['Latent Feature 1'] = latent_components[:, 0]
    plot_df['Latent Feature 2'] = latent_components[:, 1]
    
    for col in ['reviewer_reviews']:
        if col not in plot_df.columns and col in df.columns:
            plot_df[col] = df[col].reset_index(drop=True)

    # 4. Plot using Latent Features as axes
    fig = px.scatter(
        plot_df, 
        x='Latent Feature 1', 
        y='Latent Feature 2', 
        color='Cluster',
        title=f'K-Means Clusters on Latent Features (PCA) - {explained_variance:.1f}% Variance Captured',
        labels={'Cluster': 'Cluster Label'},
        hover_data=['True Genre', 'reviewer_reviews', 'score', 'log_length'] 
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    
    return plot_df, kmeans, pca, fig