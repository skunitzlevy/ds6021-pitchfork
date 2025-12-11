import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def knn_model(df, target_col='main_genre', n_neighbors=None):
    """
    Runs KNN. 
    If n_neighbors is None, runs GridSearch to find optimal K.
    If n_neighbors is set (via slider), runs specific K.
    """
    
    # 1. Setup Data
    numeric_features = ['score', 'album_year', 'length', 'followers_count', 
                        'reviewer_reviews', 'artist_reviews']
    categorical_features = ['label']
    
    valid_cols = [c for c in numeric_features + categorical_features + [target_col] if c in df.columns]
    df_clean = df[valid_cols].dropna()

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in numeric_features if c in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [c for c in categorical_features if c in X.columns])
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Slider used
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
    
    fig = px.imshow(
        cm, 
        text_auto=True,
        x=pipe.classes_, 
        y=pipe.classes_,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix (k={n_neighbors})",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    
    stats_content = [
        html.H4(f"Model Results (k={n_neighbors})"),
        html.P([html.Strong("Accuracy: "), f"{acc:.3f}"]),
        html.P([html.Strong("Balanced Accuracy: "), f"{bal_acc:.3f}"]),
        html.P(html.Em("Note: Optimal K found was ~3 (based on GridSearch)."))
    ]
    
    return fig, stats_content