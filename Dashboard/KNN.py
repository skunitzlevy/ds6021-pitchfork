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

def knn_model(df, target_col='main_genre'):
    #set features for KNN
    numeric_features = ['score', 'album_year', 'length', 'followers_count', 
                        'reviewer_reviews', 'artist_reviews']
    categorical_features = ['label']
    target = target_col

    df_clean = df[numeric_features + categorical_features+ [target]].dropna()
    
    #prepare data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(weights="distance"))
    ])

    # Train-Test
    X = df[numeric_features + categorical_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"knn__n_neighbors": range(1, 41, 2)}

    #set up for cross-validation (5 folds)
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    results_df = pd.DataFrame(grid.cv_results_)

    results_df["k"] = results_df["param_knn__n_neighbors"]
    results_df["mean_score"] = results_df["mean_test_score"]

    best_k = grid.best_params_["knn__n_neighbors"]
    best_score = grid.best_score_

    #plot the best K to use
    fig = px.line(
        results_df,
        x="k",
        y="mean_score",
        title=f"Cross-Validated Balanced Accuracy vs. K (best k = {best_k})",
        markers=True,
        labels={"k": "Number of Neighbors (k)", "mean_score": "Mean CV Balanced Accuracy"}
    )
    fig.add_scatter(
        x=[best_k],
        y=[best_score],
        mode="markers+text",
        text=[f"Best k = {best_k}"],
        textposition="top center",
        name="Best k"
    )
    fig.update_layout(hovermode="x unified", template="plotly_white")

    #Model on test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Create Confusion Matrix Text
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    
    # Return Output
    stats_content = [
        html.H4("Model Performance (Test Set)"),
        html.P([html.Strong("Best K: "), str(best_k)]),
        html.P([html.Strong("Accuracy: "), f"{acc:.3f}"]),
        html.P([html.Strong("Balanced Accuracy: "), f"{bal_acc:.3f}"]),
        html.Hr(),
        html.P(html.Strong("Confusion Matrix:")),
        # Simple rendering of the matrix
        html.Pre(str(cm), style={'backgroundColor': '#f0f0f0', 'padding': '10px'})
    ]

    return fig, stats_content