def knn_model(df, target_col='main_genre', n_neighbors=3, max_samples=2000):
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from dash import html
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

    numeric_features = ['score', 'album_year', 'length', 'followers_count', 
                        'reviewer_reviews', 'artist_reviews']
    categorical_features = ['label']

    # Keep only columns that exist
    valid_cols = [c for c in numeric_features + categorical_features + [target_col] if c in df.columns]
    df_clean = df[valid_cols].dropna()

    # Downsample if too large
    if len(df_clean) > max_samples:
        df_clean = df_clean.sample(max_samples, random_state=42)

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in numeric_features if c in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             [c for c in categorical_features if c in X.columns])
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Confusion matrix (limit to top N classes)
    unique_classes = pipe.classes_
    if len(unique_classes) > 30:
        # Limit to top 30 frequent classes
        top_classes = y_test.value_counts().head(30).index
        mask = y_test.isin(top_classes)
        y_test_plot = y_test[mask]
        y_pred_plot = pd.Series(y_pred, index=y_test.index)[mask]
        labels = top_classes
    else:
        y_test_plot = y_test
        y_pred_plot = y_pred
        labels = unique_classes

    cm = confusion_matrix(y_test_plot, y_pred_plot, labels=labels)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        title=f"Confusion Matrix (k={n_neighbors})",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )

    stats_content = [
        html.H4(f"Model Results (k={n_neighbors})"),
        html.P([html.Strong("Accuracy: "), f"{acc:.3f}"]),
        html.P([html.Strong("Balanced Accuracy: "), f"{bal_acc:.3f}"]),
        html.P(html.Em(f"Note: Data was limited to {len(df_clean)} samples for memory efficiency."))
    ]

    return fig, stats_content
