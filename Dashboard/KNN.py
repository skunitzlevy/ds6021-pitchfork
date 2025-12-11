import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def knn_model(df, target_col='main_genre', n_neighbors=5):
    """
    Low-memory KNN for Render.
    - No one-hot encoding (uses hashing instead)
    - No ColumnTransformer
    - No GridSearch
    - Uses KD-Tree KNN for memory efficiency
    """

    # -----------------------------
    # 1. Select usable features
    # -----------------------------
    numeric_features = ['score', 'album_year', 'length', 
                        'followers_count', 'reviewer_reviews', 'artist_reviews']
    categorical_features = ['label']

    available_numeric = [c for c in numeric_features if c in df.columns]
    available_cat = [c for c in categorical_features if c in df.columns]

    cols = available_numeric + available_cat + [target_col]
    data = df[cols].dropna().copy()

    if data.empty:
        return go.Figure(), [html.P("No usable data available.")]

    # -----------------------------
    # 2. Simple category hashing (fixed memory)
    # -----------------------------
    def hash_category(col):
        return col.astype(str).apply(lambda x: hash(x) % 5000)

    for c in available_cat:
        data[c] = hash_category(data[c])

    # -----------------------------
    # 3. Split
    # -----------------------------
    np.random.seed(42)
    mask = np.random.rand(len(data)) < 0.8
    train = data[mask]
    test = data[~mask]

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # -----------------------------
    # 4. Scale numeric features only (fast)
    # -----------------------------
    scaler = StandardScaler()
    X_train[available_numeric] = scaler.fit_transform(X_train[available_numeric])
    X_test[available_numeric] = scaler.transform(X_test[available_numeric])

    # -----------------------------
    # 5. KNN — use KD-Tree (memory-efficient)
    # -----------------------------
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        algorithm="kd_tree",
        leaf_size=50
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # -----------------------------
    # 6. Metrics
    # -----------------------------
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # -----------------------------
    # 7. Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=knn.classes_,
        y=knn.classes_,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix (k={n_neighbors})",
    )

    stats_content = [
        html.H4(f"KNN Results (k={n_neighbors}) — Low-Memory Mode"),
        html.P([html.Strong("Accuracy: "), f"{acc:.3f}"]),
        html.P([html.Strong("Balanced Accuracy: "), f"{bal_acc:.3f}"]),
        html.P(html.Em("Memory-optimized mode enabled: hashing + KD-Tree")),
    ]

    return fig, stats_content
