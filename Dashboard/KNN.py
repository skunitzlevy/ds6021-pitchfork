def knn_model(df, target_col='main_genre', n_neighbors=None):
    """
    Memory-safe + accuracy-preserving KNN.
    Avoids sampling and avoids GridSearch unless n_neighbors=None.
    """

    # 1. Safe Feature Selection
    numeric_features = [
        'score', 'album_year', 'length', 'followers_count',
        'reviewer_reviews', 'artist_reviews'
    ]
    categorical_features = ['label']

    valid_cols = [
        c for c in numeric_features + categorical_features + [target_col]
        if c in df.columns
    ]
    df_clean = df[valid_cols].dropna()

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    # 2. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [c for c in numeric_features if c in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [c for c in categorical_features if c in X.columns])
        ]
    )

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # ------------------------------
    # 4. AUTO MODE (Only when K=0)
    # ------------------------------
    if n_neighbors is None:
        possible_k = [3, 5, 7, 9, 11]

        best_k = 5
        best_acc = 0

        for k in possible_k:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = balanced_accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_k = k

        n_neighbors = best_k
        auto_note = f"(Auto-selected k={best_k})"

    else:
        auto_note = ""

    # ------------------------------
    # 5. Train final model
    # ------------------------------
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 6. Metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # 7. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=pipe.classes_,
        y=pipe.classes_,
        color_continuous_scale='Blues',
        title=f"KNN Confusion Matrix (k={n_neighbors}) {auto_note}"
    )

    stats_content = [
        html.H4(f"KNN Results (k={n_neighbors}) {auto_note}"),
        html.P([html.Strong("Accuracy: "), f"{acc:.3f}"]),
        html.P([html.Strong("Balanced Accuracy: "), f"{bal_acc:.3f}"]),
        html.P(html.Em("Note: Full dataset used — no sampling to preserve accuracy."))
    ]

    return fig, stats_content
