def run_elastic_net(df, selected_features, alpha_val=None, l1_ratio_val=0.5, max_points_plot=1000):
    import warnings
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    import plotly.express as px
    import pandas as pd
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning

    target = 'score'
    
    # --- 1. Validation ---
    if not selected_features:
        fig = px.scatter(title="Please select at least one variable.")
        return fig, "No features selected."

    # --- 2. Data cleaning ---
    data = df[selected_features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return px.scatter(title="No valid data points remain."), "No valid data points."

    y = data[target]
    X = data[selected_features]

    # --- 3. Preprocessing ---
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)

    # --- 4. Determine alpha and l1_ratio ---
    min_alpha = 1e-4
    mode_label = "Manual Settings"
    if alpha_val is None or alpha_val <= 0:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                cv_model = ElasticNetCV(
                    l1_ratio=[0.5, 0.7, 0.9],
                    cv=3,
                    n_jobs=1,
                    max_iter=5000,
                    random_state=42
                )
                pipe_cv = Pipeline([('preprocessor', preprocessor), ('regressor', cv_model)])
                pipe_cv.fit(X, y)
                alpha_val = max(pipe_cv.named_steps['regressor'].alpha_, min_alpha)
                l1_ratio_val = pipe_cv.named_steps['regressor'].l1_ratio_
                mode_label = "Automatic (Optimal)"
        except Exception:
            alpha_val = min_alpha
            l1_ratio_val = 0.5
            mode_label = "Automatic (Fallback)"
    else:
        alpha_val = max(alpha_val, min_alpha)

    # --- 5. Fit final Elastic Net ---
    model = ElasticNet(alpha=alpha_val, l1_ratio=l1_ratio_val, max_iter=5000, random_state=42)
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', model)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipe.fit(X, y)

    y_pred = pipe.predict(X)

    # --- 6. Stats ---
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # Extract coefficients (top 20)
    try:
        regressor = pipe.named_steps['regressor']
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': regressor.coef_})
        coef_df['Abs_Val'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Val', ascending=False)
        coef_str = ""
        for _, row in coef_df.head(20).iterrows():
            clean_name = row['Feature'].replace('num__', '').replace('cat__', '')
            coef_str += f"\n• {clean_name}: {row['Coefficient']:.4f}"
    except Exception:
        coef_str = "\n(Could not extract coefficients)"

    stats = (f"**Mode:** {mode_label}\n"
             f"**Alpha:** {alpha_val:.5f}\n"
             f"**L1 Ratio:** {l1_ratio_val:.4f}\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**R² Score:** {r2:.4f}\n"
             f"**Intercept:** {regressor.intercept_:.4f}\n"
             f"**Top Coefficients:**{coef_str}")

    # --- 7. Plot (downsample if large) ---
    if len(X) > max_points_plot:
        plot_idx = np.random.choice(len(X), max_points_plot, replace=False)
        plot_y, plot_y_pred = y.iloc[plot_idx], y_pred[plot_idx]
    else:
        plot_y, plot_y_pred = y, y_pred

    fig = px.scatter(
        x=plot_y, y=plot_y_pred,
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f"Elastic Net: Actual vs Predicted ({mode_label})",
        opacity=0.6
    )
    fig.add_shape(type="line", line=dict(dash="dash", width=3, color="red"),
                  x0=plot_y.min(), x1=plot_y.max(), y0=plot_y.min(), y1=plot_y.max())
    fig.add_annotation(x=plot_y.min(), y=plot_y.max(),
                       text="Perfect Fit (y=x)", showarrow=False, yshift=10, font=dict(color="red"))
    fig.update_layout(template="plotly_white")

    return fig, stats
