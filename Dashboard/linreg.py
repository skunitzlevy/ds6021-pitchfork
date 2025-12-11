def run_linear_regression(df, selected_features, degree=1, max_points=2000):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    if not selected_features:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{"text": "Please select at least one variable.", "xref": "paper",
                          "yref": "paper", "showarrow": False, "font": {"size": 20}}]
        )
        return fig, "No features selected."

    target = 'score'
    df_clean = df[selected_features + [target]].copy()
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean.dropna(inplace=True)

    # Downsample for plotting/prediction
    if len(df_clean) > max_points:
        df_plot = df_clean.sample(max_points, random_state=42)
    else:
        df_plot = df_clean.copy()

    X = df_plot[selected_features].astype(float).values
    y = df_plot[target].astype(float).values

    try:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        y_pred = model.predict(X)
    except Exception as e:
        return go.Figure(), f"Model Error: {str(e)}"

    # Stats
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    linear_step = model.named_steps['linearregression']
    intercept = linear_step.intercept_
    raw_coefs = linear_step.coef_

    coef_str = ""
    if degree == 1:
        start_idx = 1 if len(raw_coefs) > len(selected_features) else 0
        current_coefs = raw_coefs[start_idx:]
        for name, val in zip(selected_features, current_coefs):
            coef_str += f"\n• {name}: {val:.4f}"
    else:
        coef_str = f"\n• (Polynomial Degree {degree} - coefficients hidden)"

    stats = (f"**R² Score:** {r2:.4f}\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**Intercept:** {intercept:.4f}\n"
             f"**Coefficients:**{coef_str}\n"
             f"*Note: Plot and model used a sample of {len(df_plot)} rows for memory efficiency.*")

    # Plot
    if len(selected_features) == 1:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_range_pred = model.predict(x_range)

        fig = px.scatter(df_plot, x=selected_features[0], y=target, opacity=0.5,
                         title=f"Linear Regression: {selected_features[0]} vs Score")
        fig.add_trace(go.Scatter(
            x=x_range.flatten(), y=y_range_pred,
            mode='lines', name='Fit Line', line=dict(color='red', width=3)
        ))
    else:
        fig = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Score', 'y': 'Predicted Score'},
                         title="Actual vs Predicted Score", opacity=0.6)
        fig.add_shape(type="line", line=dict(dash="dash", width=3, color="red"),
                      x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max())
        fig.add_annotation(x=y.min(), y=y.max(), text="Perfect Fit (y=x)",
                           showarrow=False, yshift=10)

    fig.update_layout(template="plotly_white")
    return fig, stats
