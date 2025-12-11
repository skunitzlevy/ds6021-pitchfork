def run_spline_regression(df, selected_features, knot_quantile=0.5, max_points=2000):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    target = 'score'

    # Validation
    if not selected_features:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{"text": "Please select a variable.", "xref": "paper",
                          "yref": "paper", "showarrow": False}]
        )
        return fig, "No features selected."

    # Data cleaning
    data = df[selected_features + [target]].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # Downsample for memory efficiency
    if len(data) > max_points:
        data_sample = data.sample(max_points, random_state=42)
    else:
        data_sample = data.copy()

    y = data_sample[target].values

    # Feature expansion (splines)
    X_expanded_list = []
    for col in selected_features:
        x_val = data_sample[col].values
        knot = np.quantile(x_val, knot_quantile)
        X_expanded_list.append(x_val.reshape(-1, 1))
        X_expanded_list.append(np.maximum(0, x_val - knot).reshape(-1, 1))

    X_final = np.hstack(X_expanded_list)

    # Fit model
    model = LinearRegression()
    model.fit(X_final, y)
    y_pred = model.predict(X_final)

    # Stats
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    coef_str = ""
    coef_idx = 0
    for name in selected_features:
        base_beta = model.coef_[coef_idx]
        hinge_beta = model.coef_[coef_idx + 1]
        coef_str += f"\n• {name} (Base Slope): {base_beta:.4f}"
        coef_str += f"\n• {name} (Slope Change): {hinge_beta:.4f}"
        coef_idx += 2

    stats = (f"**R² Score:** {r2:.4f}\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**Intercept:** {model.intercept_:.4f}\n"
             f"**Coefficients:**{coef_str}\n"
             f"*Note: Model and plot used a sample of {len(data_sample)} rows for memory efficiency.*")

    # Plotting
    fig = px.scatter(
        x=y, y=y_pred,
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f"Spline Regression (Knots at {int(knot_quantile*100)}%)",
        opacity=0.6
    )
    fig.add_shape(type="line", line=dict(dash="dash", width=3, color="red"),
                  x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max())
    fig.add_annotation(x=y.min(), y=y.max(), text="Perfect Fit (y=x)", showarrow=False, yshift=10)
    fig.update_layout(template="plotly_white")

    return fig, stats
