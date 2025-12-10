from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, html, dcc, Input, Output, dash_table

def run_linear_regression(df, selected_features):
    
    # 1. Handle Empty Selection
    if not selected_features:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Please select at least one variable",
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return empty_fig, "No variables selected."
    

    target_col = 'score' 
    temp_df = df[selected_features + [target_col]]
    temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    temp_df.dropna(inplace=True)
    
    X = temp_df[selected_features]
    y = temp_df[target_col]

    # 3. Fit Model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # 4. Create Visualization (Actual vs Predicted)
    fig = px.scatter(
        x=y, 
        y=y_pred, 
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f'Actual vs. Predicted Score (n={len(temp_df)})',
        opacity=0.6,
        template='plotly_white'
    )
    fig.add_shape(
            type="line",
            line=dict(dash="dash", width=3, color="red"),
            x0=y.min(), x1=y.max(),
            y0=y.min(), y1=y.max()
        )
        
    fig.add_annotation(
        x=y.min(), y=y.max(),
        text="Perfect Fit (y=x)",
        showarrow=False,
        yshift=10
    )

    # 5. Format Statistics
    coeffs = list(zip(selected_features, model.coef_))
    coeff_text = [f"Intercept: {model.intercept_:.4f}"]
    for name, val in coeffs:
        coeff_text.append(f"{name}: {val:.4f}")
    
    stats_markdown = [
        html.P([html.Strong("R-Squared: "), f"{r2:.4f}"]),
        html.P([html.Strong("Samples Used: "), f"{len(temp_df)}"]),
        html.Hr(),
        html.P(html.Strong("Coefficients:")),
        html.Ul([html.Li(c) for c in coeff_text], style={'paddingLeft': '20px'})
    ]

    return fig, stats_markdown