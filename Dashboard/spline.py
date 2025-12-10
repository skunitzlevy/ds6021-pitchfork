import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def run_spline_regression(df, selected_features, knot_quantile=0.5):
    """
    Fits an additive spline regression on multiple variables.
    Plots Predicted vs. Actual score.
    """
    target = 'score'
    
    # 1. Input Validation
    if not selected_features or len(selected_features) == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, 
            yaxis={'visible': False},
            annotations=[{
                "text": "Please select at least one variable.",
                "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 20}
            }]
        )
        return fig, "No features selected."

    # 2. Data Preparation
    # Create a clean copy
    data = df[selected_features + [target]].copy()
    
    # Handle Infinite/NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    if data.empty:
        return go.Figure(), "Error: No valid data after cleaning."

    y = data[target].values
    
    # 3. Feature Engineering (Basis Expansion)
    # For every selected feature X, we create two columns for the model:
    # 1. The original X
    # 2. The spline term: max(0, X - knot)
    
    X_expanded_list = []
    knot_info = []

    for col in selected_features:
        x_col_values = data[col].values
        
        # Calculate knot for this specific variable
        knot_val = np.quantile(x_col_values, knot_quantile)
        knot_info.append(f"{col}: {knot_val:.2f}")
        
        # 1. Base Term
        X_expanded_list.append(x_col_values.reshape(-1, 1))
        
        # 2. Spline Term (The "Bend")
        spline_term = np.maximum(0, x_col_values - knot_val).reshape(-1, 1)
        X_expanded_list.append(spline_term)

    # Combine all features into one matrix
    X_final = np.hstack(X_expanded_list)

    # 4. Fit Model
    model = LinearRegression()
    model.fit(X_final, y)
    y_pred = model.predict(X_final)

    # 5. Calculate Stats
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    stats = (f"**Global Knot Quantile:** {knot_quantile:.2f}\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**R² Score:** {r2:.4f}\n"
             f"**Features:** {len(selected_features)} selected")

    # 6. Plotting: Predicted vs Actual
    # Since we have multiple dimensions, we must plot y vs y_pred
    fig = px.scatter(
        x=y, 
        y=y_pred, 
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f"Spline Regression: Actual vs Predicted (Knots at {int(knot_quantile*100)}%)",
        opacity=0.6
    )

    # Add Perfect Fit Line (y=x)
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

    fig.update_layout(template="plotly_white")

    return fig, stats