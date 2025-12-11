import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

def run_elastic_net(df, selected_features, alpha_val=0, l1_ratio_val=0.5):
    target = 'score'
    
    # 1. Validation
    if not selected_features or len(selected_features) == 0:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{"text": "Please select at least one variable.", "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 20}}]
        )
        return fig, "No features selected."

    # 2. Data Cleaning
    try:
        data = df[selected_features + [target]].copy()
    except KeyError as e:
        return go.Figure(), f"Error: Column not found {e}"

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    if data.empty:
        return go.Figure(), "Error: No valid data points remain."

    y = data[target]
    X = data[selected_features]

    # 3. Preprocessing Setup
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 4. Find Optimal Parameters
    try:
        l1_ratios_to_test = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        cv_model = ElasticNetCV(l1_ratio=l1_ratios_to_test, cv=5, random_state=42, n_jobs=-1, max_iter=20000)
        
        cv_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', cv_model)
        ])
        cv_pipe.fit(X, y)
        
        optimal_alpha = cv_pipe.named_steps['regressor'].alpha_
        optimal_l1 = cv_pipe.named_steps['regressor'].l1_ratio_
    except Exception as e:
        optimal_alpha = 0
        optimal_l1 = 0

    # 5. Determine Active Parameters for Plotting
    if alpha_val == 0:
        active_alpha = optimal_alpha
        active_l1 = optimal_l1
        mode_label = "Automatic (Optimal)"
    else:
        active_alpha = alpha_val
        active_l1 = l1_ratio_val
        mode_label = "Manual Settings"

    # 6. Fit the Active Model
    final_model = ElasticNet(alpha=active_alpha, l1_ratio=active_l1, random_state=42, max_iter=20000)
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', final_model)
    ])

    try:
        pipe.fit(X, y)
    except Exception as e:
        return go.Figure(), f"Model Fitting Error: {str(e)}"

    y_pred = pipe.predict(X)

    # 7. Statistics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Extract Coefficients
    regressor = pipe.named_steps['regressor']
    try:
        feature_names = pipe.named_steps['preprocessor'].get_feature_names_out()
        coeffs = regressor.coef_
        
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeffs})
        coef_df['Abs_Val'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Val', ascending=False)
        
        coef_str = ""
        for _, row in coef_df.head(20).iterrows(): 
            clean_name = row['Feature'].replace('num__', '').replace('cat__', '')
            coef_str += f"\n• {clean_name}: {row['Coefficient']:.4f}"
            
    except Exception:
        coef_str = "\n(Could not extract coefficients)"

    # Formatted Output showing BOTH Current and Optimal
    stats = (f"**Mode:** {mode_label}\n"
             f"**Current Alpha:** {active_alpha:.4f} (Optimal: {optimal_alpha:.4f})\n"
             f"**Current L1 Ratio:** {active_l1:.4f} (Optimal: {optimal_l1:.4f})\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**R² Score:** {r2:.4f}\n"
             f"**Intercept:** {regressor.intercept_:.4f}\n"
             f"**Top Coefficients:**{coef_str}")

    # 8. Plot
    fig = px.scatter(
        x=y, y=y_pred, 
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f"Elastic Net: Actual vs Predicted ({mode_label})",
        opacity=0.6
    )
    fig.add_shape(
        type="line", line=dict(dash="dash", width=3, color="red"),
        x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max()
    )
    fig.add_annotation(
        x=y.min(), y=y.max(),
        text="Perfect Fit (y=x)",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )
    
    fig.update_layout(template="plotly_white")

    return fig, stats