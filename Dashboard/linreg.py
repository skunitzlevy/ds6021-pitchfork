import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def run_linear_regression(df, selected_features, degree=1):
    target = 'score'

    # 1. Validation
    if not selected_features:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{"text": "Please select at least one variable.", "xref": "paper", 
                          "yref": "paper", "showarrow": False, "font": {"size": 20}}]
        )
        return fig, "No features selected."

    # 2. Data Preparation
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

    # 3. Preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    #standard scale numeric features
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    #handle categorical features
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)

    # 4. Build Pipeline
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', LinearRegression())
    ])

    # 5. Fit Model
    try:
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
    except Exception as e:
        return go.Figure(), f"Model Error: {str(e)}"

    # 6. Statistics & Coefficients
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    #get intercept
    intercept = pipe.named_steps['regressor'].intercept_

    #get coefficients
    try:
        feature_names_transformed = pipe.named_steps['preprocessor'].get_feature_names_out()
        final_feature_names = pipe.named_steps['poly'].get_feature_names_out(feature_names_transformed)
        coeffs = pipe.named_steps['regressor'].coef_
        coef_df = pd.DataFrame({'Feature': final_feature_names, 'Coefficient': coeffs})
        coef_df['Abs_Val'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Val', ascending=False)

        coef_str = ""
        #only display 20
        display_limit = 20 
        for _, row in coef_df.head(display_limit).iterrows():
            # Clean names
            clean_name = row['Feature'].replace('num__', '').replace('cat__', '')
            coef_str += f"\n• {clean_name}: {row['Coefficient']:.4f}"

    except Exception as e:
        coef_str = f"\n(Could not extract coefficients: {e})"

    stats = (f"**Degree:** {degree}\n"
             f"**RMSE:** {rmse:.4f}\n"
             f"**R² Score:** {r2:.4f}\n"
             f"**Intercept:** {intercept:.4f}\n"
             f"**Top Coefficients:**{coef_str}")

    # 7. Generate Plot
    fig = px.scatter(
        x=y, y=y_pred, 
        labels={'x': 'Actual Score', 'y': 'Predicted Score'},
        title=f"Actual vs Predicted Score (Degree {degree})",
        opacity=0.6
    )
    fig.add_shape(type="line", line=dict(dash="dash", width=3, color="red"),
                    x0=y.min(), x1=y.max(), y0=y.min(), y1=y.max())
    fig.add_annotation(x=y.min(), y=y.max(), text="Perfect Fit (y=x)", showarrow=False, yshift=10)

    fig.update_layout(template="plotly_white")
    return fig, stats