import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, dash_table
from summary_charts import summary_charts
from pca_model import run_pca
from linreg import run_linear_regression 
from KNN import knn_model
import os

df = pd.read_csv('../data/clean/Cleaned_Data.csv')

app = Dash(__name__)
server = app.server

ROW_OPTIONS = [25, 50, 100]

card_style = {
    'padding': '20px', 
    'margin': '20px 0', 
    'borderRadius': '10px', 
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
    'backgroundColor': '#ffffff'
}

slider_style = {'margin': '15px 0'}

# App Layout
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8', 'padding': '20px'}, children=[
    html.H1("🎵 Pitchfork Project Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    dcc.Tabs([
        # README Tab
        dcc.Tab(label="README", children=[
            html.Div(style=card_style, children=[
                html.H3("Project README"),
                html.P("This is a placeholder for README content. Describe your project here, your team, dataset, and objectives."),
            ])
        ]),

        # Data Preview Tab
        dcc.Tab(label="Data Preview", children=[
            html.Div(style=card_style, children=[
                html.Div([
                    html.Label("Rows to display:"),
                    dcc.Dropdown(
                        id='rows-dropdown',
                        options=[{'label': str(r), 'value': r} for r in ROW_OPTIONS],
                        value=25,
                        clearable=False,
                        style={'width': '120px'}
                    )
                ], style={'marginBottom': '20px'}),

                dash_table.DataTable(
                    id='data-table',
                    columns=[{"name": col, "id": col} for col in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'minWidth': '120px',
                        'width': '120px',
                        'maxWidth': '120px',
                        'whiteSpace': 'normal'
                    },
                    page_size=25
                )
            ])
        ]),

        # EDA Tab
        dcc.Tab(label="EDA", children=[
            html.Div(style=card_style, children=[
                html.H3("Exploratory Data Analysis"),
                dcc.Graph(id='followers_vs_score_by_artist'),
                dcc.Graph(id='score_vs_length'),
                dcc.Graph(id='pitchfork_score_distribution'),
                dcc.Graph(id='pitchfork_review_counts'),
                dcc.Graph(id='genre_score_boxplot'),
                dcc.Graph(id='top_artists_by_reviews'),
                html.Div(id='dummy-eda-trigger', style={'display': 'none'})
            ])
        ]),

        # Models Tab
        dcc.Tab(label="Models", children=[
            html.Div(style=card_style, children=[
                html.H3("Model Filters & Outputs"),

                html.Div([
                    html.Label("Album Year Range:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=int(df['album_year'].dropna().min()),
                        max=int(df['album_year'].dropna().max()),
                        step=1,
                        value=[int(df['album_year'].min()), int(df['album_year'].max())],
                        marks={y: str(y) for y in range(int(df['album_year'].min()), int(df['album_year'].max())+1, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style=slider_style),

                html.Div([
                    html.Label("Followers Count Range:"),
                    dcc.RangeSlider(
                        id='followers-slider',
                        min=int(df['followers_count'].dropna().min()),
                        max=int(df['followers_count'].dropna().max()),
                        value=[int(df['followers_count'].min()), int(df['followers_count'].max())],
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style=slider_style),

                html.Div([
                    html.Label("Train/Test Split (%)"),
                    dcc.Slider(
                        id='train-test-slider',
                        min=50,
                        max=90,
                        step=5,
                        value=80,
                        marks={i: f"{i}%" for i in range(50, 95, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style=slider_style),

                html.Div(id='model-output', children=[
                    html.P("Model outputs will appear here.")
                ], style={'marginTop': '30px'})
            ]), 

            html.Div(style=card_style, children=[
                html.H3("Linear Regression"),
                html.P("Toggle variables below to predict the Pitchfork Score.", style={'marginBottom': '15px'}),
                
                html.Div([
                    html.Label(html.Strong("Select Independent Variables (X):")),
                    dcc.Checklist(
                        id='lr-feature-checklist',
                        options=[{'label': col, 'value': col} for col in df.select_dtypes(include=[np.number]).columns if col != 'score'],
                        value=[col for col in df.select_dtypes(include=[np.number]).columns if col != 'score'][:2], 
                        inline=True,
                        style={'fontSize': '16px', 'marginTop': '10px'}
                    ),
                ], style={'padding': '15px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px', 'borderRadius': '5px'}),

                html.Div([
                    # Graph Section
                    html.Div([
                        dcc.Graph(id='lr-graph')
                    ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                    
                    # Stats Section
                    html.Div([
                        html.H4("Model Statistics"),
                        html.Div(id='lr-stats', style={'whiteSpace': 'pre-line', 'fontSize': '15px'})
                    ], style={'width': '30%', 'display': 'inline-block', 'paddingLeft': '20px', 'verticalAlign': 'top'})
                ])
            ]),

            html.Div(style=card_style, children=[
                html.H3("K-Nearest Neighbors Classifier"),
                html.P("Train a KNN model to classify the target (e.g., Genre) based on audio features.", style={'marginBottom': '15px'}),
                
                dcc.Loading(id="loading-knn", type="default", children=[
                    html.Div([
                        # Graph Section
                        html.Div([
                            dcc.Graph(id='knn-graph')
                        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginTop': '20px'}),
                        
                        # Stats Section
                        html.Div(id='knn-stats', style={'width': '30%', 'display': 'inline-block', 'paddingLeft': '20px', 'verticalAlign': 'top', 'marginTop': '20px'})
                    ])
                ])
            ])
        ]),

        # PCA Tab
        dcc.Tab(label="PCA Analysis", children=[
            html.Div(style=card_style, children=[
                html.H3("Principal Component Analysis"),
                dcc.Graph(id="pca_scree_plot"),
                dcc.Graph(id="pca_cumulative_plot"),
                dcc.Graph(id="pca_scatter_plot"),
                dash_table.DataTable(id="pca_loadings_table",
                                    columns=[{"name": col, "id": col} for col in ["Feature"] + [f"PC{i}" for i in range(1,9)]],
                                    style_table={'overflowX':'auto'},
                                    style_cell={'textAlign':'left'})
            ])
        ])
    ])
])

# Callbacks
@app.callback(
    Output('data-table', 'data'),
    Output('data-table', 'page_size'),
    Input('rows-dropdown', 'value')
)
def update_table_rows(selected_rows):
    return df.head(selected_rows).to_dict('records'), selected_rows

@app.callback(
    Output("followers_vs_score_by_artist", "figure"),
    Output("score_vs_length", "figure"),
    Output("pitchfork_score_distribution", "figure"),
    Output("pitchfork_review_counts", "figure"),
    Output("genre_score_boxplot", "figure"),
    Output("top_artists_by_reviews", "figure"),
    Input("dummy-eda-trigger", "children")
)
def update_eda(_):
    figs = summary_charts(df)
    return (
        figs["followers_vs_score_by_artist"],
        figs["score_vs_length"],
        figs["pitchfork_score_distribution"],
        figs["pitchfork_review_counts"],
        figs["genre_score_boxplot"],
        figs["top_artists_by_reviews"]
    )

@app.callback(
    Output("pca_scree_plot", "figure"),
    Output("pca_cumulative_plot", "figure"),
    Output("pca_scatter_plot", "figure"),
    Output("pca_loadings_table", "data"),
    Input("dummy-eda-trigger", "children")
)
def update_pca(_):
    figs = run_pca(df)
    loadings_df = figs["loadings"].reset_index().rename(columns={"index":"Feature"})
    return figs["scree"], figs["cumulative"], figs["scatter"], loadings_df.to_dict("records")

# Linear Regression Callback
@app.callback(
    [Output('lr-graph', 'figure'),
     Output('lr-stats', 'children')],
    [Input('lr-feature-checklist', 'value')]
)
def update_lr_graph(selected_features):
    fig, stats = run_linear_regression(df, selected_features)
    return fig, stats

# KNN Callback
@app.callback(
    [Output('knn-graph', 'figure'),
     Output('knn-stats', 'children')],
     [Input('dummy-eda-trigger', 'children')]
)
def update_knn_model(_):
    return knn_model(df, target_col='genre')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = True
    app.run(debug=debug, host="0.0.0.0", port=port)