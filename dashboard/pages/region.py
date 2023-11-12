import dash
from dash import dcc, callback, dash_table, html, page_registry
from dash.dash_table import DataTable
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from dash import html
import pandas as pd
from decimal import Decimal
from sqlalchemy import create_engine

dash.register_page(__name__, path_template="/municipality/<stat_code>")
engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")

pd.set_option('max_colwidth', None)

def layout(stat_code=None):
    municipal_name = pd.read_sql_query(f"SELECT municipality_name FROM municipality_names WHERE municipality_id = '{stat_code}'", engine)
    if not (municipal_name.empty or pd.isna(municipal_name.iloc[0]['municipality_name'])):
        municipal_name = municipal_name.iloc[0]['municipality_name']
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics - {municipal_name}'),
            html.Div(f"Municipality code: {stat_code}"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[
                    {'label': str(year), 'value': year} for year in range(2014, 2022)
                ],
                value=2016,
                style={'width': '50%'}
            ),
            html.Div([
                dcc.Graph(id='pie-chart'),
                html.Div([
                    dcc.Loading(
                        id="loading-table",
                        type="circle",
                        children=[html.Div(id="data-table")],
                    )
                ]),
            ])
        ])
    else:
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics'),
            html.Div(f"Municipality code: {stat_code}"),
            dbc.Alert("The requested municipality does not exist", color="danger"),
            ])        

@callback(
    [Output('data-table', 'children'),
     Output('pie-chart', 'figure')],
    [Input('year-dropdown', 'value'),
     Input('url', 'pathname')]
)
def update_data(current_year, pathname):
    print('Current year', current_year)
    stat_code = pathname.split('/')[-1]
    data = pd.read_sql_query(f"SELECT * FROM demo_data WHERE municipality_id = '{stat_code}'", engine)    
    # Create a table component
    filtered_data = data[data['year'] == current_year]
    print('Current filter year', current_year)
    table = generate_table(filtered_data)

    # Create a pie chart
    fig = generate_pie_chart(current_year, data)
    return table, fig

def generate_table(dataframe, max_rows=10):
    columns = ['population', 'household_size', 'population_density', 'avg_income_per_recipient', 'unemployment_rate', 'year']
    # Create DataTable columns
    table_columns = [{'name': col, 'id': col} for col in columns]
    
    return DataTable(
        id='data-table',
        columns=table_columns,
        data=dataframe.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=max_rows
    )
    
def generate_pie_chart(selected_year:int, dataframe):
    # Function to create a pie chart using Plotly Express
    pie_df = dataframe[dataframe['year'] == selected_year]
    if not (pie_df.empty or pd.isna(pie_df.iloc[0]['low_educated_population'])):
        pie_dfs = pie_df.iloc[0]
        print(pie_dfs)

        data = {
            'low_educated_population': pie_dfs['low_educated_population']*100,
            'medium_educated_population': pie_dfs['medium_educated_population']*100,
            'high_educated_population': pie_dfs['high_educated_population']*100
        }
        data = {k:[v] for k,v in data.items()}
        df = pd.DataFrame(data)

        fig = px.pie(df, 
                values=df.iloc[0],
                names=['low_educated_population', 'medium_educated_population', 'high_educated_population'],
                title='Educational Distribution')
    else:
        fig = px.scatter(x=[0], y=[0], text=["No data available"])
        # Update layout for better appearance (optional)
        fig.update_layout(
            width=400,
            height=300,
            title="Educational Distribution",
            template="plotly_white"  # You can choose different templates
        )

        # Hide the axis to make it cleaner
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False) 
        fig.update_traces(marker=dict(size=0))
    return fig