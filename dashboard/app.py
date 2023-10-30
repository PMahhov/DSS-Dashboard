# Libraries
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import geopandas as gpd
import cbsodata
import plotly.express as px

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Crimestat",
    brand_href="#",
    color="primary",
    dark=True,
)

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], use_pages=True)

# Define the app layout and set the initial page content
app.layout = html.Div([
    navbar,
    dbc.Container([ 
        dbc.Row(
            [
                dbc.Col(dash.page_container),
            ]
        ),
    ]),
])
if __name__ == '__main__':
    app.run_server(debug=True)
