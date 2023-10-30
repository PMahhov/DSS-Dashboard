# Libraries
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.express as px

#This app consists of a main app.py file, which in turn runs several pages located in the pages folder.

#This creates the navbar that is visible above. It is called later in the app
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

# Initialise the Dash app, using pages and using a custom style sheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], use_pages=True)

# Define the app layout, which includes the navbar and a Dash bootstrap component (dbc) container
# The actual page that is displayed for the main page is the home.py file in the pages folder.
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
