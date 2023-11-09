import dash
from dash import html

dash.register_page(__name__, path_template="/region/<stat_code>")

def layout(stat_code=None):
    return html.Div([
    html.H1('View municipal statistics'),
    html.Div(f"Municipality code: {stat_code}"),
])