import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1('Ik ben een pagina!'),
    html.Div('En ik ben  een tekstje.'),
])