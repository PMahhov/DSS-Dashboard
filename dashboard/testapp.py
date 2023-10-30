import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Create a simple dataframe
df = pd.DataFrame({
    'lat': [52.370216, 52.520008, 48.856613],
    'lon': [4.895168, 13.404954, 2.352222],
    'name': ['Amsterdam', 'Berlin', 'Paris']
})

# Create a simple scatter plot on map
fig = px.scatter_geo(df,
                     lat='lat',
                     lon='lon',
                     text='name',
                     scope='europe')

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='map', figure=fig),
    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('map', 'clickData'))
def update_text(clickData):
    if clickData:
        return f"You clicked on {clickData['points'][0]['text']}"
    return "Click on a point in the map"

if __name__ == '__main__':
    app.run_server(debug=True)
