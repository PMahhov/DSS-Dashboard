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

#metadata = pd.DataFrame(cbsodata.get_meta('85039NED', 'DataProperties'))
data = pd.DataFrame(cbsodata.get_data('85039NED', select = ['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
data['Codering_3'] = data['Codering_3'].str.strip()
geodata_url = 'https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson'
municipal_boundaries = gpd.read_file(geodata_url)
municipal_boundaries = pd.merge(municipal_boundaries, data,
                                left_on="statcode",
                                right_on="Codering_3")
municipal_boundaries = municipal_boundaries.to_crs(epsg=4326)
gdf_choro = municipal_boundaries.copy()
gdf_choro['geoid'] = gdf_choro.index.astype(str)
gdf_choro = gdf_choro[['geoid', 'geometry', 'statnaam', 'GeboorteRelatief_25']]

fig = px.choropleth_mapbox(gdf_choro,
                           geojson=gdf_choro.__geo_interface__,
                           locations=gdf_choro.geoid,
                           color='GeboorteRelatief_25',
                           featureidkey='properties.geoid',
                           center={'lat': 52.213, 'lon':5.2794},
                           mapbox_style='carto-positron',
                           zoom=6)

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div(children=[
    html.H1(children='CrimeStat'),
    html.Div(children='''
        CrimeStat: comparing your municipality with everything.
    '''),
    dcc.Graph(id='plotly_map', figure=fig),
    html.Div(id='click_output')
])

@app.callback(
    Output('click_output', 'children'),
    Input('plotly_map', 'clickData'))
def display_click_data(clickData):
    if clickData is None:
        return 'Click on a region to see details'
    else:
        # Extract the index of the clicked region
        point_idx = clickData['points'][0]['pointIndex']
        # Use the index to find the corresponding name and birth rate
        region_name = gdf_choro.iloc[point_idx]['statnaam']
        birth_rate = gdf_choro.iloc[point_idx]['GeboorteRelatief_25']
        return f'You clicked on {region_name}. Birth rate: {birth_rate}'

if __name__ == '__main__':
    app.run_server(debug=True)
