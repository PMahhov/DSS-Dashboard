# Libraries
import dash
from dash import dcc, callback
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import geopandas as gpd
import cbsodata
import plotly.express as px

# Let Dash know this is the actual homepage
dash.register_page(__name__, path='/')

# Download the CBS 'Kerncijfers wijken en buurten 2021' and select 3 columns
data = pd.DataFrame(cbsodata.get_data('85039NED', select = ['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
data['Codering_3'] = data['Codering_3'].str.strip() # Remove any leading characters
geodata_url = 'https://cartomap.github.io/nl/wgs84/gemeente_2021.geojson' # Download geojson file with all Dutch municipalities
municipal_boundaries = gpd.read_file(geodata_url)
municipal_boundaries = pd.merge(municipal_boundaries, data,
                                left_on="statcode",
                                right_on="Codering_3")
municipal_boundaries = municipal_boundaries.to_crs(epsg=4326)
gdf_choro = municipal_boundaries.copy()
gdf_choro['geoid'] = gdf_choro.index.astype(str)
gdf_choro = gdf_choro[['geoid', 'geometry', 'statnaam', 'GeboorteRelatief_25', 'statcode']]

fig = px.choropleth_mapbox(gdf_choro,
                           geojson=gdf_choro.__geo_interface__,
                           locations=gdf_choro.geoid,
                           color='GeboorteRelatief_25',
                           hover_name= 'statnaam',
                           hover_data = 'statcode',
                           featureidkey='properties.geoid',
                           center={'lat': 52.213, 'lon':5.2794},
                           mapbox_style='carto-positron',
                           zoom=6, height = 800)

# Define the app layout and display the graph. The layout also shows a row which displays the region on which is clicked.
layout = html.Div([
    dcc.Graph(id='plotly_map', figure=fig),
    dbc.Row(dbc.Col(id='click_output')),
])

@callback(
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