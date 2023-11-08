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
from sqlalchemy import create_engine, text
import json


# Let Dash know this is the actual homepage
dash.register_page(__name__, path='/')
engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")


# Get relevant data from the database for starting year 2022. Enable a dropdown for different years

# Define the app layout and display the graph. The layout also shows a row which displays the region on which is clicked.
layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Slider(2012, 2023,
    step=None,
    marks={
        2013: '2013',
        2014: '2014',
        2015: '2015',
        2016: '2016',
        2017: '2017',
        2018: '2018',
        2019: '2019',
        2020: '2020',
        2021: '2021',
        2022: '2022'
    },
    value=2022,
    id="year"
    ),
    dcc.Store(id='shared-data'),
    dbc.Row(dbc.Col(id='click_output'))
])

@callback(
    Output("graph", "figure"),
    Output("shared-data", "data"),
    Input("year", "value"))
def display_map(year):
    data = pd.read_sql_query(f'SELECT municipality_id, "XP"*100 AS XP, crime_score, weighted_personal*100 AS personal, weighted_property*100 AS property, weighted_societal*100 AS societal FROM CRIME_SCORE WHERE year = {year}', engine)
    data['municipality_id'] = data['municipality_id'].str.strip() # Remove any leading characters
    geodata_url = f'https://cartomap.github.io/nl/wgs84/gemeente_{year}.geojson' # Download geojson file with all Dutch municipalities
    print(geodata_url)
    municipal_boundaries = gpd.read_file(geodata_url)
    municipal_boundaries = pd.merge(municipal_boundaries, data,
                                    left_on="statcode",
                                    right_on="municipality_id")
    municipal_boundaries = municipal_boundaries.to_crs(epsg=4326)
    gdf_choro = municipal_boundaries.copy()
    gdf_choro['geoid'] = gdf_choro.index.astype(str)
    gdf_choro = gdf_choro[['geoid', 'geometry', 'statnaam', 'xp', 'statcode', 'crime_score', 'property', 'societal', 'personal']]

    fig = px.choropleth_mapbox(gdf_choro,
                            geojson=gdf_choro.__geo_interface__,
                            locations=gdf_choro.geoid,
                            color_continuous_scale='oranges',
                            color='xp',
                            hover_name= 'statnaam',
                            hover_data = ['crime_score', 'property', 'societal', 'personal'],
                            featureidkey='properties.geoid',
                            center={'lat': 52.213, 'lon':5.2794},
                            mapbox_style='carto-positron',
                            range_color=(0,10),
                            labels={'xp':'Normalised crime score', 'crime_score':'Criminality level', 'property': 'Property crime score', 'societal': 'Societal crime score', 'personal': 'Personal crime score', 'geoid': 'Municipality number'},
                            zoom=6, height = 800)
    return fig, gdf_choro.to_json()

@callback(
    Output('click_output', 'children'),
    Input('shared-data', 'data'),
    Input('graph', 'clickData'))
def display_click_data(shared_data, clickData):
    if clickData is None:
        return 'Click on a region to see details'
    else:
        gdf_choro = gpd.GeoDataFrame.from_features(json.loads(shared_data))
        print(gdf_choro)
        # Extract the index of the clicked region
        point_idx = clickData['points'][0]['pointIndex']
        # Use the index to find the corresponding name and birth rate
        region_name = gdf_choro.iloc[point_idx]['statnaam']
        return f'You clicked on {region_name}.'