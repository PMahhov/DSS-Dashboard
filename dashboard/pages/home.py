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
    dbc.Card([
        dbc.CardBody(
            [
                html.H5("More information", id="click_output"),
                html.P("More statiscs are available for municipalities. Watch now!"),
                dbc.Button("View more", color="primary", disabled=True, id="redirect-button"),
            ]
        )
    ]),
    html.H2("Data information", style={'padding': '50px'}),
    html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    "aaaaaa", title="What are the scores I see on the screen?"
                ),
                dbc.AccordionItem(
                    "bbbbb", title="What are the spots on the map without a colour?"
                ),
                dbc.AccordionItem(
                    "cccc", title="Huh, where is the red?"
                ),
                dbc.AccordionItem(
                    "cccc", title="Data sources"
                ),
            ],
            flush=True,
        ),
        ),
], style={'paddingBottom': '50px'})

# The callback works on changes in the year and retrieves the required information
@callback(
    Output("graph", "figure"),
    Output("shared-data", "data"),
    Input("year", "value"))
def display_map(year):
    data = pd.read_sql_query(f'SELECT municipality_id, ROUND("XP"::numeric*10,2) AS XP, crime_score, ROUND(weighted_personal::numeric*10,2) AS personal, ROUND(weighted_property::numeric*10,2) AS property, ROUND(weighted_societal::numeric*100,2) AS societal FROM CRIME_SCORE WHERE year = {year}', engine)
    data['municipality_id'] = data['municipality_id'].str.strip() # Remove any leading characters
    geodata_url = f'https://cartomap.github.io/nl/wgs84/gemeente_2022.geojson' # Download geojson file with all Dutch municipalities
    municipal_boundaries = gpd.read_file(geodata_url)
    municipal_boundaries = pd.merge(municipal_boundaries, data,
                                    left_on="statcode",
                                    right_on="municipality_id")
    municipal_boundaries = municipal_boundaries.to_crs(epsg=4326) # import the map with the correct settings
    gdf_choro = municipal_boundaries.copy()
    gdf_choro['geoid'] = gdf_choro.index.astype(str)
    gdf_choro = gdf_choro[['geoid', 'geometry', 'statnaam', 'xp', 'statcode', 'crime_score', 'property', 'societal', 'personal']]

    fig = px.choropleth_mapbox(gdf_choro, # create a Mapbox-style choropleth based on the scale of xp. The colours are mapped in a range of 0-10
                            geojson=gdf_choro.__geo_interface__,
                            locations=gdf_choro.geoid,
                            color_continuous_scale='reds',
                            color='xp',
                            hover_name= 'statnaam',
                            hover_data = ['crime_score', 'property', 'societal', 'personal'],
                            featureidkey='properties.geoid',
                            center={'lat': 52.213, 'lon':5.2794},
                            mapbox_style='carto-positron',
                            range_color=(0,10),
                            labels={'xp':'Normalised crime score', 'crime_score':'Criminality level', 'property': 'Property crime score', 'societal': 'Societal crime score', 'personal': 'Personal crime score', 'geoid': 'Municipality number'},
                            zoom=6, height = 800,
                            title=f"Crimes in The Netherlands for {year}")
    return fig, gdf_choro.to_json()

@callback(
    [Output('click_output', 'children'),
     Output('redirect-button', 'disabled'),
     Output('redirect-button', 'href')],
    [Input('shared-data', 'data'),
     Input('graph', 'clickData')]
)
def display_click_data(shared_data, clickData):
    if clickData is None:
        return 'Click on a municipality to see more details', True, '#'
    else:
        gdf_choro = gpd.GeoDataFrame.from_features(json.loads(shared_data))
        point_idx = clickData['points'][0]['pointIndex']
        region_name = gdf_choro.iloc[point_idx]['statnaam']
        municipality_id = gdf_choro.iloc[point_idx]['statcode']
        
        # Assuming you have a page for each municipality with the format '/municipality/{municipality_id}'
        page_link = f"/municipality/{municipality_id}"
        
        return f'You clicked on {region_name}.', False, page_link