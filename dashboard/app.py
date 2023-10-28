import pandas as pd
import plotly.express as px
from flask import Flask, render_template_string, render_template
from sqlalchemy import create_engine, text, inspect, Table
import geopandas as gpd
import folium
import cbsodata

app = Flask(__name__)

@app.route('/')
def index():
    # Your original code starts here

    # Find out which columns are available
    metadata = pd.DataFrame(cbsodata.get_meta('83765NED', 'DataProperties'))

    # Download birth rates and delete spaces from regional identifiers
    data = pd.DataFrame(cbsodata.get_data('83765NED', select = ['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
    data['Codering_3'] = data['Codering_3'].str.strip()

    # Retrieve data with municipal boundaries from PDOK
    geodata_url = 'https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson'
    municipal_boundaries = gpd.read_file(geodata_url)

    # Link data from Statistics Netherlands to geodata
    municipal_boundaries = pd.merge(municipal_boundaries, data,
                                   left_on = "statcode",
                                   right_on = "Codering_3")

    # CRS: EPSG 3857 (web mercator projection wgs84)
    municipal_boundaries.crs
    municipal_boundaries = municipal_boundaries.to_crs(epsg = 4326)

    # First column: Geoid, geometry column and data columns
    gdf_choro = municipal_boundaries.copy()
    gdf_choro['geoid'] = gdf_choro.index.astype(str)
    gdf_choro = gdf_choro[['geoid', 'geometry', 'statnaam', 'GeboorteRelatief_25']]

    # Center
    nld_lat = 52.2130
    nld_lon = 5.2794
    nld_coordinates = (nld_lat, nld_lon)

    # Folium base map
    map_nld = folium.Map(location=nld_coordinates, tiles='cartodbpositron', zoom_start=6, control_scale=True)

    # Folium choropleth
    folium.Choropleth(geo_data=gdf_choro,
                      data=gdf_choro,
                      columns=['geoid', 'GeboorteRelatief_25'],
                      key_on='feature.id',
                      fill_color='Blues',
                      legend_name='Geboorterelatief').add_to(map_nld)

    # Your original code ends here

    map_nld.save("templates/map.html")
    return render_template("map.html")

if __name__ == "__main__":
    app.run(debug=True)