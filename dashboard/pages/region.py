import dash
from dash import dcc, callback, dash_table, html, page_registry
from dash.dash_table import FormatTemplate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from dash import html
import pandas as pd
from decimal import Decimal
from sqlalchemy import create_engine

dash.register_page(__name__, path_template="/municipality/<stat_code>")
engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")

#This is the page with the details on the municipality. 

def layout(stat_code=None):
    # Obtain the municipality name for display at the top
    municipal_name = pd.read_sql_query(f"SELECT municipality_name FROM municipality_names WHERE municipality_id = '{stat_code}' LIMIT 1", engine)
    try:
        municipal_name_defined = municipal_name.iloc[0]['municipality_name']
    except IndexError:
        # If the municipal_name cannot be found, the ID is incorrect. We create a bogus municipal_name_defined that is empty for later on
        municipal_name_defined = pd.DataFrame({'A' : []})
    # Create the first tab content
    tab1_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics - {municipal_name_defined}'),
            html.Div(f"Municipality code: {stat_code}"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[
                    {'label': str(year), 'value': year} for year in range(2013, 2022)
                ],
                value=2021,
                style={'width': '50%'}
            ),
            html.Div([
                html.Div([
                    dcc.Graph(id='pie-chart'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='crime-scatter')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
                html.Div([
                    dcc.Loading(
                        id="loading-table",
                        type="circle",
                        children=[html.Div(id="data-table", style={'paddingBottom': '50px'}),
                                  dbc.Alert(id='tbl_out', color='secondary')],
                    )
                ]),
        ],
    ),
    className="mt-3",
    )
    tab2_content = dbc.Card(
    dbc.CardBody(
        [
            None
        ],
    ),
    className="mt-3",
    )
    # If the municipal name string is empty, display an error code
    if not (municipal_name.empty or pd.isna(municipal_name.iloc[0]['municipality_name'])):
        municipal_name = municipal_name.iloc[0]['municipality_name']
        return html.Div([
        dbc.Tabs(
        [
        dbc.Tab(tab1_content, label="History"),
        dbc.Tab(tab2_content, label="Compare"),
        ]
        )], style={'paddingTop': '50px'})
    else:
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics'),
            html.Div(f"Municipality code: {stat_code}"),
            dbc.Alert("The requested municipality does not exist (anymore)", color="danger"),
            ])        

@callback(
    [Output('data-table', 'children'),
     Output('pie-chart', 'figure'),
     Output('crime-scatter', 'figure')],
    [Input('year-dropdown', 'value'),
     Input('url', 'pathname')]
)
def update_data(current_year, pathname):
    # Create a dataframe called data with the required columns for the table and pie chart.
    stat_code = pathname.split('/')[-1] # Obtain the municipality ID from the URL
    data = pd.read_sql_query(f"SELECT demo_data.year AS year, demo_data.population AS population, household_size, low_educated_population, medium_educated_population, high_educated_population, population_density, avg_income_per_recipient, unemployment_rate, crime_score FROM demo_data, crime_score WHERE demo_data.municipality_id = '{stat_code}' AND demo_data.municipality_id=crime_score.municipality_id AND demo_data.year=crime_score.year", engine)    
    table = generate_table(data)

    # Create a pie chart
    fig = generate_pie_chart(current_year, data)

    # Create the crime scatter plot
    crimescatter = generate_crime_scatter(stat_code, current_year)
    return table, fig, crimescatter

def generate_table(dataframe, max_rows=15):
    # Add some human readable labels and explanation
    column_labels = {
        'year': 'Year',
        'population': 'Population',
        'household_size': 'Household Size',
        'population_density': 'Population Density',
        'avg_income_per_recipient': 'Average Income per Recipient',
        'unemployment_rate': 'Unemployment Rate (%)',
        'crime_score': 'Crime score',
    }    

    column_hints = {
        'population': 'The number of people officially registered',
        'household_size': 'The average number of people per household',
        'population_density': 'The average number of people per square kilometer',
        'avg_income_per_recipient': 'The arithmetic average personal income per person based on persons with personal income',
        'unemployment_rate': 'The unemployment rate based on the percentage of people with an unemployment benefits  (%)',
        'crime_score': 'The crime score is based on a weighted average of the number of crimes per inhabitant, combined with the severity of the crime. A crime with a long prison sentence will impact the score more compared to a sentence of several months.'
    }    
    # Create DataTable columns
    columns = [{'name': column_labels[col], 'id': col} for col in column_labels]

    # Round the 'avg_income_per_recipient' column to 0 decimal places and convert the 'unemployment_rate' to a percentage rounded to 2 decimals
    dataframe['avg_income_per_recipient'] = dataframe['avg_income_per_recipient'].round(0)
    dataframe['unemployment_rate'] = (dataframe['unemployment_rate'] * 100).round(2)
    dataframe = dataframe.replace('', 'Not known yet')

    rows = dataframe.to_dict('records') # The DataTable function requires a dictionary-based dataframe
    
    return dash_table.DataTable(
        id='data-table',
        columns=columns,
        data=rows,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=max_rows,
        sort_action='native',  
        sort_mode='single', 
        sort_by=[{'column_id': 'year', 'direction': 'desc'}], # sort the table by year in descending order
        tooltip_header={col: f'Explanation: {column_hints[col]}' for col in column_hints}, # when you hover over a column, this is what you see
        tooltip_data=[{
        'crime_score': { # when you hover over the crime_score label, you see a small tooltip with what it means
            'value': 'Compared to the rest of The Netherlands, this is doing **{} {}** than other municipalities'.format(
                '66%' if row['crime_score'] == 'low_crime' or row['crime_score'] == 'high_crime' else '33%',
                'better' if row['crime_score'] == 'low_crime' or row['crime_score'] == 'medium_crime' else 'worse'
                
            ),
            'type': 'markdown'
        }
    } for row in rows],

        style_header_conditional=[{
        'if': {'column_id': col},
        'textDecoration': 'underline',
        'textDecorationStyle': 'dotted',
    } for col in column_hints]

    )
    
def generate_pie_chart(selected_year:int, dataframe):
    # Function to create a pie chart using Plotly Express
    pie_df = dataframe[dataframe['year'] == selected_year]
    # check if there is education data for the selected year
    if not (pie_df.empty or pd.isna(pie_df.iloc[0]['low_educated_population'])):
        pie_dfs = pie_df.iloc[0]
        # convert the data to percentages
        data = {
            'low_educated_population': pie_dfs['low_educated_population']*100,
            'medium_educated_population': pie_dfs['medium_educated_population']*100,
            'high_educated_population': pie_dfs['high_educated_population']*100
        }
        data = {k:[v] for k,v in data.items()}
        df = pd.DataFrame(data)
        legend_names = ['Low Educated', 'Medium Educated', 'High Educated']


        fig = px.pie(df, 
                values=df.iloc[0], 
                names = legend_names, # Translate the columns to human readable names
                title=f'Distribution of education levels in {selected_year}')
    else:
        fig = px.scatter(x=[0], y=[0], text=["No data available"])
        # Update layout for better appearance
        fig.update_layout(
            width=400,
            height=300,
            title="Educational Distribution",
            template="plotly_white"
        )

        # Hide the axis to make it cleaner
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False) 
        fig.update_traces(marker=dict(size=0))
    return fig

def generate_crime_scatter(statcode, selected_year:int):
    # Obtain the crime scores
    data = pd.read_sql_query(f"SELECT crime_data.crime_code AS crime_code, registered_crimes, max_jailtime_yrs, category FROM crime_data, crime_type WHERE crime_data.crime_code = crime_type.crime_code AND year = '{selected_year}' AND municipality_id = '{statcode}'", engine)
    # List of crimes and their descriptions
    crime_table = {
        '1.1.1': 'Theft/Burglary Home',
        '1.1.2': 'Theft/Burglary Box/Garage/Shed',
        '1.2.1': 'Theft from/of Motor Vehicles',
        '1.2.2': 'Theft of Motor Vehicles',
        '1.2.3': 'Theft of Mopeds, Mustaches, and Bicycles',
        '1.2.4': 'Pickpocketing',
        '1.2.5': 'Theft from/of Public Transport Vehicles',
        '1.3.1': 'Accidents (Road)',
        '1.4.1': 'Sexual Offense (Rape, Public Indecency, Indecent Assault, Pornography)',
        '1.4.2': 'Murder, Manslaughter',
        '1.4.3': 'Public Violence (Person)',
        '1.4.4': 'Threatening',
        '1.4.5': 'Abuse',
        '1.4.6': 'Street Robbery',
        '1.4.7': 'Robbery',
        '1.5.2': 'Thefts (Water)',
        '1.6.1': 'Fire/Explosion',
        '1.6.2': 'Other Property Crimes',
        '1.6.3': 'Human Trafficking',
        '2.1.1': 'Drugs/Drinking Nuisance',
        '2.2.1': 'Destruction or Property Damage',
        '2.4.1': 'Neighborhood Rumor (Relationship Problems)',
        '2.4.2': 'Trespassing',
        '2.5.1': 'Theft/Burglary Companies, etc.',
        '2.5.2': 'Shoplifting',
        '2.6.1': 'Organization of the Environmental Management Act',
        '2.6.2': 'Soil',
        '2.6.3': 'Water',
        '2.6.4': 'Waste',
        '2.6.5': 'Building Materials',
        '2.6.7': 'Manure',
        '2.6.8': 'Transport of Hazardous Substances',
        '2.6.9': 'Fireworks',
        '2.6.10': 'Pesticides',
        '2.6.11': 'Nature and Landscape',
        '2.6.12': 'Spatial Planning',
        '2.6.13': 'Animals',
        '2.6.14': 'Food Safety',
        '2.7.2': 'Special Laws (Illegal Gambling, Telecommunication Law, Money Laundering)',
        '2.7.3': 'Livability (Other)',
        '3.1.1': 'Drug Trafficking',
        '3.1.2': 'Human Smuggling',
        '3.1.3': 'Arms Trade',
        '3.2.1': 'Child Pornography',
        '3.2.2': 'Child Prostitution',
        '3.3.2': 'Under the Influence (Air)',
        '3.3.5': 'Air (Other)',
        '3.4.2': 'Under the Influence (Water)',
        '3.5.2': 'Under the Influence (Road)',
        '3.5.5': 'Road (Other)',
        '3.6.4': 'Damage to Public Order',
        '3.7.1': 'Discrimination',
        '3.7.2': 'Immigration Care',
        '3.7.3': 'Societal Integrity',
        '3.7.4': 'Cybercrime',
        '3.9.1': 'Horizontal Fraud (Financial Crimes)',
        '3.9.2': 'Vertical Fraud',
        '3.9.3': 'Fraud (Other) (Using/Accepting Counterfeit Money or a Fake Police Report)'
    }

    # Map crime codes to titles and add a new 'title' column to the DataFrame
    data['title'] = data['crime_code'].map(crime_table)
    
    fig = px.scatter(data, x="max_jailtime_yrs",
                    y="registered_crimes", 
                    color="category", 
                    size="registered_crimes",  
                    hover_data={'title': True, 'category':False} ,
                    labels={'title':'Offence', 'registered_crimes':'Registered occurences', 'max_jailtime_yrs':'Maximum jailtime (years)', 
                              'category':'Category'}, 
                    title=f"Reported crime and maximum jail time in {selected_year}")
    return fig

@callback(Output('tbl_out', 'children'), [Input('data-table', 'active_cell'), Input('url', 'pathname')])
def get_graph_over_time(active_cell, pathname):
    # When you click on a table cell, you can see more statistics
    if active_cell:
        stat_code = pathname.split('/')[-1]
        column_name = active_cell['column_id']
        data = pd.read_sql_query(f"SELECT demo_data.year AS year, demo_data.population AS population, household_size, low_educated_population, medium_educated_population, high_educated_population, population_density, avg_income_per_recipient, unemployment_rate, ROUND(crime_score.\"XP\"::numeric*10,2) AS crime_score FROM demo_data, crime_score WHERE demo_data.municipality_id = '{stat_code}' AND demo_data.municipality_id=crime_score.municipality_id AND demo_data.year=crime_score.year", engine)    
        data['avg_income_per_recipient'] = data['avg_income_per_recipient'].round(0)
        data['unemployment_rate'] = (data['unemployment_rate'] * 100).round(2)

        
        fig = px.line(data, x="year", y=active_cell['column_id'], title=f'{column_name} year over year')
        return dcc.Graph(figure=fig, id='graph-over-time')
    else:
        return "Click a cell in the table the to see the progress of this variable over time"