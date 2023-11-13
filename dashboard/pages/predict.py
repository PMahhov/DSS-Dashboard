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
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")
clean_demo_data = pd.read_sql_query("select * from demo_data WHERE year >= 2013 AND year <= 2022", engine)    


pop_df = clean_demo_data

groups = pop_df.groupby('municipality_id')


# Initialize DataFrame to store the coefficients

coefs_columns = ['municipality_id']


groups = pop_df.groupby('municipality_id')

# print(coefficients_df)

first_loop = True

feature_names = list(pop_df)[1:-2]
feature_names.append(list(pop_df)[-1])

for municipality_id, group_data in groups:
  # print(municipality_id,"calculating")

  row_coefs = [municipality_id]

  # print("list pop df:",list(pop_df)[1:])
  for dem_item in feature_names:
    # print(municipality_id, dem_item)

    X = group_data['year'].values.reshape(-1,1)
    y = group_data[dem_item].values

    model = LinearRegression()

    # print("X:",X)
    # print("y:",y)

    while pd.isnull(y[0]):
      # print("before beginning nan cull:",X,y)
      y = y[1:]
      if len(X) > len(y):
        X = X[1:]
      # print("after nan cull:",X,y)
      if len(y) == 0 or len(X)==0:
        raise ValueError("Empty set after beginning nan cull")

    while pd.isnull(y[-1]):
      # print("before end nan cull:",X,y)
      y = y[:-1]
      if len(X) != len(y):
        X = X[:-1]
      # print("after end nan cull:",X,y)
      if len(y) == 0 or len(X)==0:
        raise ValueError("Empty set after end nan cull")

    for i in y:
      if pd.isnull(i):
        raise ValueError("There are remaining nans in data")

    # print("X,y before fit",X,y)
    model.fit(X,y)

    current_intercept = model.intercept_
    current_slope = model.coef_[0]


    # if first_loop or dem_item+"_intercept" not in coefs_columns:
    if first_loop:
      coefs_columns.append(dem_item + "_intercept")
      coefs_columns.append(dem_item + "_slope")

    row_coefs.append(current_intercept)
    row_coefs.append(current_slope)



  if first_loop:
    coefficients_df = pd.DataFrame(columns= coefs_columns)
    first_loop = False

  # print(len(coefs_columns),"coefs_columns:",coefs_columns)
  # print(len(row_coefs),"row_coefs",row_coefs)
  coefficients_df.loc[len(coefficients_df)] = row_coefs


print(coefficients_df.head(6))

# Create classifier
smote = SMOTE(sampling_strategy='auto', random_state=427)
X_full_oversampled, y_full_oversampled = smote.fit_resample(X, y)
best_params_custom = {
    'n_estimators': 300,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 10,
    'class_weight': 'balanced_subsample'
}

# from sklearn.metrics._plot.precision_recall_curve import average_precision_score
final_rf_classifier = RandomForestClassifier(random_state=427, n_estimators=best_params_custom['n_estimators'],
                                       max_depth=best_params_custom['max_depth'],
                                       min_samples_split = best_params_custom['min_samples_split'],
                                       min_samples_leaf = best_params_custom['min_samples_leaf'],
                                      #  class_weight = best_params_custom['class_weight'],
                                      # class_weight = None,
                                       max_features = best_params_custom['max_features'])
final_rf_classifier.fit(X_full_oversampled, y_full_oversampled)

y_pred_full_oversampled = final_rf_classifier.predict(X_full_oversampled)
y_true_full_oversampled = np.array(y_full_oversampled)


# Use the predictor
def get_demographic_value(GM_id, stat, year):
  try:    #check for existing value first
    value = float(clean_demo_data.loc[clean_demo_data['municipality_id'] == GM_id].loc[clean_demo_data['year'] == year][stat])

  except: # if no existing value is found, extrapolate from linear regression coefficients table
    slope = coefficients_df[stat + "_slope"][coefficients_df['municipality_id'] == GM_id]
    intercept = coefficients_df[stat + "_intercept"][coefficients_df['municipality_id'] == GM_id]
    value = float(year * slope + intercept)

    # if the value goes to zero due to a downwards trend, use the last non-zero prediction
    while value <= 0:
      year -= 1
      value = float(year * slope + intercept)

  return value


print("unemployment rate in GM0014:")
print("2010:",get_demographic_value("GM0014","unemployment_rate",2010))
print("2015:",get_demographic_value("GM0014","unemployment_rate",2015))
print("2020:",get_demographic_value("GM0014","unemployment_rate",2020))
print("2025:",get_demographic_value("GM0014","unemployment_rate",2025))
print("2030:",get_demographic_value("GM0014","unemployment_rate",2030))
print("2035",get_demographic_value("GM0014","unemployment_rate",2035))
print("2040:",get_demographic_value("GM0014","unemployment_rate",2040))
print("2045:",get_demographic_value("GM0014","unemployment_rate",2045))
print("2045:",get_demographic_value("GM0014","unemployment_rate",2050), '\n')


def get_all_default_predictions(GM_id, year):
  values = []
  for feature in feature_names:
    value = get_demographic_value(GM_id, feature, year)
    # values[feature] = value
    values.append(value)
  # print(values)
  values_df = pd.DataFrame(columns = feature_names)
  values_df.loc[0] = values
  return values_df


def predict_crime_class(GM_id, year, classifier = final_rf_classifier):
  values_df = get_all_default_predictions(GM_id, year)
  return str(classifier.predict(values_df)[0])

print("all predictions in GM0014 in 2033")
print(get_all_default_predictions("GM0014", 2033), '\n')

print("crime category prediction in GM0014 in 2033 is:", predict_crime_class("GM0014", 2033))


## +++++++++++++++++
## ACTUAL PAGE
## ++++++++++++++++
dash.register_page(__name__, path_template="/predict/<stat_code>")

def layout(stat_code=None):
    municipal_name = pd.read_sql_query(f"SELECT municipality_name FROM municipality_names WHERE municipality_id = '{stat_code}' LIMIT 1", engine)
    if not (municipal_name.empty or pd.isna(municipal_name.iloc[0]['municipality_name'])):
        municipal_name = municipal_name.iloc[0]['municipality_name']
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'Predict future municipal statistics - {municipal_name}'),
            html.Div(f"Municipality code: {stat_code}"),
            dcc.Dropdown(
                id='year-dropdown-2',
                options=[
                    {'label': str(year), 'value': year} for year in range(2013, 2022)
                ],
                value=2021,
                style={'width': '50%'}
            ),
            html.Div([
                html.Div([
                    dcc.Graph(id='pie-chart-2'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(id='crime-scatter-2')
                ], style={'width': '48%', 'display': 'inline-block'}),
            ]),
            html.Div([
                dcc.Loading(
                    id="loading-table",
                    type="circle",
                    children=[html.Div(id="data-table-2", style={'paddingBottom': '50px'}),
                                dbc.Alert(id='tbl_out-2', color='secondary')],
                )
            ])        
        ], style={'paddingTop': '50px'})
    else:
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics'),
            html.Div(f"Municipality code: {stat_code}"),
            dbc.Alert("The requested municipality does not exist", color="danger"),
            ])        

@callback(
    [Output('data-table-2', 'children'),
     Output('pie-chart-2', 'figure'),
     Output('crime-scatter-2', 'figure')],
    [Input('year-dropdown-2', 'value'),
     Input('url', 'pathname')]
)
def update_data(current_year, pathname):
    stat_code = pathname.split('/')[-1]
    data = pd.read_sql_query(f"SELECT demo_data.year AS year, demo_data.population AS population, household_size, low_educated_population, medium_educated_population, high_educated_population, population_density, avg_income_per_recipient, unemployment_rate, crime_score FROM demo_data, crime_score WHERE demo_data.municipality_id = '{stat_code}' AND demo_data.municipality_id=crime_score.municipality_id AND demo_data.year=crime_score.year", engine)    
    table = generate_table(data)

    # Create a pie chart
    fig = generate_pie_chart(current_year, data)

    # Create the crime scatter plot
    crimescatter = generate_crime_scatter(stat_code, current_year)
    return table, fig, crimescatter

def generate_table(dataframe, max_rows=15):
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

    # Round the 'avg_income_per_recipient' column to 0 decimal places
    dataframe['avg_income_per_recipient'] = dataframe['avg_income_per_recipient'].round(0)
    dataframe['unemployment_rate'] = (dataframe['unemployment_rate'] * 100).round(2)
    dataframe = dataframe.replace('', 'Not known yet')

    rows = dataframe.to_dict('records')
    
    return dash_table.DataTable(
        id='data-table-2',
        columns=columns,
        data=rows,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=max_rows,
        sort_action='native',  
        sort_mode='single', 
        sort_by=[{'column_id': 'year', 'direction': 'desc'}],
        tooltip_header={col: f'Explanation: {column_hints[col]}' for col in column_hints},
        tooltip_data=[{
        'crime_score': {
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
    if not (pie_df.empty or pd.isna(pie_df.iloc[0]['low_educated_population'])):
        pie_dfs = pie_df.iloc[0]

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
                names = ['Low Educated', 'Medium Educated', 'High Educated'],
                title=f'Distribution of education levels in {selected_year}')
    else:
        fig = px.scatter(x=[0], y=[0], text=["No data available"])
        # Update layout for better appearance (optional)
        fig.update_layout(
            width=400,
            height=300,
            title="Educational Distribution",
            template="plotly_white"  # You can choose different templates
        )

        # Hide the axis to make it cleaner
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False) 
        fig.update_traces(marker=dict(size=0))
    return fig

def generate_crime_scatter(statcode, selected_year:int):
    #SELECT crime_data.crime_code AS crime_code, registered_crimes, max_jailtime_yrs FROM crime_data, crime_type WHERE crime_data.crime_code = crime_type.crime_code AND year = '2022' AND municipality_id = 'GM0197'
    data = pd.read_sql_query(f"SELECT crime_data.crime_code AS crime_code, registered_crimes, max_jailtime_yrs, category FROM crime_data, crime_type WHERE crime_data.crime_code = crime_type.crime_code AND year = '{selected_year}' AND municipality_id = '{statcode}'", engine)
    # The complete table provided earlier
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

@callback(Output('tbl_out-2', 'children'), [Input('data-table-2', 'active_cell'), Input('url', 'pathname')])
def get_graph_over_time(active_cell, pathname):
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