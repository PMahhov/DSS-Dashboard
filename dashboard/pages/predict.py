import dash
from dash import dcc, callback, dash_table, html, page_registry
from dash.dash_table import FormatTemplate
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
from dash import html
import pandas as pd
from decimal import Decimal
from sqlalchemy import create_engine
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")
clean_demo_data = pd.read_sql_query("select municipality_id, population, household_size, population_density, degree_of_urbanity, distancegp, distance_supermarket, distance_daycare, distance_school, avg_income_per_recipient, year, unemployment_rate from demo_data WHERE year >= 2013 AND year <= 2022", engine)    
crime_score_df = pd.read_sql_query("select * from crime_score WHERE year >= 2013 AND year <= 2022", engine)    
X_full_oversampled = pd.read_sql_query("select * from x_oversampled", engine)    
y_full_oversampled = pd.read_sql_query("select * from y_oversampled ", engine)    
coefficients_df = pd.read_sql_query("select * from coefficients_df ", engine)    


pop_df = clean_demo_data
prediction_demo_df = pop_df

# Create classifier
best_params_custom = {
    'n_estimators': 300,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 10,
    'class_weight': 'balanced_subsample'
}

feature_names = list(pop_df)[1:-2]
feature_names.append(list(pop_df)[-1])

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
    value = clean_demo_data.loc[clean_demo_data['municipality_id'] == GM_id].loc[clean_demo_data['year'] == year][stat]
    value = float(value.iloc[0])

  except: # if no existing value is found, extrapolate from linear regression coefficients table
    slope = coefficients_df[stat + "_slope"][coefficients_df['municipality_id'] == GM_id]
    intercept = coefficients_df[stat + "_intercept"][coefficients_df['municipality_id'] == GM_id]
    value = year * slope + intercept
    value = float(value.iloc[0])

    # if the value goes to zero due to a downwards trend, use the last non-zero prediction
    while value <= 0:
      year -= 1
      value = year * slope + intercept
      value = float(value.iloc[0])

  return value

def get_all_default_predictions(GM_id, year):
  values = []
  for feature in feature_names:
    value = get_demographic_value(GM_id, feature, year)
    # values[feature] = value
    if feature == 'population':
       round_value = round(value, 0)
    elif feature == 'unemployment_rate':
       round_value = round(value, 5)
    else:
       round_value = round(value, 2)
    values.append(round_value)
  # print(values)
  values_df = pd.DataFrame(columns = feature_names)
  values_df.loc[0] = values
  values_df['year'] = year
  return values_df


def predict_crime_class(GM_id, year, classifier = final_rf_classifier):
    values_df = get_all_default_predictions(GM_id, year)
    values_df.drop('year', axis=1, inplace=True)
    return str(final_rf_classifier.predict(values_df)[0])

def predict_crime_class_dfprovided(GM_id, year, values_df, classifier = final_rf_classifier):
    return str(final_rf_classifier.predict(values_df)[0])

def add_crime_class_predictions(GM_id, df):
    # Create a new column 'crime_class' to store the predictions
    df['crime_class'] = ''

    # Iterate over rows and call predict_crime_class for each year
    for index, row in df.iterrows():
        year = row['year']
        features = row.drop(['year', 'crime_class']).values  # Exclude 'year' and 'crime_class' columns
        # For years prior =< 2022, we already have an existing crime score
        if year > 2022:
            predicted_class = predict_crime_class(GM_id, year, features)
            df.at[index, 'crime_class'] = predicted_class
            df.at[index, 'source'] = 'Prediction'
        else:
            data = pd.read_sql_query(f"SELECT crime_score FROM crime_score WHERE municipality_id = '{GM_id}' AND year = {year}", engine)    
            actual_class = data.iloc[0]
            df.at[index, 'crime_class'] = actual_class['crime_score']
            df.at[index, 'source'] = 'CBS'

    return df



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
            dcc.Dropdown(
                id='year-dropdown',
                options=[
                    {'label': str(year), 'value': year} for year in range(2023, 2050)
                ],
                value=2021,
                style={'width': '50%'}
            ),
            html.Div([
               
                    dbc.Label("Population"),
                    dbc.Input(id='population', type='number', min=0),

                    dbc.Label("Household Size"),
                    dbc.Input(id='household-size', type='number', min=0),

                    dbc.Label("Population Density"),
                    dbc.Input(id='population-density', type='number', min=0, max=10000),

                    dbc.Label("Average Income per Recipient"),
                    dbc.Input(id='avg-income-per-recipient', type='number', min=0),

                    dbc.Label("Unemployment Rate in decimals (0-1)"),
                    dbc.Input(id='unemployment-rate', type='number', min=0, max=1),

                    dbc.Label("Degree of Urbanity (range 1-5)"),
                    dbc.Input(id='degree_of_urbanity', type='number', min=0, max=5),

                    dbc.Label("Distance to GP (km)"),
                    dbc.Input(id='distancegp', type='number', min=0),

                    dbc.Label("Distance to Daycare (km)"),
                    dbc.Input(id='distance-daycare', type='number', min=0),

                    dbc.Label("Distance to School (km)"),
                    dbc.Input(id='distance-school', type='number', min=0),

                    dbc.Label("Distance to Supermarket (km)"),
                    dbc.Input(id='distance-supermarket', type='number', min=0),

                dbc.Button("Calculate", id='calculate-btn', color="primary", className="mt-3"),
                html.Div(id='output-prediction', className="mt-3")
            ]),
            html.Div(f"Municipality code: {stat_code}"),
            html.Div([
                dcc.Loading(
                    id="loading-table-2",
                    type="circle",
                    children=[html.Div(id='tbl_out-2'), html.Div(id="data-table-2", style={'paddingBottom': '50px'})],
                )
            ]),  
        ], style={'paddingTop': '50px'})
    else:
        return html.Div([
            dcc.Location(id='url', refresh=False),
            html.H1(f'View municipal statistics'),
            html.Div(f"Municipality code: {stat_code}"),
            dbc.Alert("The requested municipality does not exist", color="danger"),
            ])        

@callback(
    Output('data-table-2', 'children'),
     Input('url', 'pathname')
)
def update_data(pathname):
    stat_code = pathname.split('/')[-1]
    data = pd.read_sql_query(f"SELECT demo_data.year AS year, demo_data.population AS population, household_size, low_educated_population, medium_educated_population, high_educated_population, population_density, avg_income_per_recipient, unemployment_rate, crime_score FROM demo_data, crime_score WHERE demo_data.municipality_id = '{stat_code}' AND demo_data.municipality_id=crime_score.municipality_id AND demo_data.year=crime_score.year", engine)    
    
    # In region.py, we can simply obtain the data over the last few years. This is not possible in prediction, so we'll predict it
    initial_df = get_all_default_predictions(stat_code, 2013)
    all_predictions_df = pd.DataFrame(columns=initial_df.columns)
    # Initialize an empty list to store all predictions
    all_predictions = []
    for year in range(2014, 2051):
        predictions_for_year = get_all_default_predictions(stat_code, year)
        all_predictions.append(predictions_for_year)
    
    if len(all_predictions) > 0:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    all_predictions_df_crimes = add_crime_class_predictions(stat_code, all_predictions_df)
    table = generate_table(all_predictions_df_crimes)

    return table

def generate_table(dataframe, max_rows=50):
    column_labels = {
        'year': 'Year',
        'population': 'Population',
        'household_size': 'Household Size',
        'population_density': 'Population Density',
        'avg_income_per_recipient': 'Average Income per Recipient',
        'unemployment_rate': 'Unemployment Rate (%)',
        'crime_class': 'Crime score',
        'source': 'Data source'
    }    

    column_hints = {
        'population': 'The number of people officially registered',
        'household_size': 'The average number of people per household',
        'population_density': 'The average number of people per square kilometer',
        'avg_income_per_recipient': 'The arithmetic average personal income per person based on persons with personal income',
        'unemployment_rate': 'The unemployment rate based on the percentage of people with an unemployment benefits  (%)',
        'crime_class': 'The crime score is based on a weighted average of the number of crimes per inhabitant, combined with the severity of the crime. A crime with a long prison sentence will impact the score more compared to a sentence of several months.'
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
        style_header_conditional=[{
        'if': {'column_id': col},
        'textDecoration': 'underline',
        'textDecorationStyle': 'dotted',
    } for col in column_hints]

    )

# Define callback to update input fields based on the selected year
@callback(
    [Output('population', 'value'),
     Output('household-size', 'value'),
     Output('population-density', 'value'),
     Output('avg-income-per-recipient', 'value'),
     Output('unemployment-rate', 'value'),
     Output('degree_of_urbanity', 'value'),
     Output('distancegp', 'value'),
     Output('distance-daycare', 'value'),
     Output('distance-school', 'value'),
     Output('distance-supermarket', 'value')],
    [Input('year-dropdown', 'value'), Input('url', 'pathname')]
)
def update_inputs(selected_year, url):
    stat_code = url.split('/')[-1]
    # Filter the DataFrame based on the selected year

    selected_data = get_all_default_predictions(stat_code, selected_year)

    # Return the values for the input fields
    return (
        selected_data['population'].iloc[0],
        selected_data['household_size'].iloc[0],
        selected_data['population_density'].iloc[0],
        selected_data['avg_income_per_recipient'].iloc[0],
        selected_data['unemployment_rate'].iloc[0],
        selected_data['degree_of_urbanity'].iloc[0],
        selected_data['distancegp'].iloc[0],
        selected_data['distance_daycare'].iloc[0],
        selected_data['distance_school'].iloc[0],
        selected_data['distance_supermarket'].iloc[0],
    )


# Define callback to update prediction based on user input
@callback(
    Output('output-prediction', 'children'),
    [Input('calculate-btn', 'n_clicks')], Input('year-dropdown', 'value'), Input('url', 'pathname'),
    [State('population', 'value'),
     State('household-size', 'value'),
     State('population-density', 'value'),
     State('avg-income-per-recipient', 'value'),
     State('unemployment-rate', 'value'),
     State('degree_of_urbanity', 'value'),
     State('distancegp', 'value'),
     State('distance-daycare', 'value'),
     State('distance-school', 'value'),
     State('distance-supermarket', 'value')]
)
def update_prediction(n_clicks, year, url, population, household_size, population_density, avg_income_per_recipient,
                      unemployment_rate, degree_of_urbanity, distancegp, distance_daycare, distance_school,
                      distance_supermarket):
    
    if n_clicks != None and n_clicks > 0:
        fields = {
            'population': population,
            'household_size': household_size,
            'population_density': population_density,
            'avg_income_per_recipient': avg_income_per_recipient,
            'unemployment_rate': unemployment_rate,
            'degree_of_urbanity': degree_of_urbanity,
            'distancegp': distancegp,
            'distance_daycare': distance_daycare,
            'distance_school': distance_school,
            'distance_supermarket': distance_supermarket
        }

        invalid_fields = [field for field, value in fields.items() if value is None or pd.isna(value)]

        if invalid_fields:
            return f"Please provide valid values for the following fields: {', '.join(invalid_fields)}"
        else:

            # Perform your prediction logic here based on the input values
            stat_code = url.split('/')[-1]
            user_data = pd.DataFrame({
                'population': [population],
                'household_size': [household_size],
                'population_density': [population_density],
                'degree_of_urbanity': [degree_of_urbanity],
                'distancegp': [distancegp],
                'distance_supermarket': [distance_supermarket],
                'distance_daycare': [distance_daycare],
                'distance_school': [distance_school],
                'avg_income_per_recipient': [avg_income_per_recipient],
                'unemployment_rate': [unemployment_rate],
            })

            return "The values provided result in: "+predict_crime_class_dfprovided(stat_code, year, user_data)