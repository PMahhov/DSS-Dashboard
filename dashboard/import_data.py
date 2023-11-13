import requests
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text
json_url = "https://cartomap.github.io/nl/wgs84/gemeente_2023.geojson"
response = requests.get(json_url)
json_data = response.json()

cur_path = os.path.dirname(__file__)

# Extract municipality IDs from the JSON data
municipality_ids = [feature["properties"]["statcode"] for feature in json_data["features"]]
values_to_remove = ['GM1980', 'GM1982', 'GM1991', 'GM1992'] # these were merged later on

# Remove the values from the list
for value in values_to_remove:
    municipality_ids.remove(value)

# Read and import Excel files
def read_data(year, data_type):
    demo_file_path = f'kwb-{year}.xls'
    crime_file_path = f'policedata{year}.csv'

    if data_type == 'demo':
        try:
            demo_df = pd.read_excel(demo_file_path)
            demo_df['year'] = int(year)
        except FileNotFoundError:
            return None
        return demo_df

    if data_type == 'crime':
        try:
            crime_df = pd.read_csv(crime_file_path, delimiter=';')
        except FileNotFoundError:
            return None
        return crime_df

def merge_data(years):
    demo_dfs, crime_dfs = [], []

    for year in years:
        demo_df = read_data(year, 'demo')
        crime_df = read_data(year, 'crime')

        if demo_df is not None:
            demo_dfs.append(demo_df)
        if crime_df is not None:
            crime_dfs.append(crime_df)

    merged_demo_df = pd.concat(demo_dfs, ignore_index=True)
    merged_crime_df = pd.concat(crime_dfs, ignore_index=True)

    return merged_demo_df, merged_crime_df

# Demographic data pre processing
def clean_demo_data_func(demo_df, current_municipalities):
  demo_df = demo_df[['gwb_code', 'gm_naam', 'a_inw', 'g_hhgro', 'bev_dich',
                     'ste_mvs', 'a_opl_lg', 'a_opl_md', 'a_opl_hg', 'g_afs_hp', 'g_afs_gs',
                     'g_afs_kv', 'g_afs_sc', 'a_soz_wb','a_soz_ww', 'g_ink_po', 'year']]
  demo_df = demo_df[demo_df['gwb_code'].str.startswith('GM')]

  demo_df = demo_df[demo_df['gwb_code'].isin(current_municipalities)]



  demo_df = demo_df.rename(columns={
        'gwb_code': 'municipality_id',
        'gm_naam': 'municipality_name',
        'a_inw': 'population',
        'g_hhgro': 'household_size',
        'bev_dich': 'population_density',
        'ste_mvs': 'degree_of_urbanity',
        'a_opl_lg': 'low_educated_population',
        'a_opl_md': 'medium_educated_population',
        'a_opl_hg': 'high_educated_population',
        'g_afs_hp': 'distance_GP',
        'g_afs_gs': 'distance_supermarket',
        'g_afs_kv': 'distance_daycare',
        'g_afs_sc': 'distance_school',
        'g_ink_po': 'avg_income_per_recipient'
    })

  for col in demo_df.columns:
    demo_df[col] = demo_df[col].replace('.', np.nan)


  #convert commas to floating points in respective columns
  columns_to_convert = ['household_size', 'distance_GP', 'distance_supermarket', 'distance_daycare',
                        'distance_school', 'avg_income_per_recipient']


  for column in columns_to_convert:
    # print(demo_df[column].dtype)
    # print(demo_df[column])
    demo_df[column] = demo_df[column].str.replace(',', '.').astype(float)

    # print(demo_df[column].dtype)

  to_float_columns = ['low_educated_population',
                      'medium_educated_population', 'high_educated_population',
                      'avg_income_per_recipient', 'a_soz_wb', 'a_soz_ww']

  # demo_df['labor_force_participation'] = demo_df['labor_force_participation'].astype(float)
  for col in to_float_columns:
    demo_df[col] = demo_df[col].astype(float)
    demo_df[col] = pd.to_numeric(demo_df[col])

  demo_df['avg_income_per_recipient'] = demo_df['avg_income_per_recipient'] * 1000
  
  total_education_population = demo_df['low_educated_population'] + demo_df['medium_educated_population'] + demo_df['high_educated_population']

  for column in ['low_educated_population', 'medium_educated_population', 'high_educated_population']:
    demo_df[column] = demo_df[column] / total_education_population

  #unemployment rate calculation
  demo_df['unemployment_rate'] = (demo_df['a_soz_wb'] + demo_df['a_soz_ww']) / demo_df['population']

  municipality_names = demo_df[['municipality_id', 'municipality_name']]
  demo_df.drop(columns=['municipality_name','a_soz_wb', 'a_soz_ww'], inplace=True)


  return demo_df, municipality_names

# Criminal data preprocessing
def clean_crime_data_func(crime_df, current_municipalities):
  #keep desired columns only
  crime_df = crime_df[["SoortMisdrijf", "RegioS", "Perioden", "GeregistreerdeMisdrijven_1"]]


  #keep municipality data only
  crime_df = crime_df[crime_df["RegioS"].str.startswith('GM')]

  crime_df = crime_df[crime_df["RegioS"].isin(current_municipalities)]

  crime_df = crime_df.rename(columns={
        "SoortMisdrijf": 'crime_code',
        "RegioS": 'municipality_id',
        "Perioden": 'year',
        "GeregistreerdeMisdrijven_1": 'registered_crimes'
    })

  crime_df['crime_code'] = crime_df['crime_code'].str.strip()
  crime_codes_to_keep = ["1.6.2", "3.9.1", "1.3.1", "1.2.3", "2.2.1", "1.2.1", "1.4.5", "1.1.1",
                         "2.5.2", "3.5.2", "1.4.4", "2.5.1", "1.2.5", "3.1.1", "1.2.4", "1.1.2",
                         "3.7.3", "1.2.2", "1.4.1", "3.6.4", "3.5.5", "3.1.3", "1.6.1", "3.7.4",
                         "1.4.6", "1.4.3", "2.4.2", "1.4.2", "2.7.2", "3.9.3", "1.5.2", "1.4.7"]
  crime_df = crime_df[crime_df['crime_code'].isin(crime_codes_to_keep)]

  #drop rows with non-finite or missing values in 'registered_crimes'
  #change it with different threshold later
  # crime_df = crime_df[crime_df['registered_crimes'] >= 100]
  crime_df = crime_df.dropna(subset=['registered_crimes'], how='any')

  #convert 'registered_crimes' to integers
  crime_df['registered_crimes'] = crime_df['registered_crimes'].astype(int)

  #extract year
  crime_df['year'] = crime_df['year'].str.extract(r'(\d{4})')


  return crime_df

years = range(2013, 2024)
merged_demo_data, merged_crime_data = merge_data(years)

clean_demo_data, municipality_names_df = clean_demo_data_func(merged_demo_data, municipality_ids)

clean_crime_data = clean_crime_data_func(merged_crime_data, municipality_ids)

# Create crime_type df
def crime_type_df():
    # Define the crime code categories and max jail times
    categories = {
        'Personal crime': ["1.1.1", "1.1.2", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5", "1.4.1", "1.4.2", "1.4.3", "1.4.4", "1.4.5", "1.4.6", "1.4.7", "1.5.2", "1.6.1", "1.6.2"],
        'Property crime': ["2.2.1", "2.4.2", "2.5.1", "2.5.2", "2.7.2"],
        'Societal crime': ["3.1.1", "3.1.3", "3.5.2", "3.5.5", "3.6.4", "3.7.3", "3.7.4", "3.9.1", "3.9.3"]
    }

    max_jail_times = {
        "1.1.1": 12, "1.1.2": 4, "1.2.1": 4, "1.2.2": 4, "1.2.3": 4, "1.2.4": 4, "1.2.5": 4, "1.4.1": 6, "1.4.2": 30,
        "1.4.3": 4.5, "1.4.4": 3, "1.4.5": 7.5, "1.4.6": 9, "1.4.7": 12, "1.5.2": 4, "1.6.1": 12, "1.6.2": 4,
        "2.2.1": 2, "2.4.2": 1, "2.5.1": 4, "2.5.2": 4, "2.7.2": 4, "3.1.1": 12, "3.1.3": 8, "3.5.2": 1, "3.5.5": 0.25,
        "3.6.4": 0.5, "3.7.3": 1, "3.7.4": 4, "3.9.1": 4, "3.9.3": 9
    }


    crime_type_df = pd.DataFrame({'crime_code': max_jail_times.keys()})

    # Determine the category for each crime code
    crime_type_df['category'] = crime_type_df['crime_code'].apply(
        lambda code: next((category for category, codes in categories.items() if code in codes), None)
    )

    # Add the 'max_jailtime_yrs' column
    crime_type_df['max_jailtime_yrs'] = crime_type_df['crime_code'].map(max_jail_times)

    return crime_type_df

crime_type_df = crime_type_df()

def groupby_gm_year(crime_type_df, crime_df):

  crime_df_pivot = pd.pivot_table(crime_df, index=['municipality_id', 'year'], columns = 'crime_code', values='registered_crimes', aggfunc='sum', fill_value=0)
  crime_df_pivot = crime_df_pivot.reset_index()

  return crime_df_pivot

crime_counts_df = groupby_gm_year(crime_type_df, clean_crime_data)
crime_counts_df.columns.name = None

def weighted_sum_calc(crime_counts_df, crime_type_df):
  crime_counts_df['weighted_personal'] = 0
  crime_counts_df['weighted_property'] = 0
  crime_counts_df['weighted_societal'] = 0

  for index,row in crime_type_df.iterrows():
    crime_code = row['crime_code']
    category = row['category']
    max_jailtime_yrs = row['max_jailtime_yrs']

    crime_counts_df_col = crime_counts_df[crime_code]

    if category == 'Personal crime':
      crime_counts_df['weighted_personal'] += crime_counts_df_col * max_jailtime_yrs
    elif category == 'Property crime':
      crime_counts_df['weighted_property'] += crime_counts_df_col * max_jailtime_yrs
    elif category == 'Societal crime':
      crime_counts_df['weighted_societal'] += crime_counts_df_col * max_jailtime_yrs

  return crime_counts_df

updated_crime_counts_df = weighted_sum_calc(crime_counts_df, crime_type_df)

def divide_by_population(demo_df, crime_counts_df):

  demo_df['year']= demo_df['year'].astype(int)
  demo_df_subset = demo_df[['municipality_id', 'population', 'year']]
  crime_counts_df['year']=crime_counts_df['year'].astype(int)
  merged_data = pd.merge(crime_counts_df,  demo_df_subset,  on=['municipality_id', 'year'], how='inner')

  for column in ['weighted_personal', 'weighted_property', 'weighted_societal']:
    merged_data[column]= merged_data[column] / merged_data['population']

  merged_data.fillna(0, inplace=True)

  merged_data['X/P']= (merged_data['weighted_personal']+merged_data['weighted_property']+merged_data['weighted_societal'])

  return merged_data


x_over_p_df = divide_by_population(clean_demo_data, updated_crime_counts_df)

highest_x_p_2022 = x_over_p_df.loc[x_over_p_df['year'] == 2022, 'X/P'].max()

def crime_score(x_over_p, highest_value):
  # bins = [0, 0.1 * highest_value, 0.2 * highest_value, 0.3 * highest_value, 0.4 * highest_value, 0.5 * highest_value,
  #           0.6 * highest_value, 0.7 * highest_value, 0.8 * highest_value, 0.9 * highest_value, highest_value]
  # labels = list(range(1,11))
  # bins = [0, 0.2 * highest_value, 0.4 * highest_value, 0.6 * highest_value, 0.8 * highest_value, 1000* highest_value]
# labels = list(range(1,6))

  bins = [0, 0.333 * highest_value, 0.666 * highest_value, 1000* highest_value]
  # labels = list(range(1,4))
  labels = ["low_crime", "medium_crime", "high_crime"]


  x_over_p['crime_score'] = pd.cut(x_over_p['X/P'], bins=bins, labels=labels, include_lowest=True)

  #show distribution of crime_score
  return x_over_p

crime_score_df = crime_score(x_over_p_df, highest_x_p_2022)

def create_national_distribution(crime_score_df):
  national_df = crime_score_df.groupby('year')

  sums = national_df[["weighted_personal", "weighted_property", "weighted_societal","X/P"]].sum()

  sums["personal_percentage"] = (sums["weighted_personal"] / sums["X/P"]) * 100
  sums["property_ percentage"] = (sums["weighted_property"] / sums["X/P"]) * 100
  sums["societal_percentage"] = (sums["weighted_societal"] / sums["X/P"]) * 100


  national_table = sums[["personal_percentage", "property_ percentage", "societal_percentage"]]
  national_table.reset_index(inplace=True)

  return national_table

national_table = create_national_distribution(crime_score_df)


crime_score_df = crime_score_df.rename(columns={'X/P': 'XP'})

# Create a SQLAlchemy engine to connect to the PostgreSQL database
engine = create_engine("postgresql://student:infomdss@db_dashboard:5432/dashboard")

# Dump 'demo_df' DataFrame into 'demo_data' table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS demo_data CASCADE;"))
clean_demo_data.to_sql("demo_data", engine, if_exists="replace", index=False)

# Dump 'municipality_names_df' DataFrame into 'municipality_names' table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS municipality_names CASCADE;"))
municipality_names_df.to_sql("municipality_names", engine, if_exists="replace", index=False)

# Dump 'crime_data' DataFrame into 'crime_data' table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS crime_data CASCADE;"))
clean_crime_data.to_sql("crime_data", engine, if_exists="replace", index=False)

# Dump 'crime_type_df' DataFrame into 'crime_type' table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS crime_type CASCADE;"))
crime_type_df.to_sql("crime_type", engine, if_exists="replace", index=False)

# Dump 'crime_score_df' DataFrame into 'crime_score' table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS crime_score CASCADE;"))
crime_score_df.to_sql("crime_score", engine, if_exists="replace", index=False)
print('All data imported')