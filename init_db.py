# Import Python packages
import os, sqlite3
import pandas as pd
from pandas import DataFrame

# Define DB file
db_file = 'webapp/data.db'

conn = sqlite3.connect(db_file)
c = conn.cursor()

communes = pd.read_csv('data/communes.csv')
communes.drop(['count'], axis=1).to_sql('COMMUNES', conn, if_exists='replace', index=True)

barrio = pd.read_csv('data/barrio_table.csv')
barrio.to_sql('BARRIOS', conn, if_exists='replace', index=False)

census = pd.read_csv('data/census/cleaned_data.csv')
census.to_sql('CENSUS', conn, if_exists='replace', index=False)

prop = pd.read_csv('data/prop_data/cleaned_data.csv')
prop.to_sql('PROPERTY', conn, if_exists='replace', index=True)

culture = pd.read_csv('data/Datasets/Culture/culture_table.csv')
culture.to_sql('CULTURE', conn, if_exists='replace', index=False)

health = pd.read_csv('data/Datasets/Health/health_table.csv')
health.to_sql('HEALTH', conn, if_exists='replace', index=False)

humanity = pd.read_csv('data/Datasets/Humanity/humanity_table.csv')
humanity.to_sql('HUMANITY', conn, if_exists='replace', index=False)

sports = pd.read_csv('data/Datasets/Sports/sports_table.csv')
sports.to_sql('SPORT', conn, if_exists='replace', index=False)

importance = pd.read_csv('analysis/MIR_Feature_Selection_Results/importance.csv')
importance.columns = ['id', 'feature', 'score', 'year', 'b_id']
importance.to_sql('IMPORTANCE', conn, if_exists='replace', index=False)

proper_barrios_rent = pd.read_json('data/properati_data/rent/properati_barrios.json')
proper_barrios_rent.to_sql('RENT', conn, if_exists='replace', index=False)

proper_barrios_rent = pd.read_json('data/properati_data/rent/properati_barrios_month.json')
proper_barrios_rent.to_sql('RENT_MO', conn, if_exists='replace', index=True)

proper_barrios_sell = pd.read_json('data/properati_data/sell/properati_sell.json')
proper_barrios_sell.to_sql('SELL', conn, if_exists='replace', index=False)

proper_barrios_sell = pd.read_json('data/properati_data/sell/properati_sell_month.json')
proper_barrios_sell.to_sql('SELL_MO', conn, if_exists='replace', index=True)