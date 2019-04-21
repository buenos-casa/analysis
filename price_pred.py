#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:49:46 2019

@author: breannamarielee
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv 

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, mutual_info_regression


##  Import the data
data = pd.read_csv('pgapyrs.csv')
#data = pd.read_csv('final_file_2.csv')

print("Data imported.")

#Include only rows with USD as currency
#data = data[data['currency'] == "USD"]

#Drop unnecessary columns
data = data.loc[:, data.columns != "Unnamed: 0"]
data = data.loc[:, data.columns != "id_left"] 
data = data.loc[:, data.columns != "created_on"]
data = data.loc[:, data.columns != "country_name"] 
data = data.loc[:, data.columns != "lat_lon"]
data = data.loc[:, data.columns != "lat_x"]
data = data.loc[:, data.columns != "lat_x.1"]
data = data.loc[:, data.columns != "lat_x.2"]
data = data.loc[:, data.columns != "lon_x"]
data = data.loc[:, data.columns != "lon_x.1"]
data = data.loc[:, data.columns != "lon_x.2"]
data = data.loc[:, data.columns != "currency"] #same value for all
data = data.loc[:, data.columns != "price_aprox_local_currency"] #have the same info in price_approx_usd
data = data.loc[:, data.columns != "coordinates"]
data = data.loc[:, data.columns != "index_right"]
data = data.loc[:, data.columns != "id_right"]
data = data.loc[:, data.columns != "health_id"] #too many nans
data = data.loc[:, data.columns != "health_name"] #too many nans
data = data.loc[:, data.columns != "property_values_Date"] 
data = data.loc[:, data.columns != "property_values_Longitude"] 
data = data.loc[:, data.columns != "property_values_Latitude"] 
data = data.loc[:, data.columns != "public_wifi_lat"]
data = data.loc[:, data.columns != "public_wifi_long"]
data = data.loc[:, data.columns != "public_wifi_object"] #same for all
data = data.loc[:, data.columns != "sports_long"]
data = data.loc[:, data.columns != "sports_lat"]
data = data.loc[:, data.columns != "sports_id"]
data = data.loc[:, data.columns != "health_long"]
data = data.loc[:, data.columns != "health_lat"]
data = data.loc[:, data.columns != "transportation_long"]
data = data.loc[:, data.columns != "transportation_lat"]
data = data.loc[:, data.columns != "barrio"] #same infor as b_id
data = data.loc[:, data.columns != "comuna"]
data = data.loc[:, data.columns != "geonames_id"]
data = data.loc[:, data.columns != "price_aprox_usd"] #exact same column as price
data = data.loc[:, data.columns != "property_values_b_id"]
data = data.loc[:, data.columns != "Distance dates"]
data = data.loc[:, data.columns != "dataset_date"]


# Remove categorical columns
data = data.loc[:, data.columns != 'operation']
data = data.loc[:, data.columns != 'property_type']
data = data.loc[:, data.columns != 'place_name']
data = data.loc[:, data.columns != 'state_name']
data = data.loc[:, data.columns != 'Computer Quantile']
data = data.loc[:, data.columns != 'Cellular Quantile']
data = data.loc[:, data.columns != 'Rent Quantile']
data = data.loc[:, data.columns != 'Immigration Quantile']
data = data.loc[:, data.columns != 'Education Quantile']
data = data.loc[:, data.columns != 'Owner Quantile']
data = data.loc[:, data.columns != 'Regular Quantile']
data = data.loc[:, data.columns != 'Uninhabited Quantile']
data = data.loc[:, data.columns != 'health_object']
data = data.loc[:, data.columns != 'health_comune']
data = data.loc[:, data.columns != 'property_values_Commune']
data = data.loc[:, data.columns != 'public_wifi_id']
data = data.loc[:, data.columns != 'public_wifi_name']
data = data.loc[:, data.columns != 'public_wifi_comune']
data = data.loc[:, data.columns != 'sports_object']
data = data.loc[:, data.columns != 'sports_name']
data = data.loc[:, data.columns != 'sports_comune']
data = data.loc[:, data.columns != 'transportation_id']
data = data.loc[:, data.columns != 'transportation_object']
data = data.loc[:, data.columns != 'transportation_name']
data = data.loc[:, data.columns != 'Mislabled barrio']
data = data.loc[:, data.columns != 'b_id']






cols = data.columns.tolist()

cat_data = [] # emptied

cont_data = []  # column names of continuous features
for x in cols:
    if x not in cat_data:
        cont_data.append(x)

#cont_data.remove("price")
#cont_data.remove("price_aprox_usd")


scaler = StandardScaler()
for column in cont_data:
    print(column)
    if column == 'Year':
        continue
    #print(column)
    x = data[[column]].values.astype(float)
    x_scaled = scaler.fit_transform(x)  #array of scaled values
    data[column] = x_scaled  #replace values with scaled valued
    
print("Categorical data is scaled.")    


## OneHotEncoding (convert categorical data) 

data = pd.get_dummies(data, columns=cat_data)
#print(OHE_data.columns.tolist())

print("Categorical features succesffuly converted.")


##  Split data by year: 2015, 2016, 2017, 2018

#Will use the dataset_date column

#make lists to hold indices of rows with the respective year

year2015 = []
year2016 = []
year2017 = []
year2018 = []

print("Gathering indices to group data by year....")
for index, row in data.iterrows():
    #yearval = data.loc[index,'dataset_date'] 
    #year = int(str(yearval)[:4])
    year = data.loc[index,'Year'] 
    if year == 2015:
        year2015.append(index)
    elif year == 2016:
        year2016.append(index)
    elif year == 2017:
        year2017.append(index)
    elif year == 2018:
        year2018.append(index)
    else:
        print("Error in assigning indices by year")


# Create dataframes grouped by year


data_2015 = pd.DataFrame(columns = cols)
data_2015 = data.loc[year2015]

data_2016 = pd.DataFrame(columns = cols)
data_2016 = data.loc[year2016]

data_2017 = pd.DataFrame(columns = cols)
data_2017 = data.loc[year2017]

data_2018 = pd.DataFrame(columns = cols)
data_2018 = data.loc[year2018]

print("Data is now grouped by year")

print("Starting Feature Selection")

def my_score(X, y):
    return mutual_info_regression(X, y, random_state=0, n_neighbors=2)

#split 2015 by barrio
print("Starting 2015....")

data_2015 = data_2015.loc[:, data_2015.columns != "Year"]
cols = data_2015.columns.tolist()


df = pd.DataFrame(columns = cols)            # create empty dataframe             
for index, row in data_2015.iterrows():      # filter for respective bid - iterate through each row
        # access data using column names
        #print(index,row['b_id'])
    
    df_temp = pd.Series(data_2015.loc[index])
           
    df = df.append(df_temp,ignore_index=True)
    #remove b_id
    #df = df.loc[:, df.columns != "b_id"]
    
X = df.loc[:, df.columns != "gap"]
y = df.loc[:, "gap"]
#feature selection
selector = SelectKBest(score_func=my_score, k='all').fit(X,y)
#x_new = selector.transform(X)
scores = selector.scores_
feature_labels = X.columns.tolist()
results = np.vstack((feature_labels, scores))
results = results.T
s = sorted(results, key=lambda x:x[1])
s = np.flipud(s)
       
#Export CSV
filename = "2015_"+"_rankedFeatures_gapPred.csv"
#df.to_csv(filename,index=False, columns = cols)
with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(s)
    
#split 2016 by barrio
print("Starting 2016....")

data_2016 = data_2016.loc[:, data_2016.columns != "Year"]
cols = data_2016.columns.tolist()


df = pd.DataFrame(columns = cols)            # create empty dataframe             
for index, row in data_2016.iterrows():      # filter for respective bid - iterate through each row
        # access data using column names
        #print(index,row['b_id'])
    
    df_temp = pd.Series(data_2016.loc[index])
           
    df = df.append(df_temp,ignore_index=True)
    #remove b_id
    #df = df.loc[:, df.columns != "b_id"]
    
X = df.loc[:, df.columns != "gap"]
y = df.loc[:, "gap"]
#feature selection
selector = SelectKBest(score_func=my_score, k='all').fit(X,y)
#x_new = selector.transform(X)
scores = selector.scores_
feature_labels = X.columns.tolist()
results = np.vstack((feature_labels, scores))
results = results.T
s = sorted(results, key=lambda x:x[1])
s = np.flipud(s)
       
#Export CSV
filename = "2016_"+"_rankedFeatures_gapPred.csv"
#df.to_csv(filename,index=False, columns = cols)
with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(s)


#split 2017 by barrio
print("Starting 2017....")

data_2017 = data_2017.loc[:, data_2017.columns != "Year"]
cols = data_2017.columns.tolist()


df = pd.DataFrame(columns = cols)            # create empty dataframe             
for index, row in data_2017.iterrows():      # filter for respective bid - iterate through each row
        # access data using column names
        #print(index,row['b_id'])
    
    df_temp = pd.Series(data_2017.loc[index])
           
    df = df.append(df_temp,ignore_index=True)
    #remove b_id
    #df = df.loc[:, df.columns != "b_id"]
    
X = df.loc[:, df.columns != "gap"]
y = df.loc[:, "gap"]
#feature selection
selector = SelectKBest(score_func=my_score, k='all').fit(X,y)
#x_new = selector.transform(X)
scores = selector.scores_
feature_labels = X.columns.tolist()
results = np.vstack((feature_labels, scores))
results = results.T
s = sorted(results, key=lambda x:x[1])
s = np.flipud(s)
       
#Export CSV
filename = "2017_"+"_rankedFeatures_gapPred.csv"
#df.to_csv(filename,index=False, columns = cols)
with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(s)



#split 2018 by barrio
print("Starting 2018....")

data_2017 = data_2018.loc[:, data_2017.columns != "Year"]
cols = data_2018.columns.tolist()


df = pd.DataFrame(columns = cols)            # create empty dataframe             
for index, row in data_2018.iterrows():      # filter for respective bid - iterate through each row
        # access data using column names
        #print(index,row['b_id'])
    
    df_temp = pd.Series(data_2018.loc[index])
           
    df = df.append(df_temp,ignore_index=True)
    #remove b_id
    #df = df.loc[:, df.columns != "b_id"]
    
X = df.loc[:, df.columns != "gap"]
y = df.loc[:, "gap"]
#feature selection
selector = SelectKBest(score_func=my_score, k='all').fit(X,y)
#x_new = selector.transform(X)
scores = selector.scores_
feature_labels = X.columns.tolist()
results = np.vstack((feature_labels, scores))
results = results.T
s = sorted(results, key=lambda x:x[1])
s = np.flipud(s)
       
#Export CSV
filename = "2018_"+"_rankedFeatures_gapPred.csv"
#df.to_csv(filename,index=False, columns = cols)
with open(filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(s)


# Feature selection - based on properati price
print("Data is now grouped by barrio and year")
print("CSV files export for feature selection is complete.")



