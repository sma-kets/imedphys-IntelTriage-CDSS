#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports


# In[2]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# In[3]:


# functions


# In[171]:

npy_encoders = ['Φύλο ασθενή', 'Κατηγορία άφιξης ασθενή']

def discretize_age(age):
    #define age classes 18-25, 26-35, 35-40, 40-64, >64 age_labels = [1,2,3,4,5]
    age = [0+1000 if value<25 else value for value in age]
    age = [1+1000 if value< 35 else value for value in age ]
    age = [2+1000 if value< 40 else value for value in age ]
    age = [3+1000 if value< 65 else value for value in age ]
    age = [4+1000 if value<500 else value for value in age] 
    age = np.array(age)-1000
    return age

def discretize_n_edvisits(n_edvisits):
    n_edvisits = [0+1000 if value <= 3 else value for value in n_edvisits] 
    n_edvisits = [1+1000 if value <=7 else value for value in n_edvisits]
    n_edvisits = [2+1000 if value <=10 else value for value in n_edvisits]
    n_edvisits = [3+1000 if value <=500 else value for value in n_edvisits]
    n_edvisits = np.array(n_edvisits)-1000
    return n_edvisits

#5 classes for vitals, high-risk/low-risk
def discretize_triage_vital_temp(temp):
    temp = [0 + 1000 if value <=34.3 else value for value in temp]
    temp = [1 + 1000 if value <=35.6 else value for value in temp]
    temp = [2 + 1000 if value <=37.3 else value for value in temp]
    temp = [3 + 1000 if value <=38 else value for value in temp]
    temp = [4 + 1000 if value <=500 else value for value in temp]
    temp = np.array(temp)-1000
    return temp

def discretize_triage_vital_hr(hr):
    hr = [0+1000 if value <=48 else value for value in hr] 
    hr = [1+1000 if value <=59 else value for value in hr]
    hr = [2+1000 if value <=104 else value for value in hr]
    hr = [3+1000 if value <=109 else value for value in hr]
    hr = [4+1000 if value <=500 else value for value in hr]
    hr = np.array(hr)-1000
    return hr

def discretize_triage_vital_rr(rr):
    rr = [0 + 1000 if value <=12 else value for value in rr]
    rr = [1 + 1000 if value <=15 else value for value in rr]
    rr = [2 + 1000 if value <=22 else value for value in rr]
    rr = [3 + 1000 if value <=500 else value for value in rr]  
    rr = np.array(rr)-1000
    return rr

def discretize_triage_vital_sbp(sbp):
    sbp = [0 + 1000 if value <=98 else value for value in sbp]
    sbp = [1 + 1000 if value <=107 else value for value in sbp] 
    sbp = [2 + 1000 if value <=176 else value for value in sbp]
    sbp = [3 + 1000 if value <=199 else value for value in sbp]
    sbp = [4 + 1000 if value <=500 else value for value in sbp]
    sbp = np.array(sbp)-1000
    return sbp

def discretize_triage_vital_dbp(dbp):
    dbp = [0 if value <=70 else value for value in dbp]
    dbp = [1 if value > 1 else value for value in dbp] 
    return dbp

def discretize_triage_vital_o2(o2):
    o2 = [0 + 1000 if value <=89 else value for value in o2]  
    o2 = [1 + 1000 if value <=94 else value for value in o2]
    o2 = [2 + 1000 if value <=95 else value for value in o2]  
    o2 = [3 + 1000 if value <=500 else value for value in o2] 
    o2 = np.array(o2)-1000
    return o2

def discretize_continuous_variables(data, to_discret = []):
    # input: dataframe containing some continuous variables
    # output: discretized dataframe 
    # discretize continuous variables  providing a list of column names
    # set to_discret = ['all'] to discretize alla possible continuous variables
    
    if to_discret == ['all']:
        to_discret = data.columns

    for col in to_discret:
        if col == 'Ηλικία ασθενή':                                                 
            data.loc[:, col] = discretize_age(data[col])
        elif col == 'Καρδιακοί παλμοί (HR) (min)':                                 
            data.loc[:, col] = discretize_triage_vital_hr(data[col])
        elif col == 'Θερμοκρασία (C)':  
            data.loc[:, col] = discretize_triage_vital_temp(data[col])
        elif col == 'Αναπνευστικός ρυθμός (min)':    
            data.loc[:, col] = discretize_triage_vital_rr(data[col])
        elif col == 'Συστολική αρτηριακή πίεση (mmHg)':    
            data.loc[:, col] = discretize_triage_vital_sbp(data[col])
        elif col == 'Κορεσμός οξυγόνου (SpO2) (%)':      
            data.loc[:, col] = discretize_triage_vital_o2(data[col])
        elif col == 'Αριθμός νοσηλειών ή επισκέψεων (κατά τον περασμένο χρόνο)':   
            data.loc[:, col] = discretize_n_edvisits(data[col])
        elif col == 'Διαστολική αρτηριακή πίεση (mmg)':    
            data.loc[:, col] = discretize_triage_vital_dbp(data[col])
    
    return data

def MinMaxScaler(feature):
    minimum = np.min(feature)
    maximum = np.max(feature)
    scaled_feature = (feature - minimum)/(maximum - minimum)
    return scaled_feature.values

def fahreneiht_to_celsius(temperature):
    temperature = (temperature - 32) * 5/9
    return temperature

def label_encoder(y, col_name):
    original = y
    # create mask for nan values
    mask = y.isnull()
    y = LabelEncoder().fit_transform(y)
    # encode all values except nans
    y = pd.Series(y).where(~mask, original)
    return y

# input is a pandas dataframe containing data collected during triage
# output is teh processed pandas dataframe
# dataset_key = 'public_data' or 'AHEPA_dataset'
# to_discret = list of continuous variables to discretize e.g. age or vital signs, if empty pass procedure
# norm = boolean: to apply MinMax scaling
def preprocessing(data, dataset_key, discret_list = [], norm=False):   
    
    # encode label 
    y = [int(1) if value == 'Εισαγωγή' else int(0) if value == 'Εξιτήριο' else 'wrong label' for value in data['Αποτέλεσμα']]
    data['Αποτέλεσμα'] = y

    if dataset_key == 'public_data':
        # farehneit to celsius
        data.loc[:,'Θερμοκρασία (C)'] = fahreneiht_to_celsius(data['Θερμοκρασία (C)'].values)
                   
    # label encoder
    for col in npy_encoders:
        data.loc[:,col] = label_encoder(data[col], col)  

    # drop zero-variance columns
    data = data.loc[:, (data!=0).any(axis=0)]
    
    # discretize continuous variables
    data = discretize_continuous_variables(data, to_discret = discret_list)
    
    # scale dataset
    if norm:
        for col in data:  data.loc[:, col] = MinMaxScaler(data[col])
    
    # handling nan values, replace with 0
    data = data.fillna(0)
    return data


