
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import time

start_time = time.time()


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')


y_train = train_transaction["isFraud"]

train_transaction = train_transaction.drop(columns = ['isFraud'])


test_identity.columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
       'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
       'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
       'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
       'DeviceInfo']




transaction_data = pd.concat([train_transaction, test_transaction])
identity_data = pd.concat([train_identity, test_identity])


c = (identity_data.dtypes == 'object')
n = (identity_data.dtypes != 'object')
cat_id_cols = list(c[c].index)
num_id_cols = list(n[n].index) 


c = (transaction_data.dtypes == 'object')
n = (transaction_data.dtypes != 'object')
cat_trans_cols = list(c[c].index)
num_trans_cols = list(n[n].index) 


shape_of_train_trans = train_transaction.shape
shape_of_train_id    = train_identity.shape

shape_of_test_trans  = test_transaction.shape
shape_of_test_id     = test_identity.shape

del train_transaction
del train_identity
del test_transaction
del test_identity


for i in cat_id_cols:
    print(identity_data[i].value_counts())
    print(i, "missing values: ", identity_data[i].isnull().sum())
    print(identity_data[i].isnull().sum()*100/ len(identity_data[i]), "\n")


low_missing_cat_id_cols = []
medium_missing_cat_id_cols = []   
many_missing_cat_id_cols = []     

for i in cat_id_cols:
    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
    if percentage < 15:
        low_missing_cat_id_cols.append(i)
    elif percentage >= 15 and percentage < 60:
        medium_missing_cat_id_cols.append(i)
    else:
        many_missing_cat_id_cols.append(i)
        

for i in num_id_cols:
    print(identity_data[i].value_counts())
    print(i, "missing values: ", identity_data[i].isnull().sum()) 
    print(identity_data[i].isnull().sum()*100/len(identity_data[i]), "\n")


low_missing_num_id_cols = []
medium_missing_num_id_cols = [] 
many_missing_num_id_cols = []  

for i in num_id_cols:
    percentage = identity_data[i].isnull().sum() * 100 / len(identity_data[i])
    if percentage < 15:
        low_missing_num_id_cols.append(i)
    elif percentage >= 15 and percentage < 60:
        medium_missing_num_id_cols.append(i)
    else:
        many_missing_num_id_cols.append(i)
        

low_missing_num_trans_cols = []
medium_missing_num_trans_cols = []
many_missing_num_trans_cols = []

for i in num_trans_cols:
    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
    if percentage < 15:
        low_missing_num_trans_cols.append(i)
    elif percentage >= 15 and percentage < 60:
        medium_missing_num_trans_cols.append(i)
    else:
        many_missing_num_trans_cols.append(i)


low_missing_cat_trans_cols = []
medium_missing_cat_trans_cols = []
many_missing_cat_trans_cols = []

for i in cat_trans_cols:
    percentage = transaction_data[i].isnull().sum() * 100 / len(transaction_data[i])
    if percentage < 15:
        low_missing_cat_trans_cols.append(i)
    elif percentage >= 15 and percentage < 60:
        medium_missing_cat_trans_cols.append(i)
    else:
        many_missing_cat_trans_cols.append(i)


transaction_data = transaction_data.drop(columns = many_missing_num_trans_cols)
identity_data = identity_data.drop(columns = many_missing_num_id_cols)


n = (transaction_data.dtypes != 'object')
num_trans_cols = list(n[n].index) 

n = (identity_data.dtypes != 'object')
num_id_cols = list(n[n].index) 


from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy = 'mean') 
my_imputer.fit(transaction_data[low_missing_num_trans_cols])


transaction_data[low_missing_num_trans_cols] = my_imputer.transform(transaction_data[low_missing_num_trans_cols])



my_imputer = SimpleImputer(strategy = 'mean') 
my_imputer.fit(identity_data[low_missing_num_id_cols])

identity_data[low_missing_num_id_cols] = my_imputer.transform(identity_data[low_missing_num_id_cols])


my_imputer = SimpleImputer(strategy = 'median') 
my_imputer.fit(transaction_data[medium_missing_num_trans_cols])

transaction_data[medium_missing_num_trans_cols] = my_imputer.transform(transaction_data[medium_missing_num_trans_cols])


my_imputer = SimpleImputer(strategy = 'median') 
my_imputer.fit(identity_data[medium_missing_num_id_cols])


identity_data[medium_missing_num_id_cols] = my_imputer.transform(identity_data[medium_missing_num_id_cols])



object_counter = 0
int_counter = 0
float_counter = 0

not_detected = []

for i in transaction_data.columns:
        if transaction_data[i].dtype == 'object':
            object_counter += 1
        elif transaction_data[i].dtype == 'int':
            int_counter += 1
        elif transaction_data[i].dtype in ['float', 'float16', 'float32', 'float64']:
            float_counter += 1
        else:
            not_detected.append(i)
            

total = object_counter + int_counter  + float_counter

object_counter = 0
int_counter = 0
float_counter = 0

not_detected = []

for i in identity_data.columns:
        if identity_data[i].dtype == 'object':
            object_counter += 1
        elif identity_data[i].dtype == 'int':
            int_counter += 1
        elif identity_data[i].dtype in ['float', 'float16', 'float32', 'float64']:
            float_counter += 1
        else:
            not_detected.append(i)
            
            
total = object_counter + int_counter  + float_counter

if total != len(identity_data.columns):
    
   
    for i in not_detected:
        print(identity_data[i].dtype, "\n")


def detect_num_cols_to_shrink(list_of_num_cols, dataframe):
 
    convert_to_int8 = []
    convert_to_int16 = []
    convert_to_int32 = []
    
    convert_to_float16 = []
    convert_to_float32 = []
    
    for col in list_of_num_cols:
        
        if dataframe[col].dtype in ['int', 'int8', 'int32', 'int64']:
            describe_object = dataframe[col].describe()
            minimum = describe_object[3]
            maximum = describe_object[7]
            diff = abs(maximum - minimum)

            if diff < 255:
                convert_to_int8.append(col)
            elif diff < 65535:
                convert_to_int16.append(col)
            elif diff < 4294967295:
                convert_to_int32.append(col)   
                
        elif dataframe[col].dtype in ['float', 'float16', 'float32', 'float64']:
            describe_object = dataframe[col].describe()
            minimum = describe_object[3]
            maximum = describe_object[7]
            diff = abs(maximum - minimum)

            if diff < 65535:
                convert_to_float16.append(col)
            elif diff < 4294967295:
                convert_to_float32.append(col) 
        
    list_of_lists = []
    list_of_lists.append(convert_to_int8)
    list_of_lists.append(convert_to_int16)
    list_of_lists.append(convert_to_int32)
    list_of_lists.append(convert_to_float16)
    list_of_lists.append(convert_to_float32)
    
    return list_of_lists

num_cols_to_shrink_trans = detect_num_cols_to_shrink(num_trans_cols, transaction_data)

convert_to_int8 = num_cols_to_shrink_trans[0]
convert_to_int16 = num_cols_to_shrink_trans[1]
convert_to_int32 = num_cols_to_shrink_trans[2]

convert_to_float16 = num_cols_to_shrink_trans[3]
convert_to_float32 = num_cols_to_shrink_trans[4]


for col in convert_to_int16:
    transaction_data[col] = transaction_data[col].astype('int16') 
    
for col in convert_to_int32:
    transaction_data[col] = transaction_data[col].astype('int32') 

for col in convert_to_float16:
    transaction_data[col] = transaction_data[col].astype('float16')
    
for col in convert_to_float32:
    transaction_data[col] = transaction_data[col].astype('float32')
    

num_cols_to_shrink_id = detect_num_cols_to_shrink(num_id_cols, identity_data)

convert_to_int8 = num_cols_to_shrink_id[0]
convert_to_int16 = num_cols_to_shrink_id[1]
convert_to_int32 = num_cols_to_shrink_id[2]

convert_to_float16 = num_cols_to_shrink_id[3]
convert_to_float32 = num_cols_to_shrink_id[4]

# %% [code]
for col in convert_to_float16:
    identity_data[col] = identity_data[col].astype('float16')
    
for col in convert_to_float32:
    identity_data[col] = identity_data[col].astype('float32')
    
    
transaction_data = transaction_data.drop(columns = many_missing_cat_trans_cols)

identity_data = identity_data.drop(columns = many_missing_cat_id_cols)

c = (transaction_data.dtypes == 'object')
cat_trans_cols = list(c[c].index) 

c = (identity_data.dtypes == 'object')
cat_id_cols = list(c[c].index) 


low_card_trans_cols = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M6", "M7", "M8", "M9"]
high_card_trans_cols = ["P_emaildomain"]


for i in cat_trans_cols:
    most_frequent_value = transaction_data[i].mode()[0]
    transaction_data[i].fillna(most_frequent_value, inplace = True)

from sklearn.preprocessing import LabelEncoder
    
label_encoder = LabelEncoder()
transaction_data[high_card_trans_cols] = label_encoder.fit_transform(transaction_data[high_card_trans_cols])


low_card_id_cols =  ["id_12", "id_15", "id_16", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType"]
high_card_id_cols = ["id_30", "id_31", "id_33", "DeviceInfo"]
    

for i in cat_id_cols:
    most_frequent_value = identity_data[i].mode()[0]
    identity_data[i].fillna(most_frequent_value, inplace = True)

label_encoder = LabelEncoder()


for col in high_card_id_cols:
    identity_data[col] = label_encoder.fit_transform(identity_data[col])


low_card_trans_encoded = pd.get_dummies(transaction_data[low_card_trans_cols], dummy_na = False)
transaction_data.drop(columns = low_card_trans_cols, inplace = True)


low_card_id_encoded = pd.get_dummies(identity_data[low_card_id_cols], dummy_na = False)
identity_data.drop(columns = low_card_id_cols, inplace = True)




transaction_concatted = pd.concat([transaction_data, low_card_trans_encoded], axis = 1)


identity_concatted = pd.concat([identity_data, low_card_id_encoded], axis = 1)

train_transaction = transaction_concatted.iloc[0:590540]
test_transaction = transaction_concatted.iloc[590540:]

train_identity = identity_concatted.iloc[0:144233]
test_identity = identity_concatted.iloc[144233:]

train_data  = pd.concat([train_transaction, train_identity], axis = 1)


counter = 0

for i in train_data.columns:
    
    summ = train_data[i].isnull().sum()
    print(i, summ)
    if summ > 0:
        counter += 1
        


test_data  = pd.concat([test_transaction, test_identity], axis = 1)
counter = 0

for i in test_data.columns:
    
    summ = test_data[i].isnull().sum()
    print(i, summ)
    if summ > 0:
        counter += 1
        
from xgboost import XGBClassifier


clf = XGBClassifier(objective = 'binary:logistic',
                   gamma = 0.05,
                   colsample_bytree = 0.5, 
                   eval_metric = 'auc',
                   n_estimators = 1350,         
                   max_depth = 8,
                   min_child_weight = 2, 
                   learning_rate = 0.02,
                   subsample = 0.8,
                   n_jobs = -1,
                   silent = False,
                   verbosity = 0)        
                

clf.fit(train_data, y_train)


sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

if clf.predict_proba(test_data)[:,1] ==0:
    return True
else:
    return False