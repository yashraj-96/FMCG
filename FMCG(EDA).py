#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Importing the test and train dataset
missing_values= ["NA","na","--"]
FMCG_train=pd.read_csv("G://Data Science Project//FMCG//train1.csv",na_values=missing_values)
FMCG_test=pd.read_csv("G://Data Science Project//FMCG//test1.csv",na_values=missing_values)


#Creating a dataframe for the training dataset
df_train=pd.DataFrame(FMCG_train)
df_train.columns

#Dropping the irrelevant column
df_train=df_train.drop('Unnamed: 0',axis=1)

#Checking for the NA/null values
df_train.isnull()
print(df_train.isnull().sum())#The dataset doesn't have any missing values


#Checking the datatypes
df_train.dtypes

#Renaming the columns
df_train.columns
df_train=df_train.rename(columns={"PROD_CD":"Product_Code","SLSMAN_CD":"Salesman_Code","PLAN_MONTH":"Month","PLAN_YEAR":"Year","TARGET_IN_EA":"Target","ACH_IN_EA":"Achievement"})

#Dropping the duplicate rows'
df_train_duplicated=df_train[df_train.duplicated()]   
print('The duplicated rows for the FMCG dataset are :',df_train_duplicated)
#There are no duplicated rows in the following dataset

###############################################################################
#Converting the categorial columns top integer data type
df_train.dtypes
df_train.columns
LE=LabelEncoder()
#df_train2=df_train

x1=df_train['Product_Code']
x2=df_train['Salesman_Code']


y1=LE.fit_transform(x1)
y2=LE.fit_transform(x2)


#Dropping the categorial columns
df_train.drop(['Product_Code', 'Salesman_Code'],inplace=True,axis=1)

#Merging the label encoded/integer columns
df_train['Product_Code']=y1
df_train['Salesman_Code']=y2


#**********************************************************************************
#Using the 'Regular Expression' to allow special characters to be used  
import re
p=re.compile(r'\D')
#When the UNICODE flag is not specified, matches any non-digit character; this is equivalent to the set [^0-9]. 
#With UNICODE, it will match anything other than character marked as digits in the Unicode character properties database.
x3=df_train['Target']
x4=df_train['Achievement']
x3=[p.sub('',x) for x in x3]
x4=[p.sub('',x) for x in x4]

#Converting the required columns to numeric
df_train['Target']=pd.to_numeric(x3)
df_train['Achievement']=pd.to_numeric(x4)
df_train.dtypes

#Arranging the order of columns
df_train=df_train[['Product_Code', 'Salesman_Code', 'Month','Year','Target','Achievement']]

#****************************************************        
#The feature 'Year' does not seem to contribute to the output variable and can be dropped.
df_train.columns
df_train.drop(['Year'],inplace=True,axis=1)

###############################################################################
#FMCG_test
FMCG_test.columns
FMCG_test.drop('Unnamed: 6',inplace=True,axis=1)

#Renamking the columns
FMCG_test=FMCG_test.rename(columns={"PROD_CD":"Product_Code","SLSMAN_CD":"Salesman_Code","PLAN_MONTH":"Month","PLAN_YEAR":"Year","TARGET_IN_EA":"Target","Unnamed: 0":"Achievement"})

#Arranging the order of columns
FMCG_test=FMCG_test[['Product_Code', 'Salesman_Code', 'Month', 'Year','Target','Achievement']]

#Converting the datatype of the categorial features to int
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

a1=FMCG_test['Product_Code']
a2=FMCG_test['Salesman_Code']
        
b1=LE.fit_transform(a1)
b2=LE.fit_transform(a2)

FMCG_test.drop(['Product_Code','Salesman_Code'],inplace=True,axis=1)

FMCG_test['Product_Code']=b1
FMCG_test['Salesman_Code']=b2

#Using the 'Regular Expression' to allow special characters to be used  
import re
p=re.compile(r'\D')

a3=FMCG_test['Target']
a3=[p.sub('',x) for x in a3]

#Converting the required columns to numeric
FMCG_test['Target']=pd.to_numeric(a3)
FMCG_test.dtypes

#Arranging the order of columns
FMCG_test.columns
FMCG_test=FMCG_test[['Product_Code', 'Salesman_Code','Month', 'Year','Target', 'Achievement']]

df_test=FMCG_test
df_test.drop('Year',inplace=True,axis=1)
#**********************************************************

