# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open', 'Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']* 100.0
  
df['HL_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']* 100.0
  
df = df[['Adj. Close', 'HL_PCT', 'HL_Change', 'Adj. Volume']]

#extratc wanted values to create a label
forcast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#try to forcast using previous data

forcast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forcast_col].shift(-forcast_out)

print(df.tail())

print(df.head())

#using features as X and labels as y

X = np.array(df.drop(['label'], 1))


#Scaling our data
X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]


df.dropna(inplace=True)

y = np.array(df['label'])
y = np.array(df['label'])

#Create train and test
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#find the classifier

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
#print(accuracy)

#Predict the X data

forcast_set = clf.predict(X_lately)
print(forcast_set, accuracy, forcast_out)
df['Forcast'] = np.nan
#find out what the last day data is

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

#Populate the dataframe with the new date and the forcast values
for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i]
    
    df['Adj. Close'].plot()
    df['Forcast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    