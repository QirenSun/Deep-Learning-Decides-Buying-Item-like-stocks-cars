# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:37:29 2018

@author: epinsky
"""
# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.weekday_name  
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean()        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

def get_last_digit(y):
        x = str(round(float(y),2))
        x_list = x.split('.')
        fraction_str = x_list[1]
        if len(fraction_str)==1:
            return 0
        else:
            return int(fraction_str[1])


ticker='S'
start_date='2014-01-01'
end_date='2019-01-01'
s_window = 14
l_window = 50
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock(ticker, start_date, end_date, s_window, l_window)
df['last digit'] = df['Open'].apply(get_last_digit)

df['count'] = 1
total = len(df)

df_1 = df.groupby(['last digit'])['count'].sum()
df_2 = df_1.to_frame()
df_2.reset_index(level=0, inplace=True)
df_2['digit_frequency'] = df_2['count']/total
df_2['uniform'] = 0.10

output_file = os.path.join(input_dir, ticker + '_digit_analysis.csv')
df_2.to_csv(output_file, index=False)


#df.to_csv(output_file, index=False)
with open(output_file) as f:
    lines = f.read().splitlines()
#df[['Short_MA', 'Long_MA', 'Adj Close']].plot()
#df_2 = pd.read_csv(output_file)

#Task 1
df_mean=df['Adj Close'].mean()
df_std=df['Adj Close'].std()
nd_down=df_mean-2*df_std
nd_up=df_mean+2*df_std
df_num=df['Adj Close'].count()
df_value=df['Adj Close'].value_counts()
num=0
for i in df['Adj Close']:
    if i>nd_up or i<nd_down:
        num+=1
probability=100*num/df_num
print('mean: '+str(format(df_mean,'0.2f'))+'\nstd: '+str(format(df_std,'0.2f')))
print('Probability '+str(format(probability,'0.2f'))+'% is different with 5% predicted. The stock does not follow standard distribution.')

#Task 2
A=[float(format(i,'0.3f')) for i in df_2['digit_frequency']]
P=[i for i in df_2['uniform']]
print('A: '+str(A)+'\nP: '+str(P))

#max absolute error
mae_list=[]
for i in range(len(P)):
    mae_list.append(abs(A[i]-P[i]))    
mae=float(format(max(mae_list),'0.4f'))    
print('max absolute error: '+str(mae))


#median absolute error
median_list=[]
for i in range(len(P)):
    median_list.append(abs(A[i]-P[i]))    
median_list.sort() 
median_error=float(format((median_list[4]+median_list[5])/2,'0.4f'))
print('median absolute error: '+str(median_error))


#mean absolute error
mean_abs=0
for i in range(len(P)):
    mean_abs=+abs(A[i]-P[i])
mean_error=float(format(mean_abs/len(P),'0.4f'))
print('mean absolute error: '+str(mean_error))

#root mean squared error
root_mean_abs=0
for i in range(len(P)):
    root_mean_abs=+ (A[i]-P[i])**2
root_mean_error=float(format(math.sqrt(root_mean_abs/len(P)),'0.4f'))
print('root mean squred error: '+str(root_mean_error))



