# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:22:28 2019

@author: Administrator
"""

# run this  !pip install pandas_datareader
from pandas_datareader import data as web
import os
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
        df['Weekday'] = df['Date'].dt.weekday_name  
        df['Short_MA'] = df['Adj Close'].rolling(window=s_window, min_periods=1).mean()
        df['Long_MA'] = df['Adj Close'].rolling(window=l_window, min_periods=1).mean() 
        df['Short_std'] = df['Adj Close'].rolling(window=s_window, min_periods=1).std()
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA','Short_std']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

def year_pro(year,k):    
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    
    profit=0
    
    for i in range(last_day-sum_year+1,last_day+1):
        if df['Adj Close'][i]< df['Short_MA'][i]-k*df['Short_std'][i]:
            #lines[i+1]+=','+(format(df['Adj Close'][i],'0.2f'))
            profit+=float(format(100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
        elif df['Adj Close'][i]> df['Short_MA'][i]+k*df['Short_std'][i]:
            #lines[i+1]+=','+(format(-df['Adj Close'][i],'0.2f'))
            profit+=float(format(-100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
        else:
            #lines[i+1]+=','+('0')
            last_adj=df['Adj Close'][i]
    #lines[0]+=',Trading'
    profit_year=float(format(profit*last_adj,'0.2f'))    
    return profit_year    
    
#year_pro(year=eval(input('Year: ')),k=eval(input('k: ')))    
    
def hyper():
    k=[0.5,1,1.5,2,2.5,3,3.5]
    profit_list=[]
    for i in k:    
        profit_list.append(year_pro(2018,i)) #Change Year Here
    return profit_list


ticker='S'
start_date='2014-01-01'
end_date='2019-01-01'

l_window = 50
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

trade={}
for i in range(10,101,10):
    s_window = i
    df = get_stock(ticker, start_date, end_date, s_window, l_window)
    trade.update({i:hyper()})
print(trade)

#df.to_csv(output_file, index=False)


#with open(output_file) as f:
#    lines = f.read().splitlines()
'''    
w=3    
for i in range(len(df)-3):
    retu=[r for r in df['Return'][i:i+w] ]      
    if max(retu) <0:
        i+w
'''  
def image():
    x_w=[]
    y_k=[]  
    for x,y in trade.items():
        x_w.append(x)
        y_k.append(y)
    #x_w=np.array(x_w)
    #y_k=np.array(y_k)
    k=[]
    w=[]
    for i in range(10):
        k.append([0.5,1,1.5,2,2.5,3,3.5])
    for i in range(7):    
        w.append(x_w)
    k=np.array(k)
    w=np.array(w).transpose()
    y_k=np.array(y_k)
    
    
      
    color=np.where(y_k <= 0, y_k, 1)
    color=np.where(color >0, color, 0)
    colors=np.array(['g','r'])
    
    
    data = {'a': k,
            'b':w,
            #'c':color.flatten().astype(int),
            'd':abs(y_k)/10
            }
    
    plt.scatter('b', 'a',c=colors[color.flatten().astype(int)],s='d', data=data)
    plt.xlabel('W')
    plt.ylabel('k')
    plt.show()
    return

image()


def year_pro_a3(year):    
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    
    profit=0
    
    for i in range(last_day-sum_year+1,last_day+1):
        if df['Short_MA'][i]> df['Long_MA'][i]:
            profit+=float(format(100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
        elif df['Short_MA'][i]< df['Long_MA'][i]:
            profit+=float(format(-100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
        else:
            last_adj=df['Adj Close'][i]
    profit_year=float(format(profit*last_adj,'0.2f'))    
    return profit_year    


    


ticker='S'
start_date='2014-01-01'
end_date='2019-01-01'


input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')


trade={}
pro=[]
for i in range(10,101,10):
    s_window = i
    for n in range(10,i+1,10):
        pro.append(0)        
    
    for m in range(i+10,101,10):
        l_window=m
        df = get_stock(ticker, start_date, end_date, s_window, l_window)
        pro.append(year_pro_a3(2014)) #Change the year here
        trade.update({i:pro})
    pro=[]
        
print(trade)


x_s=[]
y_l=[]  
for x,y in trade.items():
    x_s.append(x)
    y_l.append(y)
#x_w=np.array(x_w)
#y_k=np.array(y_k)
s_ma=[]
l_ma=[]
for i in range(10):
    s_ma.append([range(10,91,10)])
for i in range(9):    
    l_ma.append([range(10,101,10)])
s_ma=np.array(s_ma)
l_ma=np.array(l_ma).transpose()
y_l=np.array(y_l)





color=np.where(y_l <= 0, y_l, 1)
color=np.where(color >0, color, 0)
colors=np.array(['g','r'])


data = {'a': s_ma,
        'b':l_ma,
        #'c':color.flatten().astype(int),
        'd':abs(y_l.transpose())/20
        }

plt.scatter('b', 'a',c=colors[color.flatten('F').astype(int)],s='d', data=data)
plt.xlabel('Long_MA')
plt.ylabel('Short_MA')
plt.show()












