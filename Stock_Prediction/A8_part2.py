from pandas_datareader import data as web
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def get_stock_turtle(ticker, start_date, end_date, s_window, l_window):
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
        df['Up_line'] = df['High'].rolling(window=s_window, min_periods=1).mean()
        df['Down_line'] = df['Low'].rolling(window=s_window, min_periods=1).mean()    
        df['Middle_line'] = (df['Down_line']+df['Up_line'])/2 
        true_range=[0.0]
        for i in range(len(df)-1):
            true_range.append(max(df['High'][i+1]-df['Low'][i+1],df['High'][i+1]-df['Adj Close'][i],df['Adj Close'][i]-df['Low'][i+1]))
        df['true_range']=true_range
        df['ATR']= df['true_range'][1:].rolling(window=20,min_periods=1).mean()
        df['ATR'].fillna(0, inplace = True)
        df['Open']=df['Open']
        df['Labels']=0
        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Up_line', 'Down_line','Middle_line','true_range','ATR','Open','Labels']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

ticker='S'
start_date='2014-01-01'
end_date='2019-02-28'
s_window = 15
l_window = 80
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock_turtle(ticker, start_date, end_date, s_window, l_window)

sum_year=0
for i in range(len(df)):
    if df['Year'][i]==2018:
        sum_year+=1
        last_day=i
start=last_day-sum_year+1
end=last_day+1


def poly(w):
    rate=[]
    for i in range(start,end-w):
        x = np.array([range(1,w+1)]).reshape(w,)
        y = np.array(df['Adj Close'][start:(start+w)].tolist())
        degree = 1
        weights = np.polyfit(x,y,degree)
        model = np.poly1d(weights)
        predicted = model(11)
        if predicted > y[-1]:
            rate.append(1)
        else:
            rate.append(-1)
    
    day_return=df['Return'][start:end-w].tolist()
    day_return=[1 if i>0 else -1 for i in day_return]
    correct=np.sum(np.array(day_return)==np.array(rate))/len(day_return)
    print('W: '+str(w))
    print('Accyracy: '+str(round(correct,4)))
    print('\n'*3)
    return correct
#We would like to investigate the accuracy of using linear regression for different W
for w in [10,20,30]:
    poly(w)










