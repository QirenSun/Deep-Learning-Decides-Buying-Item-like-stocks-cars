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
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Short_MA', 'Long_MA']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

ticker='S'
start_date='2014-01-01'
end_date='2018-12-31'
s_window = 14
l_window = 50
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock(ticker, start_date, end_date, s_window, l_window)
df.to_csv(output_file, index=False)


with open(output_file) as f:
    lines = f.read().splitlines()



#df[['Short_MA', 'Long_MA', 'Adj Close']].plot()


#df_2 = pd.read_csv(output_file)
date=[]
W_day,all_w,week,l_week=[],[],[],[]
for i in range(1,len(lines)-1):
    W_day.append(str(lines[i+1].split(',')[4])+','+ str((eval(lines[i+1].split(',')[6])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]) )
    +','+str( (eval(lines[i+1].split(',')[5])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]) )
    +','+str( (eval(lines[i+1].split(',')[9])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]) ))
for i in range(1,len(lines)):
    l_week.append(lines[i].split(',')[4])
    date.append(lines[i].split(',')[0])
   
for i in range(1,len(lines)):
    
    if lines[i].split(',')[4]=='Monday':
        m=i
    elif lines[i].split(',')[4]=='Friday':
        n=i
        if n-m<=4 & n-m>0:
            week.append(date[m-1:n])
            
    
for i in range(1,len(W_day)):
    weekday=W_day[i].split(',')[0]
    all_w=[]
    if weekday=='Monday':
        med.append(i)
        '''
        for m in range(i,i+5):
            all_w.append(W_day[m].split(',')[3])
        all_w.sort()
        med.append(all_w[2])
    else:
        '''

for i in range(len(med)):
    if not med[i+1]-med[i]==5:
        print(False)  
        
#连续下降买入
W=[]
for i in range(1,len(lines)-3):        
        if float(lines[i+1].split(',')[9])-float(lines[i].split(',')[9])<0:
            if float(lines[i+2].split(',')[9])-float(lines[i+1].split(',')[9])<0:                
                W.append( str(lines[i+2].split(',')[0]) 
                +','+ str(float(lines[i+3].split(',')[9])-float(lines[i+2].split(',')[9]))
                +',')
date,W_date,trades=[],[],[]
for i in range(1,len(lines)):
    date.append(lines[i].split(',')[0])
for i in range(len(W)):
    W_date.append(W[i].split(',')[0])
    
x=input("Enter Dates like 20xx-xx-xx:")

for i in range(1,len(lines)):
    if lines[i].split(',')[0]==x:      
        for c in date[i:]:
            for d in range(len(W_date)):            
                if W_date[d]==c:
                    trades.append((W[d:d+10]))
profit=[]
for i in range(10):
    if float(trades[0][i].split(',')[1])>=0:
        profit.append(trades[0][i].split(',')[0]+',   10,'+'   1,   '+trades[0][i].split(',')[1]+',   0'+',   0')
    else:
        profit.append(trades[0][i].split(',')[0]+',   10,'+'   0,'+'   0,  '+'1,    '+trades[0][i].split(',')[1])

                    
profit.insert(0,'W    Trades     profitable  Profit/Per trade   losing   Loss/Per trade')
print(profit)
                        
                      
        














