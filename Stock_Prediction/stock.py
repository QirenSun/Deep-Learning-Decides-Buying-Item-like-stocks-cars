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
end_date='2019-01-01'
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



'''
Week
(Adjust_Close2-Adjust_Close1)/Adjust_Close1
(High2-Adjust_Close1)/Adjust_Close1
(Low2-Adjust_Close1)/Adjust_Close1

Month
Adjust_Close=Total Adjust_Close of the Month/Days
(Adjust_Close2-Adjust_Close1)/Adjust_Close1
(High2-Adjust_Close1)/Adjust_Close1
(Low2-Adjust_Close1)/Adjust_Close1
'''
# 1. Each day of the week compute average, min and max of daily returns
def week_choose():
    date,best_day=[],[]
    W_day,volume=[],[]
    avg=[]
    for i in range(1,len(lines)-1):
        W_day.append(str(lines[i+1].split(',')[4])+',      '+ str(format(100*(eval(lines[i+1].split(',')[6])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]),'0.2f') )+'%'
        +',  '+str( format(100*(eval(lines[i+1].split(',')[5])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]),'0.2f') )+'%'
        +',  '+str( format(100*(eval(lines[i+1].split(',')[9])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]),'0.2f') )+'%')
        
        avg.append(format((eval(lines[i+1].split(',')[9])-eval(lines[i].split(',')[9])) / eval(lines[i].split(',')[9]),'0.4f'))
    
    for i in range(1,len(lines)):
        l_week.append(lines[i].split(',')[4])
        date.append(lines[i].split(',')[0])
        volume.append(lines[i].split(',')[8])
    m=10   
    
    
    
    
    for i in range(1,len(W_day)):
        med,vol=[],0  
        if W_day[i].split(',')[0]=='Monday':
            m=i
            
        elif W_day[i].split(',')[0]=='Friday':
            n=i        
            if n-m<=4 :
                for d in range(m,n+1):
                    med+=int(int(volume[d+1])/100000)*avg[d].split(' ')
                    vol+=int(int(volume[d+1])/100000)
                med=[eval(i) for i in med]
                med.sort()
                med=100*med[int(vol/2)]
                max1={}            
                for c in range(m,n+1):
                    W_day[c]=W_day[c]+',  '+str(format(med,'0.2f'))+'%'
                    dict1={float(W_day[c].split(',')[3].lstrip().replace('%','')):c}
                    max1.update(dict1)
                m1=max(max1)
                best_day.append(W_day[max1[m1]])
            elif n-m==8:
                for d in range(m+5,n+1):
                    med+=int(int(volume[d+1])/100000)*avg[d].split(' ')
                    vol+=int(int(volume[d+1])/100000)
                med=[eval(i) for i in med]
                med.sort()
                med=100*med[int(vol/2)]
                max1={}            
                for c in range(m+5,n+1):
                    W_day[c]=W_day[c]+',  '+str(format(med,'0.2f'))+'%'
                    dict1={float(W_day[c].split(',')[3].lstrip().replace('%','')):c}
                    max1.update(dict1)
                m1=max(max1)
                best_day.append(W_day[max1[m1]])

    
    W_day.insert(0,'Day of the week '+'min   '+'max   '+'average   '+'median  ')    
    best_day.insert(0,'Best day the week '+'min   '+'max   '+'average   '+'median  ')
         
    return best_day[:30]
                              
week_choose()            


         
# 2. For each month of the week, compute average, min and max of daily returns
def month_choose():
    ac,Min,Max,c,Min1,Max1,ac1=0,0,0,0,[],[],[]
    volume1,vol1,month1=0,[],[]
    date=[]
    volume=[]
    for i in range(1,len(lines)):
        l_week.append(lines[i].split(',')[4])
        date.append(lines[i].split(',')[0])
        volume.append(lines[i].split(',')[8])
    
    for i in range(1,len(lines)-1):
        if  int(lines[i+1].split(',')[3])>int(lines[i].split(',')[3]):
           
            ac+=  float(lines[i].split(',')[9])
            Min+= float(lines[i].split(',')[6])          
            Max+= float(lines[i].split(',')[5])
            volume1+=int(int(volume[i-1])/100000)
            c+=1
        else:
            ac1.append(format(float(ac)/c,'0.4f'))
            Min1.append(format(float(Min)/c,'0.4f'))
            Max1.append(format(float(Max)/c,'0.4f'))    
            month1.append(lines[i].split(',')[2])
            vol1.append(volume1)
            ac,Min,Max,c,volume1=0,0,0,0,0
            #lines[i].split(',')[2]+','+
    
    Min,Max,ac=[],[],[]
    for i in range(len(ac1)-1):
        Min.append(month1[i]+',  '+format(100*(float(Min1[i+1])-float(ac1[i]))/float(ac1[i]),'0.2f')+'%'
                   +','+ format(100*(float(Max1[i+1])-float(ac1[i]))/float(ac1[i]),'0.2f')+'%'
                   +','+ format(100*(float(ac1[i+1])-float(ac1[i]))/float(ac1[i]),'0.2f')+'%')
        #Max.append(format(100*(float(Max1[i+1])-float(ac1[i]))/float(ac1[i]),'0.2f')+'%')    
        #ac.append(format(100*(float(ac1[i+1])-float(ac1[i]))/float(ac1[i]),'0.2f')+'%')
    med,med1=[],[]            
    for i in range(len(month1)-1):
        if int(month1[i+1])>int(month1[i]):
            med.append(vol1[i]*(str(ac1[i])+' '))
            if i==len(month1)-2:
                med.append(vol1[i+1]*(str(ac1[i+1])+' '))
                med1.append(med)
        elif int(month1[i+1])<int(month1[i]):
            med.append(vol1[i]*(str(ac1[i])+' '))
            med1.append(med)
            med=[]
    
    med3=[]
    for i in range(len(med1)):
        med2=[]
        for m in range(len(med1[i])):
            med2+=med1[i][m].split(' ')[:-1]        
        med2=[float(i) for i in med2]
        med2.sort()    
        med3.append(med2[int(len(med2)/12)])
    c,m=0,0
    for i in range(len(Min)):
        c+=1
        if  c%12==0: 
            m+=1
            Min[i]+=','+str(med3[m])+'%'
        else:
            
            Min[i]+=','+str(med3[m])+'%'
    best_month=[]
    max1={}
    for i in range(1,len(Min)):
        if i%12!=0:          
            dict1={float(Min[i].split(',')[3].lstrip().replace('%','')):i}
            max1.update(dict1)   
            if i==len(Min)-1:
                m1=max(max1)
                best_month.append(Min[max1[m1]])   
        elif i%12==0:
            m1=max(max1)
            best_month.append(Min[max1[m1]])   
            max1={} 
                
    Min.insert(0,'Month '+'  min '+'  max '+'average '+'median')     
    best_month.insert(0,'Best month the year '+'min   '+'max   '+'average   '+'median  ')
                   
    return best_month            
                
month_choose()            
            

# 3. After W consecutive declines, buy on day W ($100), sell on W+1
# Buying at Date and Selling at Date+1. Geting profit/loss.
def conse_declines(x):    
    W=[]
    for i in range(1,len(lines)-4):        
            if float(lines[i+1].split(',')[9])-float(lines[i].split(',')[9])<0:
                if float(lines[i+2].split(',')[9])-float(lines[i+1].split(',')[9])<0:   
                    if float(lines[i+3].split(',')[9])-float(lines[i+2].split(',')[9])<0:
                        W.append( str(lines[i+3].split(',')[0]) 
                        +','+ str(format(float(lines[i+4].split(',')[9])-float(lines[i+3].split(',')[9]),'0.2f'))
                        +',')
    date,W_date,trades=[],[],[]
    for i in range(1,len(lines)):
        date.append(lines[i].split(',')[0])
    for i in range(len(W)):
        W_date.append(W[i].split(',')[0])
        
    
    
    for i in range(1,len(lines)):
        if lines[i].split(',')[0]==x:      
            for c in date[i:]:
                for d in range(len(W_date)):            
                    if W_date[d]==c:
                        trades.append((W[d:d+10]))
      
            
    profit=[]
    for i in range(10):
        if float(trades[0][i].split(',')[1])>=0:
            profit.append(trades[0][i].split(',')[0]+',   10,'+'   10,      '+str(format(10*float(trades[0][i].split(',')[1]),'0.2f'))+',         0'+',   0')
        else:
            profit.append(trades[0][i].split(',')[0]+',   10,'+'   0,   '+format('   0','4s')+',      '+'      10,    '+str(format(10*float(trades[0][i].split(',')[1]),'0.2f')))
    
                        
    profit.insert(0,'Date        Trades     profit  Profit/Per  losing   Loss/Per trade')
    return profit

x=input("Enter Dates like 2014-01-02:")
conse_declines(x)                            
                      
# 4. If adj close > s_ma then buy
def buy_sell():
    buy,sell=[],[]
    for i in range(1,len(lines)):
        if float(lines[i].split(',')[9]) > float(lines[i].split(',')[11]):
            buy.append(lines[i].split(',')[0]+',  '+str(format(float(lines[i].split(',')[9]),'0.4f')))
        else:
            sell.append(lines[i].split(',')[0]+',  '+str(format(float(lines[i].split(',')[9]),'0.4f')))              
    buy.insert(0,'Date        Buy')
    sell.insert(0,'Date        Sell')
    return buy[:20],sell[:20]

buy_sell()    






