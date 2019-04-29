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
        df['ON return']=(df['Open'][1:]-df['Adj Close'])/df['Adj Close']
        df['ON return'].fillna(0,inplace=True)
        df['Labels']=0
        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Up_line', 'Down_line','Middle_line','true_range','ATR','Open','ON return','Labels']
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


def year_pro_turtle(year):    
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    
    profit=0
    num=0
    buy=0
    global label
    label=[]
    for i in range(last_day-sum_year+1,last_day+1):
        if df['Adj Close'][i]> df['Up_line'][i]:
            profit+=float(format(100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
            num-=1   
            buy=i
            label.append(1)

        elif df['Adj Close'][i]<= df['Up_line'][buy]-2*df['ATR'][i]:
            profit+=float(format(-100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
            num+=1
            label.append(-1)
            
        elif df['Adj Close'][i]< df['Down_line'][i]:
            profit+=float(format(-100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
            num+=1
            buy=i
            label.append(-1)

        elif df['Adj Close'][i]>= df['Up_line'][buy]+0.5*df['ATR'][i]:
            profit+=float(format(100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
            num-=1            
            last_adj=df['Adj Close'][i]  
            label.append(1)

        else:
            last_adj=df['Adj Close'][i]
            label.append(0)

    labels(year)
    profit_year=float(format(profit*last_adj+100*num,'0.2f'))    
    return profit_year    


def labels(year):
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    start=last_day-sum_year+1
    end=last_day+1
    print('Year: '+str(year)+'\nStart:End '+str(start)+' : '+str(end))
    df['Labels'][start:end]=label
    df['Labels'].fillna(0, inplace = True)

    return df    

for year in range(2014,2020):
    year_pro_turtle(year)

for f in [2017,2018]:
    year=f
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    start=last_day-sum_year+1
    end=last_day+1
    num1,num2=0,0
    profit=0
    for i in range(start,end):
        if df['ON return'][i]>0:
            num1+=1
            shares=100/df['Open'][i]
            profit+=(-df['Open'][i]+df['Close'][i])*shares
        elif df['ON return'][i]<0:
            num2+=1
            shares=100/df['Open'][i]
            profit+=(df['Open'][i]-df['Close'][i])*shares
    
    from pandas.core.frame import DataFrame
    #global dict1,seq,x_mean,x_std,stock_week
    weekday=[]
    for i in range(start,end):
        weekday.append(df['Weekday'][i])
    
    week_list=list(range(60))
    coin=0
    dict1={}
    m,x_mean,x_std,x_label=start,[],[],[]
    seq,cou=[],0
    test,n=[],0
    for i in range(start,end):
        if df['Weekday'][i]=='Monday':
            m=i
            if m-n==5:
                for d in range(m-4,m):
                    coin+=df['Adj Close'][d]*df['Labels'][d]
                x_mean.append(df['Return'][m+5:n+1].mean())
                x_std.append(df['Return'][m+5:n+1].std())
                
                cou+=1
                seq.append('R'+str(cou))                        
                if coin>=0:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)
                    coin=0
                else:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)                
                    coin=0
                #print('Count: ',i)
                test.append(i)            
        elif df['Weekday'][i]=='Friday':
            n=i
            if n-m<=4:
                for d in range(m,n+1):
                    coin+=df['Adj Close'][d]*df['Labels'][d]
                x_mean.append(df['Return'][m:n+1].mean())
                x_std.append(df['Return'][m:n+1].std())
                cou+=1
                seq.append('R'+str(cou))
                if coin>=0:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)
                    coin=0
                else:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)
                    coin=0
               # print('Count: ',i)
                test.append(i)
            elif n-m==8:
                for d in range(m+5,n+1):
                    coin+=df['Adj Close'][d]*df['Labels'][d]
                x_mean.append(df['Return'][m+5:n+1].mean())
                x_std.append(df['Return'][m+5:n+1].std())
                cou+=1
                seq.append('R'+str(cou))                        
                if coin>=0:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)
                    coin=0
                else:
                    dict1.update({week_list[0]:coin})
                    week_list.pop(0)                
                    coin=0
                #print('Count: ',i)
                test.append(i)
    print('Week: ',len(seq))                
    x_s=[]
    y_l=[]  
    for x,y in dict1.items():
        x_s.append(x)
        y_l.append(y)
    y_l=np.array(y_l)
    
    
    color=np.where(y_l <= 0, y_l, 1)
    color=np.where(color >0, color, 0)
    colors=np.array(['r','g'])
    label=color.tolist()
    for i in range(len(label)):
        if label[i]>0:
            label[i]='green'
        else:
            label[i]='red'
    
    if year==2017:
        Week={'Week':list(range(1,len(seq)+1)),'Daily_return':x_mean,'Average_STD':x_std,'Label':label}
        xmean=x_mean
        xstd=x_std
        color1=color
    elif year==2018:
        Week1={'Week':list(range(52,len(seq)+52)),'Daily_return':x_mean,'Average_STD':x_std,'Label':label}
        xmean.extend(x_mean)
        xstd.extend(x_std)
        color2=color
color3=np.append(color1,color2)
w=DataFrame(Week1)
#print(w)

import math    
for i in range(len(xmean)):
    if math.isnan(xmean[i]) == True:
        xmean[i]=0.0
for i in range(len(xstd)):
    if math.isnan(xstd[i]) == True:
        xstd[i]=0.0

co=[]
for i in range(len(xstd)):
    y=xmean[i]+0.02
    if y>xstd[i]:
        co.append(0)
    else: 
        co.append(1)
co=np.array(co) 
data=pd.DataFrame({'Daily_return':xmean,'Average_STD':xstd,'Label':color3.tolist()},
                   columns = ['Daily_return', 'Average_STD', 'Label'])


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X=data[['Daily_return','Average_STD']].values
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = data['Label'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.5,shuffle=False)

error_rate=[]
for n in range(1,11):
    for d in range(1,6):
        clf = RandomForestClassifier(n_estimators=n,max_depth=d,criterion = 'entropy')
        clf = clf.fit(X_train,Y_train)
        accuracy = clf.score(X_test, Y_test)
        error_rate.append((1-accuracy)*1000)

#print('Descision Tree accuracy: ',accuracy)

error_rate=np.array(error_rate).reshape(10,5)
y,x=[],[]
for i in range(1,11):
    for m in range(1,6):
        y.append(i)
        x.append(m)
#x.reverse() 

N= int(np.argmin(error_rate)/5) + 1
D= np.argmin(error_rate)%5 + 1      
print('Best N: ',N)
print('Best D: ',D)
plt.scatter(x,y,s=error_rate)
plt.xlabel('D')
plt.ylabel('N')










