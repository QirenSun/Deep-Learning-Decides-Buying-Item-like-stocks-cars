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
s_window = 10
l_window = 80
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock(ticker, start_date, end_date, s_window, l_window)
df.to_csv(output_file, index=False)


with open(output_file) as f:
    lines = f.read().splitlines()



#task 1 Add naïve strategy:
def year_pro_a1(year):    
    retu=np.array(df['Return'])
    #adj_close=np.array([df['Adj Close']])
    buy=np.where(retu <=0,retu,1)
    naive=np.where(buy>=0,buy,-1)
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    
    profit=0
    num=0
    for i in range(last_day-sum_year+1,last_day):
        if naive[i]== naive[i+1] and naive[i+1]==-1:
            profit+=float(format(100/df['Adj Close'][i+1],'0.2f'))
            num-=1
            last_adj=df['Adj Close'][i+1]
        elif naive[i]== naive[i+1] and  naive[i+1]==1:
            #profit+=float(format(-100/df['Adj Close'][i],'0.2f'))
            last_adj=df['Adj Close'][i]
            #num+=1
        else:
            last_adj=df['Adj Close'][i]
    profit_year=float(format(profit*last_adj+num*100,'0.2f'))    
    return profit_year    

year=[year_pro_a1(2014),year_pro_a1(2015),year_pro_a1(2016),year_pro_a1(2017),year_pro_a1(2018)]
print(year)


#---------------------------------------------------------------
#---------------------------------------------------------------
#---------------------------------------------------------------
#Task 2 generate and plot your label data set

'''
def year_pro_turtle(year,total):    
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    
    profit=0
    num=0
    #gu=0
    orgin=total
    buy,unit=0,0
    for i in range(last_day-sum_year+1,last_day+1):
        if df['Adj Close'][i]> df['Up_line'][i]:
            unit=total*0.01/df['ATR'][i]
            total=total-unit*df['Adj Close'][i]
            profit+=float(format(unit,'0.3f'))
            #gu+=unit
            buy=i
            last_adj=df['Adj Close'][i]
            num-=1
            print(total)
            
        elif df['Adj Close'][i]<= df['Up_line'][buy]-2*df['ATR'][i]:
            last_adj=df['Adj Close'][i]
            
            total+=float(format(profit*last_adj,'0.2f'))            
            print(total)
            profit=0
        elif df['Adj Close'][i]< df['Down_line'][i]:
            last_adj=df['Adj Close'][i]
            total+=float(format(profit*last_adj,'0.2f'))  
            num+=1
            profit=0
            print(total)
        elif df['Adj Close'][i]>= df['Up_line'][buy]+0.5*df['ATR'][i]:
            profit+=float(format(unit,'0.3f'))
            total=total-unit*df['Adj Close'][i]
            buy=i
            print(total)
            #gu+=unit
            last_adj=df['Adj Close'][i]
            

        else:
            last_adj=df['Adj Close'][i]
    profit_year=float(format(total-orgin,'0.2f'))    
    return profit_year   

'''

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
        df['Labels']=0
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday',
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return', 'Up_line', 'Down_line','Middle_line','true_range','ATR','Labels']
        df = df[col_list]
        return df
    except Exception as error:
        print(error)
        return None

ticker='S'
start_date='2014-01-01'
end_date='2019-01-01'
s_window = 20
l_window = 80
input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, ticker + '.csv')

df = get_stock_turtle(ticker, start_date, end_date, s_window, l_window)
#df.to_csv(output_file, index=False)


#with open(output_file) as f:
    #lines = f.read().splitlines()
'''
#Generate labels and get the year's profit
'''
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

for year in range(2014,2019):
    year_pro_turtle(year)
'''
For each week, compute the weekly return and weekly standard deviation
For 2018 you labeled your dataset (green and red) for each week
52 labelled weeks for 2018
(a)	Add more labels (at least for 2017)
(b)	Plot your labels as follows:
'''
from pandas.core.frame import DataFrame

def week(year):
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    start=last_day-sum_year+1
    end=last_day+1
    weekday=[]
    for i in range(start,end):
        weekday.append(df['Weekday'][i])
    
    week_list=list(range(60))
    coin=0
    dict1={}
    m,x_mean,x_std=0,[],[]
    seq,cou=[],0
    for i in range(start,end):
        if df['Weekday'][i]=='Monday':
            m=i
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
        
    data_1={'Week':seq,'Week_Mean':x_mean,'Week_std':x_std}
    stock_week=DataFrame(data_1)    
    
    x_s=[]
    y_l=[]  
    for x,y in dict1.items():
        x_s.append(x)
        y_l.append(y)
    y_l=np.array(y_l)
    
    
    color=np.where(y_l <= 0, y_l, 1)
    color=np.where(color >0, color, 0)
    colors=np.array(['r','g'])
    
    
    data = {'a':range(len(x_s)),
            'b':x_s,
            #'c':color.flatten().astype(int),
            'd':abs(y_l)
            }
    
    plt.scatter('a','b',c=colors[color.flatten('F').astype(int)],s='d', data=data)
    plt.xlabel('Week')
    plt.ylabel('')
    fig = plt.gcf()
    input_dir = r'C:\Users\Administrator\Desktop\Python_data'
    file_name = os.path.join(input_dir, str(year)+'labaled_weeks.pdf')
    fig.savefig(file_name)
    
    fig.show()
    
    return stock_week

week(2017)


'''
For each week in 2018, 
compute weekly return (Fri-Fri) and plot it together with your label on a graph:
'''
def week_return(year):
    
    sum_year=0
    for i in range(len(df)):
        if df['Year'][i]==year:
            sum_year+=1
            last_day=i
    start=last_day-sum_year+1
    end=last_day+1
    
    week_list=list(range(60))
    return_w,coin,num,dict1=[],0,0,{}
    for i in range(start,end):
        if df['Weekday'][i]=='Friday':
            m=i
            return_w.append(df['Adj Close'][i])
            for n in range(num,m+1):
                coin+=df['Adj Close'][n]*df['Labels'][n]
            if coin>=0:
                dict1.update({week_list[0]:coin})
                week_list.pop(0)
                coin=0
            else:
                dict1.update({week_list[0]:coin})
                week_list.pop(0)                
                coin=0           
        num=i
    
    w={'Adj Close':return_w}
    w_return=DataFrame(w)
    w_return['Return']=w_return['Adj Close'].pct_change()
    w_return['Return'].fillna(0, inplace = True)        
    
    
    x_s=[]
    y_l=[]  
    for x,y in dict1.items():
        x_s.append(x)
        y_l.append(y)
    y_l=np.array(y_l)
    
    
    color=np.where(y_l <= 0, y_l, 1)
    color=np.where(color >0, color, 0)
    colors=np.array(['r','g'])
    
    
    data = {'a':range(len(w_return['Return'])),
            'b':range(len(w_return['Return'])),
            'd':abs(w_return['Return']*300)
            }
    
    plt.scatter('a','b',c=colors[color.flatten('F').astype(int)],s='d', data=data)
    plt.xlabel('Week')
    plt.ylabel('')
    fig = plt.gcf()
    input_dir = r'C:\Users\Administrator\Desktop\Python_data'
    file_name = os.path.join(input_dir, str(year)+'_Return_weeks.pdf')
    fig.savefig(file_name)
    
    fig.show()
    return
    
week_return(2018)    
    
    
    
    


