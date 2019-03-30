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

for year in range(2014,2020):
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
    global dict1,seq,x_mean,x_std,stock_week
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
    m,x_mean,x_std=start,[],[]
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

    data_1={'Week':seq,'Week_Mean':x_mean,'Week_std':x_std,'Label':label}
   
    stock_week=DataFrame(data_1)    
    stock_week['Week_Mean'].fillna(0, inplace = True)
    stock_week['Week_std'].fillna(0, inplace = True)

    
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
    
    '''
    #test week num
    print(test)
    for i in range(len(test)-1):
        if test[i+1]-test[i]>5:
            print('Count:',test[i],'   ',test[i+1])
    '''
    
    return stock_week

#week(2017)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
'''
#Training Set
#Testing Set
#Valid Set
'''
def train(year):
    week(year)
    global x_train,y_train
    stock=stock_week
    x_train=stock[['Week_Mean','Week_std']].values
    y_train=stock['Label'].values
    return
def test(year):
    week(year)
    global x_test,y_test
    stock=stock_week
    x_test=stock[['Week_Mean','Week_std']].values
    y_test=stock['Label'].values
    return

def vaild(year):
    week(year)
    global x_vaild,y_vaild
    stock=stock_week
    x_vaild=stock[['Week_Mean','Week_std']].values
    y_vaild=stock['Label'].values
    return
    

train(2017)
test(2018)
vaild(2019)
'''
Training Set
Testing Set
Vaild Set
KNN impelement
'''

Y=np.insert(y_test, 0, values=y_train, axis=0)
X=np.insert(x_test, 0, values=x_train, axis=0)
#X=np.insert(x_vaild,0,values=X,axis=0)
#Y=np.insert(y_vaild,0,values=Y,axis=0)
le=LabelEncoder()
z=StandardScaler().fit_transform(X)
Y=le.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(z,Y,test_size=0.5,shuffle=False)

def plot():    
    plt.figure(figsize=(10,4))
    ax=plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1,31,2),error_rate,color='red',linestyle='dashed',
             marker='o',markerfacecolor='black',markersize=10)
    plt.title('Error rate vs. k for Tik Tok Subset')
    plt.xlabel('number of neighbors: k')
    plt.ylabel('Error Rate')
    return

#manhattan
error_rate=[]
def manh(x):
    global error_rate
    y_t=[]
    for k in range(len(X_test)):
        k1=X_test[k]-X_train
        k1=abs(k1[:,0])+abs(k1[:,1])
        k1=k1.tolist()
        k2,y=[],[]
        for i in range(len(Y_train)):
            k2.append((k1[i],i))
        k2.sort()
        for m in k2[:x]:
            y.append(Y_train[m[1]])
        if y.count(0)>y.count(1):
            y_t.append(0)
        else:
            y_t.append(1)
    error_rate.append(np.sum(y_t!=Y_test)/len(Y_test))
    return 
for i in range(1,31,2):
    manh(i)
plot()


#Euclidean
error_rate=[]
def euclidean(x):
    global error_rate
    y_t=[]
    for k in range(len(X_test)):
        k1=(X_test[k]-X_train)**2
        k1=np.sqrt(abs(k1[:,0])+abs(k1[:,1]))
        k1=k1.tolist()
        k2,y=[],[]
        for i in range(len(Y_train)):
            k2.append((k1[i],i))
        k2.sort()
        for m in k2[:x]:
            y.append(Y_train[m[1]])
        if y.count(0)>y.count(1):
            y_t.append(0)
        else:
            y_t.append(1)
    error_rate.append(np.sum(y_t!=Y_test)/len(Y_test))
    #print('k=',x,'\nerror rate: ',error_rate,'\n')
    return 
for i in range(1,31,2):
    euclidean(i)
plot()

#Minkowski
error_rate=[]
def minkowski(x):
    global error_rate
    y_t=[]
    for k in range(len(X_test)):
        k3=[]
        for n in range(len(X_train)):
            k1=(X_test[k]-X_train[n])
            k3.append(np.linalg.norm(k1,ord=1.5))
        k2,y=[],[]
        for i in range(len(Y_train)):
            k2.append((k3[i],i))
        k2.sort()
        for m in k2[:x]:
            y.append(Y_train[m[1]])
        if y.count(0)>y.count(1):
            y_t.append(0)
        else:
            y_t.append(1)
    error_rate.append(np.sum(y_t!=Y_test)/len(Y_test))
    #print('k=',x,'\nerror rate: ',error_rate,'\n')
    return 
for i in range(1,31,2):
    minkowski(i)
plot()










