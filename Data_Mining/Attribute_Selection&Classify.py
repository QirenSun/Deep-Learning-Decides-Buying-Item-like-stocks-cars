# -*- coding: utf-8 -*-
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
#attribute select
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2

dm=pd.read_csv(r'C:\Users\Administrator\Desktop\Data_mining\data.csv')


num=0
for i in range(0,len(dm),50):
    class1=[1,2,3,4,5,6,7,8,9,10]
    dm['type'][i:i+50]=class1[num]
    num+=1

#Attribute Selection
#1. ClassifierAttributeEval
X1=dm[['opusNum','followerNumInc','type','followerNum','name','nri','likeNum','commentNum','uid']]

#2. CfsSubsetEval
X2=dm[['nri','opusNum','originalMusicBeUsedNum']]
    

#3. CorrelationAttributeEval
X3=dm[['nri','likeNum','commentNum','likeNum','douyinFansNum','fansTotal','followerNum','repostNum',
    'toutiaoFansNum','originalMusicBeUsedNum','originalMusicNum','type']]

#4. ReliefFAttributeEval 
X4=dm[['nri','name','likeNum','commentNum','originalMusicBeUsedNum','toutiaoFansNum','repostNum','originalMusicNum',
      'followerNumInc','huoshanFansNum']]


#Chosen by myself
X5=dm[['commentNum','nri','followerNum','followerNumInc','type','repostNum','likeNum','opusNum',
      'likeOpusNum','fansTotal','douyinFansNum','dynamicNum']].values

Y=dm[['rankPosition']].values

def preprocess(X,Y):
    global X_train,X_test,Y_train,Y_test,x_vaild,y_vaild,X_test,Y_test
    le=LabelEncoder()
    z=StandardScaler().fit_transform(X)
    Y=le.fit_transform(Y)
    X_train,X_test,Y_train,Y_test=train_test_split(z,Y,test_size=0.3,random_state=2)
    x_vaild=X_test[-20:,:]
    y_vaild=Y_test[-20:]
    X_test=X_test[:-20,:]
    Y_test=Y_test[:-20]
    return


def er_rate():
    global error_rate    
    error_rate=[]
    for k in range(1,31,2):
        knn_classifier= KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train,Y_train)
        pred_k=knn_classifier.predict(X_test)
        error_rate.append(np.mean(pred_k!=Y_test))
    #y_test=np.array(np.random.randint(0,2,(52,)))
    #y_test=y_test.astype(float)
    
    plt.figure(figsize=(10,4))
    ax=plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1,31,2),error_rate,color='red',linestyle='dashed',
             marker='o',markerfacecolor='black',markersize=10)
    plt.title('Error rate vs. k for Tik Tok Subset')
    plt.xlabel('number of neighbors: k')
    plt.ylabel('Error Rate')
    return


def KNN(x_vaild,y_vaild,k):
    knn_classifier= KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,Y_train)
    new_instance=x_vaild
    prediction=knn_classifier.predict(new_instance.reshape(20,-1))

    print('KNN Prediction: ',prediction)
    print('KNN Vaild: ', y_vaild)
    print('KNN Correct', np.mean(prediction==y_vaild))
    return

def Naive_Bayesian(x_vaild,y_vaild):
    NB_classifier = GaussianNB().fit(X_train, Y_train)
    new_instance = x_vaild
    prediction = NB_classifier.predict(new_instance)
    
    print('NB Prediction: ',prediction)
    print('NB Vaild: ', y_vaild)
    print('NB Correct', np.mean(prediction==y_vaild))
    return

dict1={'Attribute Selection':X1,'CfsSubsetEval':X2,'CorrelationAttributeEval':X3,
       'ReliefFAttributeEval':X4,'Chosen by myself':X5}
for selection,i in dict1.items(): 
    print(selection)
    preprocess(i,Y)
    er_rate()
    num=error_rate.index(max(error_rate))
    k=2*num+1
    KNN(x_vaild,y_vaild,k) 
    Naive_Bayesian(x_vaild,y_vaild)
    print('\n')

        
    








       