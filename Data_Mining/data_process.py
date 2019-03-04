# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:18:49 2019

@author: Administrator

import csv
dm=pd.read_csv(r'C:\Users\Administrator\Desktop\Data_mining\CSS_699_Dataset.csv')

lines=[]
with open("CSS_699_Dataset.csv","r",encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        lines.append(line)

for i in range(len(dm)):
    dm['name'][i]=i


num=0
for i in range(0,len(dm),50):
    class1=['Entertainment', 'Celebrity', 'Funny', 
            'Game', 'Sports', 'Business', 'Technology', 'Fashion', 'Social', 'Anime Comic and Games']
    dm['type'][i:i+50]=class1[num]
    num+=1

for i in range(0,len(dm)):
    if dm['rankPosition'][i] >20:
        dm['rankPosition'][i]=0
    else:
        dm['rankPosition'][i]=1
        
dm.rename({'likeNum':'CurrentOpusNum','opusNum':'CurrentOpusNum'},inplace=True)
       
dm.to_csv('data.csv',mode='w',encoding='utf-8-sig')
"""
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

dm=pd.read_csv(r'C:\Users\Administrator\Desktop\Data_mining\data.csv')


num=0
for i in range(0,len(dm),50):
    class1=[1,2,3,4,5,6,7,8,9,10]
    dm['type'][i:i+50]=class1[num]
    num+=1



X=dm[['commentNum','nri','followerNum','followerNumInc','type','repostNum','likeNum','opusNum',
      'likeOpusNum','fansTotal','douyinFansNum','dynamicNum']].values

Y=dm[['rankPosition']].values

le=LabelEncoder()
z=StandardScaler().fit_transform(X)
Y=le.fit_transform(Y)
X_train,X_test,Y_train,Y_test=train_test_split(z,Y,test_size=0.2,random_state=0)
x_vaild=X_test[-10:,:]
y_vaild=Y_test[-10:]
X_test=X_test[:-10,:]
Y_test=Y_test[:-10]
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

def KNN(x_vaild,y_vaild):
    knn_classifier= KNeighborsClassifier(n_neighbors=17)
    knn_classifier.fit(X_train,Y_train)
    new_instance=x_vaild
    prediction=knn_classifier.predict(new_instance.reshape(10,-1))

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

KNN(x_vaild,y_vaild)
Naive_Bayesian(x_vaild,y_vaild)











       