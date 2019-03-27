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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
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
X1=dm[['opusNum','followerNumInc','type','followerNum','name','likeNum','commentNum','uid']]

#2. CfsSubsetEval
X2=dm[['opusNum','originalMusicBeUsedNum']]
    

#3. CorrelationAttributeEval
X3=dm[['likeNum','commentNum','likeNum','douyinFansNum','fansTotal','followerNum','repostNum',
    'toutiaoFansNum','originalMusicBeUsedNum','originalMusicNum','type']]

#4. ReliefFAttributeEval 
X4=dm[['name','likeNum','commentNum','originalMusicBeUsedNum','toutiaoFansNum','repostNum','originalMusicNum',
      'followerNumInc','huoshanFansNum']]


#Chosen by myself
X5=dm[['commentNum','followerNum','followerNumInc','type','repostNum','likeNum','opusNum',
      'likeOpusNum','fansTotal','douyinFansNum','dynamicNum']]

Y=dm[['rankPosition']].values

def preprocess(X,Y,selection):
    global X_train,X_test,Y_train,Y_test,x_vaild,y_vaild,X_test,Y_test
    le=LabelEncoder()
    z=StandardScaler().fit_transform(X)
    Y=le.fit_transform(Y)
    X_train,X_test,Y_train,Y_test=train_test_split(z,Y,test_size=0.3,random_state=9)
    x_vaild=X_test[-20:,:]
    y_vaild=Y_test[-20:]
    X_test=X_test[:-20,:]
    Y_test=Y_test[:-20]
    #X_name=list(X.keys())
   # d1,d2={},{}
   # for i in range(X_train.shape[1]):
    #    d1.update({X_name[i]:X_train[:,i]})
    #    d2.update({X_name[i]:X_test[:,i]})
    #d1.update({'Rank_Class':Y_train})
    #d2.update({'Rank_Class':Y_test})
    #df1=pd.DataFrame(data=d1)
    #df2=pd.DataFrame(data=d2)
    #df1.to_csv(selection+'_train.csv',mode='w',encoding='utf-8-sig')
    #df2.to_csv(selection+'_test.csv',mode='w',encoding='utf-8-sig')
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
    y_score=knn_classifier.predict_proba(X_test)[:,1]

    new_instance=x_vaild
    prediction=knn_classifier.predict(new_instance.reshape(len(y_vaild),-1))

    #print('KNN Prediction: ',prediction)
    #print('KNN Vaild: ', y_vaild)
    print('KNN Correct', np.mean(prediction==y_vaild))
    #c_m=confusion_matrix(Y_test,prediction)

    fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score)
    print('TP rate: '+ str(np.mean(tpr_rf).tolist()))
    print('FP rate: '+ str(np.mean(fpr_rf).tolist()))
    print('ROC Area: '+str(roc_auc_score(Y_test, y_score).tolist()))
    print('\n') 
    return

def Naive_Bayesian(x_vaild,y_vaild):
    NB_classifier = GaussianNB().fit(X_train, Y_train)
    y_score=NB_classifier.predict_proba(X_test)[:,1]

    new_instance = x_vaild
    prediction = NB_classifier.predict(new_instance)
    
   # print('NB Prediction: ',prediction)
   # print('NB Vaild: ', y_vaild)
    print('NB Correct', np.mean(prediction==y_vaild))
    #c_m=confusion_matrix(Y_test,prediction)

    fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score)
    print('TP rate: '+ str(np.mean(tpr_rf).tolist()))
    print('FP rate: '+ str(np.mean(fpr_rf).tolist()))
    print('ROC Area: '+str(roc_auc_score(Y_test, y_score).tolist()))
    print('\n')
    return

def SVC_classify(X_test,Y_test):
    clf=SVC(gamma='auto')
    clf.fit(X_train,Y_train)
    y_score=clf.fit(X_train,Y_train).decision_function(X_test)

    prediction=clf.predict(X_test.reshape(len(Y_test),-1)) 
    correct=np.mean(prediction==Y_test)       
    #print('SVC Prediction: ',prediction)
    #print('SVC Vaild: ', Y_test)
    print('SVC Correct', correct)    
    #c_m=confusion_matrix(Y_test,prediction)

    fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score)
    print('TP rate: '+ str(np.mean(tpr_rf).tolist()))
    print('FP rate: '+ str(np.mean(fpr_rf).tolist()))
    print('ROC Area: '+str(roc_auc_score(Y_test, y_score).tolist()))
    print('\n')
    return

def mlp_classifier(X_test,Y_test):
    mlp=MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    #mlp.fit(X_train,Y_train)
    y_score=mlp.fit(X_train,Y_train).predict_proba(X_test)[:,1]
    prediction=mlp.predict(X_test.reshape(len(Y_test),-1)) 
    correct=np.mean(prediction==Y_test)   
    print('\nMLP Correct', correct) 
    #c_m=confusion_matrix(Y_test,prediction)

    fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_score)
    print('TP rate: '+ str(np.mean(tpr_rf).tolist()))
    print('FP rate: '+ str(np.mean(fpr_rf).tolist()))
    print('ROC Area: '+str(roc_auc_score(Y_test, y_score).tolist()))
    print('\n')   
    
    return   

dict1={'Attribute Selection':X1,'CfsSubsetEval':X2,'CorrelationAttributeEval':X3,
       'ReliefFAttributeEval':X4,'Chosen by myself':X5}
for selection,i in dict1.items(): 
    print(selection)
    preprocess(i,Y,selection)
    er_rate()
    num=error_rate.index(min(error_rate))
    k=2*num+1
    KNN(X_test,Y_test,k) 
    Naive_Bayesian(X_test,Y_test)
    SVC_classify(X_test,Y_test)
    mlp_classifier(X_test,Y_test)
    print('\n')


