# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:25:17 2019

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

tips=pd.read_csv(r'C:\Users\Administrator\Desktop\Python_data\tips.csv')

#1.	Are tips higher for lunch or dinner?
tips['per']=tips['tip'][:]/tips['total_bill'][:] 
dinner=(tips.loc[tips['time'] == 'Dinner'])['per'].mean()
lunch=(tips.loc[tips['time'] == 'Lunch'])['per'].mean()
if dinner > lunch:
    print('Dinner is higher than lunch.')
else:
    print('Lunch is higher than dinner.')


#2.When are tips highest (which day and time)? 
day=['Thur','Sun','Sat','Fri']
d_tip,l_tip=[],[]
dinner=(tips.loc[tips['time'] == 'Dinner'])
lunch=(tips.loc[tips['time'] == 'Lunch'])
for i in day:
    d_tip.append((dinner.loc[ tips['day']==i])['per'].mean())
    l_tip.append((lunch.loc[ tips['day']==i])['per'].mean())

if max(d_tip)>max(l_tip):
    print('Day: ',day[d_tip.index(max(d_tip))],'\nTime: Dinner')
elif max(d_tip)<max(l_tip):
    print('Day: ',day[l_tip.index(max(l_tip))],'\nTime: Lunch')
    

#3.	Is there any relationship between price and tipping percentage?
per=tips['tip'][:]/tips['total_bill'][:] 
plt.scatter(tips['tip'],tips['total_bill'],s=per*100)
plt.xlabel('tip')
plt.ylabel('total_bill') 
print('The most part of tips are bewteen 10%-25% ')
    
#4.Any relationship between tip (as a percentage) and size of the group
plt.scatter(tips['size'],per)
plt.xlabel('size')
plt.ylabel('tips percentage')
print('The size of 1 may give more tips.')

#5.	What percentage of people are smoking?
smoker=tips.loc[tips['smoker'] == 'Yes']['size'].sum()
smoker_per=smoker/tips['size'].sum()
print('The persentage of people are smokers: ',smoker_per)

#6.	Assume that rows are arranged in time Are tips increasing with time?
plt.scatter(range(len(tips['size'])),per)
plt.xlabel('arrange in time')
plt.ylabel('tips percentage')
print('Tips do not increase with time. ') 

#7.	Any correlation between gender and time (assume that each meal is split into two equal times)
dinner=tips.loc[tips['time'] == 'Dinner']
lunch=tips.loc[tips['time'] == 'Lunch']
m_d=dinner.loc[dinner['sex']=='Male']['size'].sum()
f_d=dinner.loc[dinner['sex']=='Female']['size'].sum()
m_l=lunch.loc[lunch['sex']=='Male']['size'].sum()
f_l=lunch.loc[lunch['sex']=='Female']['size'].sum()

print('Dinner time has more male.')

#8.	Correlation between tip amounts from smokers and non-smokers
smoker=tips.loc[tips['smoker'] == 'Yes']['tip'].sum()
non_smoker=tips.loc[tips['smoker'] == 'No']['tip'].sum()
plt.scatter(tips['smoker'],per)
plt.xlabel('smoker')
plt.ylabel('tips percentage')
print('Smokers may give more tips.')

#9.	Average tip for each day of the week
day=['Thur','Sun','Sat','Fri']
wd_tip=[]
for i in day:
    wd_tip.append((tips.loc[ tips['day']==i])['per'].mean())
for i in range(len(day)):
    print(day[i],'tips: ',wd_tip[i])


#10.Which gender smokes more?
smoker=tips.loc[tips['smoker'] == 'Yes']
m=tips.loc[tips['sex']=='Male']['size'].sum()
fm=tips.loc[tips['sex']=='Female']['size'].sum()
s_m=(smoker.loc[smoker['sex']=='Male']['size'].sum())/m
s_f=(smoker.loc[smoker['sex']=='Female']['size'].sum())/fm
print('Male smokes more')





    