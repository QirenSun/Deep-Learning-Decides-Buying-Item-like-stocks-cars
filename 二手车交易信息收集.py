from random import randint
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
import operator
from collections import OrderedDict
import datetime
import random

#url='https://www.autotrader.com/cars-for-sale/BMW/X6/Brighton+MA-02135?zip=02135&marketExtension=true&startYear=1981&endYear=2019&makeCodeList=BMW&searchRadius=50&modelCodeList=X6&sortBy=relevance&numRecords=100&firstRecord=0'
#url='https://www.autotrader.com/cars-for-sale/Audi/Q7/Brighton+MA-02135?zip=02135&marketExtension=true&startYear=1981&endYear=2019&makeCodeList=AUDI&searchRadius=50&modelCodeList=Q7&sortBy=relevance&numRecords=100&firstRecord=0'
url='https://www.autotrader.com/cars-for-sale/MINI/Cooper/Brighton+MA-02135?zip=02135&marketExtension=true&startYear=1981&endYear=2019&makeCodeList=MINI&searchRadius=50&modelCodeList=COOPER&sortBy=relevance&numRecords=100&firstRecord=0'
headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
         'Accept-Encoding': 'gzip, deflate, br',
         'Accept-Language': 'zh,en-US;q=0.9,en;q=0.8,zh-TW;q=0.7',
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
html=requests.get(url,headers=headers)   
bsObj=BeautifulSoup(html.text,"lxml")
"""
#Getting cars' number in zipcode
num=bsObj.find('button',{'id':"filter-results-button"}).get_text()
cars_num=num.split(' ')[2]
"""
price=bsObj.findAll("div",{"class":"text-gray-base text-bold text-size-500"})   
content=bsObj.findAll("h2",{"class":"text-size-500 link-unstyled text-bold"})
mileage=bsObj.findAll("span",{'class':"text-bold"})
mileages=[]   
for i in list(mileage):
    if str(i)[23]=='>':
        mileages.append(str(i)[24:36])
        
m=mileages
for n in range(10):
    for i in m:   
        if '$' in i:
            #print(i)
            mileages.remove(i)

car=[]
c=-1   
#range改抓取个数
for i in range(100):
    c+=1
    if str(content[i])[50]=='U' or str(content[i])[50]=='C' :        
        car.append(str(content[i])[50:74]+' '+str(price[i])[71:78]+' '+str(mileages[c]))
    elif str(content[i])[50]=='N':
        c-=1
        if str(price[i])[71]=='$':
            car.append(str(content[i])[50:74]+' '+str(price[i])[71:78])
        elif str(price[i])[71]=='<':
            car.append(str(content[i])[50:74]+' '+str(price[i])[166:174])
car.sort()
print(car)


"""
a,b=0,0
for i in str(content[0]):
   if i!= '>':
       a+=1
   else:
       break

for i in str(content[0]):
   if i!= '/':
       b+=1
   else:
       break
       
a,b=0,0
for i in str(price[0]):
   if i!= '>':
       a+=1
   else:
       break

for i in str(price[0]):
   if i!= '/':
       b+=1
   else:
       break
print(a,b)

a,b=0,0
for i in str(mileage[9]):
   if i!= '>':
       a+=1
   else:
       break

for i in str(mileage[9]):
   if i!= '/':
       b+=1
   else:
       break
print(a,b)
""" 
#计算2016年份二手车差价和英里数
mon,mil,money,mileage=[],[],[],[]
a=-1
for i in car:
    a+=1
    if '2016' in i and 'xDrive3' in i:
        mon.append(car[a][25:33])
        mil.append(car[a][33:40])
for i in range(len(mon)):
    mon1=mon[i].replace(',','')
    mon1=mon1.replace('$','')
    money.append(mon1)
    mil1=mil[i].replace(',','')
    mileage.append(mil1)    


for i in range(1,len(money)):
    print('Money:'+str(int(money[i-1])-int(money[i])))
    print('Mileages:'+str(int(mileage[i-1])-int(mileage[i])))
    print('')
            


