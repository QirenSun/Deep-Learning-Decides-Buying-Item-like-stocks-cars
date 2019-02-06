#API
import os   
import pandas as pd
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import json

headers={'Key':'b466b320b9b445e4befd72ebd',
         'Content-Type':'application/x-www-form-urlencoded;charset=utf-8',
         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
         'Accept-Encoding': 'gzip, deflate, br',
         'Accept-Language': 'zh,en-US;q=0.9,en;q=0.8,zh-TW;q=0.7',
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

url='https://api.newrank.cn/api/sync/douyin/rank/account/day'

data='uid=76725372134&from=2018-12-21&to=2018-12-28&page=1&size=20'
#data=json.dumps({'uid':'76725372134', 'from':'2019-01-01','to':'2019-01-10','page':'1','size':'20'}) 
html=requests.post(url,headers=headers,data=data)   
dataset=html.json()
#data=pd.read_json(html.content)

#bsObj=BeautifulSoup(html.text,"lxml")   
#tbl = pd.read_html(bsObj.prettify(),header = 0)[0]  
#data.to_csv(r'1.csv',mode='a',encoding='utf_8_sig',header=1,index=0)
#list(dataset.values())[0][0]
title=list(list(dataset.values())[0][0].keys())
content=[]
for i in range(len(list(dataset.values())[0])):
    content.append(list(list(dataset.values())[0][i].values()))
content.insert(0,title)    



#python2可以用file替代open
with open("live.csv","w") as csvfile: 
    writer = csv.writer(csvfile)

    #先写入columns_name
    #writer.writerow(["index","a_name","b_name"])
    #写入多行用writerows
    writer.writerows(content)

with open("live.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        print (line)


