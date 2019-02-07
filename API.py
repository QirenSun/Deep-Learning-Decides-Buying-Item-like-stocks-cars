#API
import os   
import pandas as pd
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import json

#日榜
#url='https://api.newrank.cn/api/sync/douyin/rank/account/day'
#data='uid=76725372134&from=2018-12-21&to=2018-12-28&page=1&size=20'
#data=json.dumps({'uid':'76725372134', 'from':'2019-01-01','to':'2019-01-10','page':'1','size':'20'}) 
'''
#获取指定新榜类别的周榜TOP50
url1='https://api.newrank.cn/api/sync/douyin/rank/type/week/top50'
data1={'type':'娱乐'.encode('utf-8'),
      'date':'2019-01-13',
      'page':'1',
      'size':'20'}
'''
headers={'Key':'9aa7255d65564d88b906e7ca2',
         'Content-Type':'application/x-www-form-urlencoded;charset=utf-8',
         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
         'Accept-Encoding': 'gzip, deflate, br',
         'Accept-Language': 'zh,en-US;q=0.9,en;q=0.8,zh-TW;q=0.7',
         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}

#获取指定新榜类别的日榜TOP50
class1=['娱乐','才艺','搞笑','游戏','体育','企业','科技','时尚','社会','二次元']
url1='https://api.newrank.cn/api/sync/douyin/rank/type/day/top50'
content1=[]
for typ in class1:
    data1={'type':typ.encode('utf-8'),
           'date':'2019-01-13',
           'page':'1',
           'size':'50'}
    html1=requests.post(url1,headers=headers,data=data1,verify=False)   
    dataset1=html1.json()
    
    title1=list(list(dataset1.values())[0][0].keys())

    for i in range(len(list(dataset1.values())[0])):
        content1.append(list(list(dataset1.values())[0][i].values()))
#抖音号粉丝、作品等统计信息
content,content2=[],[]

p=0
for i in range(445,len(content1)):        
    account=content1[i][1]
    data={'account':account}
    url='https://api.newrank.cn/api/sync/douyin/account/stats'
    html=requests.post(url,headers=headers,data=data,verify=False)   
    dataset=html.json()
    
    
    for m in range(len(list(dataset.values())[0])):
        content.append(list(list(dataset.values())[0][m].values()))        
        content[p].pop(4)
        content[p].pop(-4)
        content[p].pop(-3)
        p+=1
        
    content2.append(content1[i]+content[i])

title=list(list(dataset.values())[0][0].keys())
title.pop(4)
title.pop(-4)
title.pop(-3)
title2=title1+title
content2.insert(0,title2)    





#python2可以用file替代open
with open("live1.csv","w",newline='',encoding='utf-8-sig') as csvfile: 
    writer = csv.writer(csvfile)

    #先写入columns_name
    #writer.writerow(["index","a_name","b_name"])
    #写入多行用writerows
    writer.writerows(content2)

with open("live.csv","r") as csvfile:
    reader = csv.reader(csvfile)
    #这里不需要readlines
    for line in reader:
        print (line)


