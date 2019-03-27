import pandas as pd
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
    if dm['rankPosition'][i] >25:
        dm['rankPosition'][i]=0
    else:
        dm['rankPosition'][i]=1
        
dm.rename({'likeNum':'CurrentOpusNum','opusNum':'CurrentOpusNum'},inplace=True)
       
dm.to_csv('data.csv',mode='w',encoding='utf-8-sig')

