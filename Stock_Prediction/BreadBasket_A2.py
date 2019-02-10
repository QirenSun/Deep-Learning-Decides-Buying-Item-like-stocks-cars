# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 11:19:06 2019

@author: Administrator
"""
import os
import pandas as pd

input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')


with open(output_file) as f:
    lines = f.read().splitlines()
bb=pd.read_csv(output_file)

# What is busiest
#Day of the week
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)

n,m,day_busy=0,0,[]

for i in range(len(day_list)):
    if bb['Weekday'][day_list[i]]=='Sunday':  
        n=i
        
    elif bb['Weekday'][day_list[i]]=='Saturday':
        m=i
        day_tran=[]
        if m-n<=6:
            for q in range(n,m+1):
                day_tran.append((bb['Transaction'][day_list[q]:day_list[q+1]].sum(),bb['Weekday'][day_list[q]]))
            day_tran.sort(reverse=True)
            day_busiest=day_tran[0]
            day_busy.append(day_busiest)
        if m-n==11:
            for q in range(n+7,m+1):
                day_tran.append((bb['Transaction'][day_list[q]:day_list[q+1]].sum(),bb['Weekday'][day_list[q]]))
            day_tran.sort(reverse=True)
            day_busiest=day_tran[0]
            day_busy.append(day_busiest)
                
#Time of the Day      
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)            
            
same_hour=0
hour=[]
all_hour=0
for i in range(len(bb)-1):
    if bb['Hour'][i]==bb['Hour'][i+1]:
        same_hour+=1
        all_hour+=1        
    else:
        same_hour+=1
        all_hour+=1
        hour.append(same_hour)
        same_hour=0
hour_line=0
hour_list=[]        
for i in range(len(hour)):
    hour_line+=hour[i]
    hour_list.append(hour_line)            

hour_busy=[]
hour_tran=[]
for i in range(len(hour_list)-1):    
    if hour_list[i] in day_list:
        hour_tran.sort(reverse=True)
        hour_buisest=hour_tran[0]
        hour_busy.append(hour_buisest)
        hour_tran=[]
        hour_tran.append((bb['Transaction'][hour_list[i]:hour_list[i+1]].sum(),bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
    else:
        hour_tran.append((bb['Transaction'][hour_list[i]:hour_list[i+1]].sum(),bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
        
#Period of the day
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)            
            
same_per=0
per=[]
all_per=0
for i in range(len(bb)-1):
    if bb['Period'][i]==bb['Period'][i+1]:
        same_per+=1
        all_per+=1        
    else:
        same_per+=1
        all_per+=1
        per.append(all_per)
        all_per=0
per_line=0
per_list=[]        
for i in range(len(per)):
    per_line+=per[i]
    per_list.append(per_line)            

per_busy=[]
per_tran=[]
for i in range(len(per_list)-1):    
    if per_list[i] in day_list:
        per_tran.sort(reverse=True)
        per_buisest=per_tran[0]
        per_busy.append(per_buisest)
        per_tran=[]
        per_tran.append((bb['Transaction'][per_list[i]:per_list[i+1]].sum(),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
    else:
        per_tran.append((bb['Transaction'][per_list[i]:per_list[i+1]].sum(),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
            
            
#What is the most profitable time
#Day of the week
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)

n,m,day_busy=0,0,[]

for i in range(len(day_list)):
    if bb['Weekday'][day_list[i]]=='Sunday':  
        n=i
        
    elif bb['Weekday'][day_list[i]]=='Saturday':
        m=i
        day_tran=[]
        if m-n<=6:
            for q in range(n,m+1):
                day_tran.append((float(format(bb['Item_Price'][day_list[q]:day_list[q+1]].sum(),'0.2f')),bb['Weekday'][day_list[q]]))
            day_tran.sort(reverse=True)
            day_busiest=day_tran[0]
            day_busy.append(day_busiest)
        if m-n==11:
            for q in range(n+7,m+1):
                day_tran.append((float(format(bb['Item_Price'][day_list[q]:day_list[q+1]].sum(),'0.2f')),bb['Weekday'][day_list[q]]))
            day_tran.sort(reverse=True)
            day_busiest=day_tran[0]
            day_busy.append(day_busiest)
                
#Time of the Day      
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)            
            
same_hour=0
hour=[]
all_hour=0
for i in range(len(bb)-1):
    if bb['Hour'][i]==bb['Hour'][i+1]:
        same_hour+=1
        all_hour+=1        
    else:
        same_hour+=1
        all_hour+=1
        hour.append(same_hour)
        same_hour=0
hour_line=0
hour_list=[]        
for i in range(len(hour)):
    hour_line+=hour[i]
    hour_list.append(hour_line)            

hour_busy=[]
hour_tran=[]
for i in range(len(hour_list)-1):    
    if hour_list[i] in day_list:
        hour_tran.sort(reverse=True)
        hour_buisest=hour_tran[0]
        hour_busy.append(hour_buisest)
        hour_tran=[]
        hour_tran.append((float(format(bb['Item_Price'][hour_list[i]:hour_list[i+1]].sum(),'0.2f')),bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
    else:
        hour_tran.append((float(format(bb['Item_Price'][hour_list[i]:hour_list[i+1]].sum(),'0.2f')),bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
        
#Period of the day
same_day=0
day=[]
all_day=0
for i in range(len(bb)-1):
    if bb['Day'][i]==bb['Day'][i+1]:
        same_day+=1
        all_day+=1        
    else:
        same_day+=1
        all_day+=1
        day.append(same_day)
        same_day=0
day_line=0
day_list=[]        
for i in range(len(day)):
    day_line+=day[i]
    day_list.append(day_line)            
            
same_per=0
per=[]
all_per=0
for i in range(len(bb)-1):
    if bb['Period'][i]==bb['Period'][i+1]:
        same_per+=1
        all_per+=1        
    else:
        same_per+=1
        all_per+=1
        per.append(all_per)
        all_per=0
per_line=0
per_list=[]        
for i in range(len(per)):
    per_line+=per[i]
    per_list.append(per_line)            

per_busy=[]
per_tran=[]
for i in range(len(per_list)-1):    
    if per_list[i] in day_list:
        per_tran.sort(reverse=True)
        per_buisest=per_tran[0]
        per_busy.append(per_buisest)
        per_tran=[]
        per_tran.append((float(format(bb['Item_Price'][per_list[i]:per_list[i+1]].sum(),'0.2f')),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
    elif i==0:
        per_tran.append((float(format(bb['Item_Price'][:per_list[i]].sum(),'0.2f')),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
        per_tran.append((float(format(bb['Item_Price'][per_list[i]:per_list[i+1]].sum(),'0.2f')),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
    
    else:
        per_tran.append((float(format(bb['Item_Price'][per_list[i]:per_list[i+1]].sum(),'0.2f')),bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
                        
            
            
            
            
            
            
            
            