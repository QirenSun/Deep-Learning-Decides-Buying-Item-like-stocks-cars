# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 11:19:06 2019

@author: Administrator
"""
import os
import pandas as pd
import math

input_dir = 'C:\\Users\\Administrator\\Desktop\\Python_data'
output_file = os.path.join(input_dir, 'BreadBasket_DMS_output.csv')


with open(output_file) as f:
    lines = f.read().splitlines()
bb=pd.read_csv(output_file)

# What is busiest
#Day of the week
def  day_busy1():
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
                    day_tran.append((day_list[q+1]-day_list[q],bb['Weekday'][day_list[q]]))
                day_tran.sort(reverse=True)
                day_busiest=day_tran[0]
                day_busy.append(day_busiest)
            if m-n==11:
                for q in range(n+7,m+1):
                    day_tran.append((day_list[q+1]-day_list[q],bb['Weekday'][day_list[q]]))
                day_tran.sort(reverse=True)
                day_busiest=day_tran[0]
                day_busy.append(day_busiest)
    return day_busy      

day_busy1()
          
#Time of the Day      
def hour_busy1():
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
            hour_tran.append((hour_list[i+1]-hour_list[i],bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
        else:
            hour_tran.append((hour_list[i+1]-hour_list[i],bb['Weekday'][hour_list[i]],bb['Hour'][hour_list[i]]))
    return hour_busy
hour_busy1()


            
#Period of the day
def per_busy1():
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
            per_tran.append((per_list[i+1]-per_list[i],bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
        else:
            per_tran.append((per_list[i+1]-per_list[i],bb['Weekday'][per_list[i]],bb['Period'][per_list[i]]))
    return per_busy
per_busy1()                
            
#What is the most profitable time
#Day of the week
def day_profit():
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
    return day_busy
day_profit()                

#Time of the Day 
def hour_profit():     
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
    return hour_busy
hour_profit()            

#Period of the day
def period_profit():
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
    
    return per_busy
period_profit()                            

#popular         
def popular():
    max_po=[bb['Item'].value_counts().keys()[0]]
    for i in range(1,50):
        if bb['Item'].value_counts().get(i+1)==bb['Item'].value_counts().get(1):
            max_po.append(bb['Item'].value_counts().keys()[i+1])
    
    print('The most popular item: '+''.join(max_po))
    
    min_po=[bb['Item'].value_counts().keys()[-1]]
    
    for i in range(1,50):
        if bb['Item'].value_counts().get(-i-1)==bb['Item'].value_counts().get(-1):
            min_po.append(bb['Item'].value_counts().keys()[-i-1])
    
    
    print('The least popular item: '+', '.join(min_po))
    return

popular()

        
#'barista' each day of week
def barista_num():
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
    
    bar=[]
    bar.append((math.ceil(day_list[0]/60),bb['Weekday'][0]))
    for i in range(len(day_list)-1):           
        day_tran=day_list[i+1]-day_list[i]
        bar.append((math.ceil(day_tran/60),bb['Weekday'][day_list[i]]))
    bar.append((math.ceil((len(bb)-day_list[-1])/60),bb['Weekday'][0]))
    return bar
        
barista_num()

# Combination of 2 items
def combination_2_items():   
    item=[]
    item_com=[]
    for i in range(1,len(bb)-1):
        if bb['Transaction'][i]==bb['Transaction'][i+1]:
            item.append(bb['Item'][i])
        else:
            if bb['Transaction'][i]==bb['Transaction'][i-1]:
                item.append(bb['Item'][i])
                item_com.append(list(set(item)))
                item=[]
    item=[]            
    for i in range(len(item_com)):
        if len(item_com[i])>=2:
            item.append(','.join(item_com[i]))         
    
    item_set=set(item)
    item_count={}
    for i in list(item_set):
        item_count.update({item.count(i):i})
    item_best=sorted(item_count.items(),reverse=True)[0]
    item_least=sorted(item_count.items(),reverse=True)[-3]
        
    print('The most popular combination of 2 items: '+item_best[1])
    print('The least popular combination of 2 items: '+item_least[1])
    return 

combination_2_items()
            
#Food & Drink
def food_drink():
    item_set=set(bb['Item'])
    
    food=['Alfajores','Bacon','Baguette','Bare Popcorn','Bread','Bread Pudding','Brioche and salami','Brownie','Cake',
    'Caramel bites','Cherry me Dried fruit','Chicken Stew','Chicken sand','Chocolates', 'Cookies', 'Crepes','Crisps',
    'Duck egg', 'Eggs', 'Empanadas','Focaccia', 'Frittata', 'Gingerbread syrup', 'Granola', 'Honey', 'Jam', 'Muesli',
    'Kids biscuit', 'Muffin', 'Pastry', 'Polenta', 'Raspberry shortbread sandwich', 'Raw bars', 'Salad', 'Sandwich',
    'Scone',  'Spanish Brunch', 'Tacos/Fajita', 'Toast', 'Truffles']
    print('Food: '+str(food)+'\n')       
            
    drink=['Coffee','Coffee granules ','Coke','Hot chocolate', 'Juice', 'Lemon and coconut', 'Mineral water', 'Soup','Smoothies',
    'Tea']
    print('Drink: '+str(drink)+'\n')
    unknown=[]
    for i in item_set:
        if i not in food and i not in drink:
            unknown.append(i)
    print('Unknown: '+str(unknown)+'\n')
    
    f_list=[]
    f_price=0
    f_count=0
    all_f_price=0
    for f in food:
        for i in range(len(bb)):
            if f==bb['Item'][i]:
                all_f_price+=bb['Item_Price'][i]
                
                if f not in set(f_list):             
                    f_price+=bb['Item_Price'][i]
                    f_count+=1
                f_list.append(bb['Item'][i])
    f_avg=f_price/f_count       
    print('Foods average price:'+format(f_avg,'0.2f'))
    print('Foods total price:'+format(all_f_price,'0.2f'))
    
    
    
    d_list=[]
    d_price=0
    d_count=0
    all_d_price=0
    for d in drink:
        for i in range(len(bb)):
            if d==bb['Item'][i]:
                all_d_price+=bb['Item_Price'][i]
                
                if d not in set(d_list):             
                    d_price+=bb['Item_Price'][i]
                    d_count+=1
                d_list.append(bb['Item'][i])
    d_avg=d_price/d_count    
    print('Drinks average price:'+format(d_avg,'0.2f'))
    print('Drinks total price:'+format(all_d_price,'0.2f'))
    return   
# It will spend some time. Don't stop it, please.
food_drink()

#Group size
def group_size():
    drink=['Coffee','Coffee granules ','Coke','Hot chocolate', 'Juice', 'Lemon and coconut', 'Mineral water', 'Soup','Smoothies',
    'Tea']
    count,tran_count=0,[]
    for i in range(1,len(bb)-1):
        if bb['Transaction'][i]==bb['Transaction'][i+1]:
            if bb['Item'][i] in drink:
                count+=1
        
        elif bb['Transaction'][i-1]==bb['Transaction'][i]:           
            if bb['Item'][i] in drink:
                count+=1
            tran_count.append(count)
            count=0    
    
    cou={}
    for i in set(tran_count):
        n=tran_count.count(i)    
        cou.update({i:n})
    
    group='Group size:'+  str( cou  )  
    return group

group_size()












            
            
            
            
            