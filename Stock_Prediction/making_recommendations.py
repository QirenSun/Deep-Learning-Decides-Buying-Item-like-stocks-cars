# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:07:21 2019

@author: Administrator
"""
import numpy as np
critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 3.5},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
'The Night Listener': 4.5, 'Superman Returns': 4.0,
'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 2.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

#Euclidean Distance Score
from math import sqrt
def distance(prefs,p1,p2):
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item]=1
            
    if len(si)==0:
        return 0
    sum_of_squares=sum([pow(prefs[p1][item]-prefs[p2][item],2) 
    for item in prefs[p1] if item in prefs[p2]])
    return 1/(1+sqrt(sum_of_squares))

distance(critics,'Toby','Lisa Rose')

#Pearson Correlation Score
def sim_pearson(prefs,p1,p2):
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item]=1
    # Find the number of elements
    n=len(si)
    # if they are no ratings in common, return 0
    if n==0: return 0
    # Add up all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    # Sum up the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
    # Sum up the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    # Calculate Pearson score
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    r=num/den
    return r

sim_pearson(critics,'Gene Seymour','Lisa Rose')

def topMatches(prefs,person,n=5,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other)
    for other in prefs if other!=person]
    # Sort the list so the highest scores appear at the top
    scores.sort()
    scores.reverse()
    return scores[0:n]

topMatches(critics,'Toby',n=8)

def getRecommendations(prefs,person,similarity=sim_pearson):
    scores=[(similarity(prefs,person,other),other)
    for other in prefs if other!=person]

    scores.sort()
    scores.reverse()
    
    total,results=0,{}
    a,b,num=[],[],0
    for i in list(prefs.values()):
        a.append(list(i.keys()))
    for i in a:
        for l in i:
            b.append(l)
    all_m=set(b)
    for m in all_m:
        for p2 in scores[:5]:
            if m in prefs[p2[1]]:
                total+=prefs[p2[1]][m]*p2[0]
                num+=p2[0]                
        results.update({total/num:m})
        total=0
        num=0
    return results 

getRecommendations(critics,'Toby',similarity=sim_pearson)

def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
    # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)
    # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
        # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
        # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
        # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

getRecommendations(critics,'Toby',similarity=sim_pearson)















