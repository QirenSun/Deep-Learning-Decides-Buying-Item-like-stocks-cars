# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:30:16 2019

@author: epinsky
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

output_dir = r'C:\Users\epinsky\bu\python\data_science_with_Python\plots'


data = pd.DataFrame(
        {'id': [ 1,2,3,4,5,6,7,8],
        'Label': ['green', 'green', 'green', 'green', 'red', 'red', 'red', 'red'],
        'Height': [5, 5.5, 5.33, 5.75, 6.00, 5.92,  5.58, 5.92],
        'Weight': [100, 150, 130, 150, 180, 190, 170, 165], 
        'Foot': [6, 8, 7, 9, 13, 11, 12, 10]},
         columns = ['id', 'Height', 'Weight', 'Foot', 'Label']
        )


centroids = {1: np.array([5,7]), 2: np.array([5.5, 9])}

fig = plt.figure(figsize=(5, 5))   
plt.scatter(data['Height'], data['Foot'], color=data['Label'], s =100)

for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.1
    y_text = data['Foot'].iloc[i] + 0.3
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)


colmap = {1: 'blue', 2: 'magenta'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], alpha=0.5, s =300, edgecolor='k')
plt.xlim(4, 7)
plt.ylim(5, 15)
plt.xlabel('Height')
plt.ylabel('Foot')
plt.title("initial centroids")
plt.savefig(os.path.join(output_dir,'kmeans_simple_example_initial_centroids.pdf'))
plt.show()

# assignment stage

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['Height'] - centroids[i][0]) ** 2
                + (df['Foot'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(data, centroids)
print(df)

fig = plt.figure(figsize=(5, 5))

for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.1
    y_text = data['Foot'].iloc[i] + 0.3
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)
    
plt.scatter(df['Height'], df['Foot'], color=df['color'], s=100)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], s=300, alpha=0.5, edgecolor='k')
    
plt.xlim(4,7)
plt.ylim(5, 15)
plt.xlabel('Height')
plt.ylabel('Foot')
plt.title('initial assignment')
plt.savefig(os.path.join(output_dir,'kmeans_simple_example_initial_assignment.pdf'))
plt.show()




# update stage

import copy

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['Height'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['Foot'])
    return k

centroids = update(centroids)
    
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()


for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.1
    y_text = data['Foot'].iloc[i] + 0.3
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=10)

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i],s=300)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=0.1, head_length=0.3, 
             fc=colmap[i], ec=colmap[i])

plt.scatter(data['Height'], df['Weight'], color=df['color'],s=100)

plt.title('updating centroids')
plt.xlim(4,7)
plt.ylim(5, 15)
plt.xlabel('Height')
plt.ylabel('Foot')
plt.savefig(os.path.join(output_dir,'kmeans_simple_example_updating_centroids.pdf'))
plt.show()


# repeat assignment stage
df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(5, 5))


plt.scatter(df['Height'], df['Foot'], color=df['color'],s=100)
for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.1
    y_text = data['Foot'].iloc[i] + 0.3
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], s=300)

plt.xlim(4,7)
plt.ylim(5, 15)
plt.xlabel('Height')
plt.ylabel('Foot')
plt.title('New Assignment')
plt.savefig(os.path.join(output_dir,'kmeans_simple_example_new_assignment.pdf'))
plt.show()


# run additional iterations
# Continue until all assigned categories don't change any more
count = 0
while True:
    count = count + 1
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break
print('converged after ' + str(count) + ' additional iterations')



fig = plt.figure(figsize=(5, 5))
plt.scatter(df['Height'], df['Foot'], color=df['color'], s=100)

for i in range(len(data)):
    x_text = data['Height'].iloc[i] + 0.1
    y_text = data['Foot'].iloc[i] + 0.3
    id_text = data['id'].iloc[i]
    plt.text(x_text, y_text, str(id_text), fontsize=14)

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], s=300)
plt.xlim(4,7)
plt.ylim(5, 15)
plt.xlabel('Height')
plt.ylabel('Foot')
plt.title('Final Assignment')
plt.savefig(os.path.join(output_dir,'kmeans_simple_example_final_assignment.pdf'))
plt.show()

