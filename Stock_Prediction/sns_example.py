# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:23:27 2019

@author: epinsky
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



input_dir = r'C:\Users\epinsky\bu\python\data_science_with_Python\datasets'
tips = sns.load_dataset('tips')

output_file = os.path.join(input_dir,'tips.csv')
#tips.to_csv(output_file, index=False)

x = tips.copy()
x['percent_tip'] = 100.0 * x['tip']/x['total_bill']

y = x.groupby(['smoker'])['percent_tip'].mean()
z = y.reset_index(['smoker'])

x = tips.copy()
x['percent_tip'] = 100.0 * x['tip']/x['total_bill']
y_gender_tip = x.groupby(['sex'])['percent_tip'].mean()
z_gender_tip = y_gender_tip.reset_index(['sex'])

x = tips.copy()
x['percent_tip'] = 100.0 * x['tip']/x['total_bill']
y_gender_smoke_tip = x.groupby(['sex','smoker'])['percent_tip'].mean()
z_gender_smoke_tip = y_gender_smoke_tip.reset_index(['sex', 'smoker'])

"""
fig = plt.figure()
axes1 = fig.add_subplot(1,1,1)
axes1.hist(tips['total_bill'], bins = 30, color='red')
axes1.set_title('Histogram of Bill')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('Total Bill')
fig.show()


fig = plt.figure()
axes1 = fig.add_subplot(1,1,1)
axes1.hist(100.* tips['tip']/tips['total_bill'],
           bins = 20, color='green')
axes1.set_title('Histogram of tips')
axes1.set_xlabel('Frequency')
axes1.set_ylabel('tip as a percent')
fig.show()


scatter_plot = plt.figure()
axes1 = scatter_plot.add_subplot(1,1,1)
axes1.scatter(tips['total_bill'], tips['tip'])
axes1.set_title('Bill and Tips')
axes1.set_xlabel('Total Bill')
axes1.set_ylabel('Tip')
scatter_plot.show()


# seaborn

hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'])
ax.set_title('Bill with Density')
plt.show()


# set kernel density estimation to false
hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde=False,
                  color='magenta')
ax.set_title('Bill w/o Density')
plt.show()


# display density only
hist, ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist=False,
                  color='black')
ax.set_title('Bill with Density')
ax.set_xlabel('Total Bill')
ax.set_ylabel('probability')
plt.show()


#count plots
count, ax = plt.subplots()
ax = sns.countplot('day', data=tips)
ax.set_title('count of days')
ax.set_xlabel('day of the week')
ax.set_ylabel('Frequency')
plt.show()


# scatter plot and regression line
scatter, ax = plt.subplots()
ax = sns.regplot(x='total_bill', y='tip',
                 data=tips, color='green')
ax.set_title('Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()


# ignore regression
scatter, ax = plt.subplots()
ax = sns.regplot(x='total_bill', y='tip',
                 data=tips, 
                 color='green',fit_reg=True)
ax.set_title('Bill and Tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
plt.show()

# implot creates a figure (calls regplot)

fig = sns.lmplot(x='total_bill', y='tip',
                 data=tips) 
plt.show()



# density plot for two variables

kde, ax = plt.subplots()
ax = sns.kdeplot(data=tips['total_bill'],
                 data2=tips['tip'], shade=True)
ax.set_title('Kernel Density Estimation for Bill and Tips')
ax.set_xlabel('Total Bill')
ax.set_ylabel('tip')
plt.show()

kde_joint = sns.jointplot(x='total_bill', y='tip',
                           data=tips, kind='kde')

# bar plots show multiple variables (with means)

bar, ax = plt.subplots()
ax = sns.barplot(x='time',y='total_bill',
                 data=tips)
ax.set_title('average bill per days')
ax.set_xlabel('time of day')
ax.set_ylabel('average total bill')
plt.show()


# boxplots
# show mean, max, median, quartiles

box, ax = plt.subplots()
ax = sns.boxplot(x='time',y='total_bill',
                 data=tips)
ax.set_title('boxplot of bill per days')
ax.set_xlabel('time of day')
ax.set_ylabel('average total bill')
plt.show()

# violin data - show details on box plots
violin, ax = plt.subplots()
ax = sns.violinplot(x='time',y='total_bill',
                 data=tips)
ax.set_title('violin plot of bill per days')
ax.set_xlabel('time of day')
ax.set_ylabel('average total bill')
plt.show()



# pairwise relationships
fig = sns.pairplot(tips)
plt.show()


# remove duplicates
pair_grid = sns.PairGrid(tips)
pair_grid = pair_grid.map_upper(sns.regplot)
pair_grid = pair_grid.map_lower(sns.kdeplot)
pair_grid = pair_grid.map_diag(sns.distplot,rug=True)
plt.show()



# colors - can pass hue 

violin, ax = plt.subplots()
ax = sns.violinplot(x='time',y='total_bill',
                 hue='sex', data=tips, split=True)
ax.set_title('violin plot of bill per days')
ax.set_xlabel('time of day')
ax.set_ylabel('average total bill')
plt.show()

"""
fig = sns.lmplot(x='total_bill', y='tip',
                 hue='sex', data=tips, fit_reg=False) 
plt.show()

"""
fig=sns.pairplot(tips,hue='sex')

"""

