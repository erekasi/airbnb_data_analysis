#!/usr/bin/env python
# coding: utf-8

# # Project 1
# DSND Term 2 <br> <br>
# 09/13/2020
# ***

# ### Import libraries

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

#from seaborn import regplot
#from statsmodels.graphics.gofplots import qqplot
#import statsmodels.api as sm
#from scipy.stats import t as tstat

#import seaborn as sns

import time
import os

#import matplotlib.patches as mpatches
#from sklearn.preprocessing import Imputer, LabelEncoder, StandardScaler
#from sklearn.preprocessing import OneHotEncoder #finally not used
#from sklearn.decomposition import PCA
#from sklearn.decomposition import FastICA #finally not used
#from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)

# to produce visualizations in notebook:
get_ipython().run_line_magic('matplotlib', 'inline')

os.chdir('/Users/erekasi/Documents/DA/002_DSND_Term-2/Project 1')
start_full = time.time()


# ### Loading the data

# In[2]:


# Seattle datasets
dfs_cal=pd.read_csv('calendar_seattle.csv')
dfs_lis=pd.read_csv('listings_seattle.csv')
dfs_rev=pd.read_csv('reviews_seattle.csv')


# In[3]:


# Boston datasets
dfb_cal=pd.read_csv('calendar_boston.csv')
dfb_lis=pd.read_csv('listings_boston.csv')
dfb_rev=pd.read_csv('reviews_boston.csv')


# ### Data understanding

# #### Size of the datasets

# In[4]:


#Size of the datasets
print('Shape of calendar data on Seattle and Boston(rows, columns): ', dfs_cal.shape, ' & ', dfb_cal.shape)
print('Shape of listings data on Seattle and Boston(rows, columns): ', dfs_lis.shape, ' & ', dfb_lis.shape)
print('Shape of reviews data on Seattle and Boston(rows, columns): ', dfs_rev.shape, ' & ', dfb_rev.shape)


# > __Finding__:<br>
# Boston listings contains 3 more columns than Seattle's
# 

# In[5]:


#Check the difference in columns between the Boston and the Seattle listings datasets

dfs_lis_cols = set(dfs_lis.columns)
#print(dfs_lis_cols)

dfb_lis_cols = set(dfb_lis.columns)
df_lis_dif = dfb_lis_cols - dfs_lis_cols
print('Features only covered in the Boston dataset: ', df_lis_dif)


# In[6]:


# General information
dfb_rev.info()


# In[7]:


#To see the full list of the listings datasets' features
dfb_lis.info()


# In[8]:


dfs_rev.head()


# In[9]:


dfb_rev.head(3)


# In[10]:


dfs_lis.head(3)


# In[11]:


dfs_cal.head()


# In[12]:


dfb_cal.head()


# ### Data preparation

# In[13]:


#Drop columns from Boston listings which are missing from Seattle listings
dfb_lis = dfb_lis.drop(columns=['interaction', 'house_rules', 'access'])


# In[14]:


dfb_lis.shape


# In[15]:


#Rename some columns to enable merging
dfs_lis = dfs_lis.rename(columns={'id':'listing_id'})
dfb_lis = dfb_lis.rename(columns={'id':'listing_id'})


# In[16]:


#Merge listings and calendar datasets
start = time.time()

dfs = pd.merge(dfs_lis, dfs_cal, how = 'outer')
dfb = pd.merge(dfb_lis, dfb_cal, how = 'outer')

end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[22]:


dfs.shape[0]


# In[ ]:


dfb.shape[0]


# In[17]:


#Add a city column to differentiate records by city in the concatenated dataset
dfs['seattle'] = 1
dfb['seattle'] = 0


# In[18]:


#Join Seattle and Boston datasets

start = time.time()

df = pd.concat([dfs, dfb])
end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[23]:


print('Proportion of Seattle data in the complete dataset:')
dfs.shape[0]/df.shape[0]


# In[24]:


print('Proportion of Boston data in the complete dataset:')
dfb.shape[0]/df.shape[0]


# In[20]:


df.columns.sort_values()


# #### Assessing missing data

# In[25]:


dfs_mis = pd.DataFrame(np.sum(dfs.isnull())/dfs.shape[0])


# In[26]:


dfs_mis.head(3)


# In[27]:


dfb_mis = pd.DataFrame(np.sum(dfb.isnull())/dfb.shape[0])


# In[28]:


dfb_mis.head(3)


# In[29]:


df_mis = pd.merge(dfb_mis, dfs_mis, left_index=True, right_index=True)
df_mis.columns = ['Boston', 'Seattle']
df_mis['Difference'] = dfb_mis - dfs_mis


# In[30]:


df_mis.shape


# In[31]:


df_mis.head(3)


# In[32]:


#Compare difference in the proportion of missing values per feature by city

df_mis.style.bar(subset=['Difference'], width=50, align='mid', color=['#d65f5f', '#5fba7d'])
#fig1=plt.gcf()
#plt.savefig('difference-in-missing.png')


# > __Comment:__<br>
#     >There are hardly any features of which no values are missing. Exceptions are: listing_id, license and seattle. <br>
#     >The extent of missing values is significantly different for the two cities in case of the following features:
#         - neighbourhood_group_cleansed
#         - has_availability
#         - jurisdiction_names

# In[33]:


#Create a dataframe of the number of missing values per column
start = time.time()

mis_df = pd.DataFrame(np.sum(df.isnull())/df.shape[0])
mis_df.columns = ["missing_proportion"]
mis_df_sort = mis_df.sort_values(by=["missing_proportion"])

###Create feature from the index
mis_df_sort['feature'] = mis_df_sort.index

end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[34]:


mis_df_sort.head(3)


# In[35]:


#Visualizing proportion of missing values per feature
start = time.time()

plt.figure(figsize=(20,5))

x = range(mis_df_sort.shape[0])
y = mis_df_sort['missing_proportion']

col = []

for i in y:
    if i <= .6:
        col.append('darkblue')
    else:
        col.append('lightgray')
        plt.bar(x,y)

plt.bar(x,y,color=col)

ax = plt.subplot()
ax.set_xticks(range(mis_df_sort.shape[0]))
ax.set_xticklabels(mis_df_sort['feature'], rotation = 90)
plt.title('Proportion of missing values by feature')
plt.savefig("Proportion_missing_by_feature.png")

end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[36]:


#Calculate the number of remaining features if dropping features by proportion of missing values


# In[37]:


start = time.time()

#Maximum % missing
threashold = list(range(40,100,5))
remaining_features = []

for i in threashold:
    remaining_features.append(len(df.columns[np.sum(df.isnull())/df.shape[0] < i/100]))
    
threashold_divided = [i/100 for i in threashold]

#Creating dataframe for visualization
df_thr_fea = pd.DataFrame({"threashold":threashold_divided, "number_of_features": remaining_features})

end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[44]:


df_thr_fea


# In[45]:


plt.figure(figsize=(10,5))
plt.plot(df_thr_fea['threashold'], df_thr_fea['number_of_features'])
plt.xlabel("Maximum proportion of missing values")
plt.title("Number of remaining features by proportion of missing values tolerated per feature")
plt.ylabel('No. of remaining features')

plt.savefig("threashold_features.png")
plt.show()


# >__Assumption:__ <br> more information available makes an accommodation more attractive --> higher prices or better reviews

# In[46]:


#Create a feature to count missing values
#df['Missing_values'] = df.apply(lambda x: x.isnull().count(),axis=1) #This approach would require excessive amount of time


# In[47]:


start = time.time()

df['missing_values'] = df.isnull().sum(axis=1)

end = time.time()
t = end - start
print('------------ \n Time spent on task execution: \n ', int(t/60), 'minute(s)', round(t%60,2), 'second(s)')


# In[48]:


plt.hist(df['missing_values'], bins=25)
plt.xlabel('Number of missing values in a record')
plt.ylabel('Number of records')
plt.title('Distribution of missing values per row')
plt.savefig("missing_values_per_row_histogram.png")


# In[43]:


plt.figure(figsize=(50,10))
plt.hist(df['missing_values'][df.seattle == 1], bins=20, alpha = .5, label='Seattle')
plt.hist(df['missing_values'][df.seattle == 0], bins=20, alpha = .5, label='Boston')
plt.xlabel('Number of missing values in a record')
plt.ylabel('Number of records')
plt.legend(loc='upper left')
plt.title('Distribution of missing values per row - grouped by city')
plt.savefig("missing_per_record_by_city.png")


# In[49]:


df.missing_values.describe()


# #### Investigating missing values <br>
# (missing at random, missing completely at random, missing not at random)

# In[50]:


def clean_price(x):
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))    
    return(x)


# In[51]:


df['price'] = df['price'].apply(clean_price).astype('float')


# In[52]:


df.to_csv('merged_price.csv')


# In[53]:


df.plot(x = 'missing_values', y = 'price', kind = 'scatter')
plt.title('Price by number of missing values in a record')
plt.xlabel('Missing values per record')
plt.ylabel('Price')
plt.savefig("price_by_missing_values_per_record.png")


# >__Comment:__<br>
#     Based on a simple visual approach, and disregarding seemingly outlier values, there is no clear correlation between the number of missing values of a record and the corresponding price level. However, there might be more relatively expensive accommodations among the ones with the most missing values.

# In[54]:


df.plot(x = 'missing_values', y = 'number_of_reviews', kind = 'scatter', color = 'mediumvioletred')
plt.title('Number of reviews by number of missing values in a record')
plt.xlabel('Missing values per record')
plt.ylabel('No. of reviews')
plt.savefig("no_of_reviews_by_missing_values_per_record.png")


# >__Comment:__<br>
#     It might occur that accommodations for which less information is available online are less popular, assuming that number of reviews is a valid proxy for popularity.

# In[55]:


df.plot(x = 'missing_values', y = 'review_scores_value', kind = 'scatter', color = 'grey')
plt.title('Review scores value by number of missing values in a record')
plt.xlabel('Missing values per record')
plt.ylabel('Review scores value')
plt.savefig("review_scores_value_by_missing_values_per_record.png")


# >__Comment:__<br>
#     There are seemingly no review scores values available in the dataset for records where are more than cca. 25 missing values in a record.

# In[56]:


df.plot(x = 'missing_values', y = 'reviews_per_month', kind = 'scatter', color = 'green')
plt.title('Reviews per month by number of missing values in a record')
plt.xlabel('Missing values per record')
plt.ylabel('No. of reviews per month')
plt.savefig("no_of_reviews_per_month_by_missing_values_per_record.png")


# >__Comment:__<br>
#     - There is seemingly no data available on reviews per month for records (accommodations) in which more than cca. 30 values are missing.
#     - Interestingly, not those records receive the most review per month which have the least missing values.

# In[57]:


#Drop columns which contain the most missing values
most_missing60 = df.columns[np.sum(df.isnull())/df.shape[0] >= .6]
len(most_missing60)


# In[58]:


df = df.drop(columns = most_missing60)
df.shape


# In[59]:


#Sorted list of columns which remain in the dataframe
df.columns.sort_values()


# In[60]:


#Drop columns which do not add value to the current dataset but potentially contain several missing values
df = df.drop(['street', 'market', 'city', 'state', 'country', 'country_code', 'smart_location'], axis = 1)
df.shape


# ### Data analysis

# In[61]:


df_num = df.select_dtypes(include='number')
df_num.shape


# In[62]:


df_num.columns.sort_values()


# In[63]:


#Drop unnecessary IDs for plotting
df_num_plo = df_num.drop(columns = ['host_id', 'listing_id', 'scrape_id'])
df_num.shape


# #### Heatmaps of correlation matrix w/o missing data

# In[64]:


#Heatmap of correlation matrix with missing values
ax = sns.heatmap(df_num_plo.corr(), cmap=sns.diverging_palette(20, 220, n=200))
plt.title('Heatmap of correlations among numerical variables - with missing values')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# In[65]:


plt.figure(figsize=(20,5))
ax = sns.heatmap(round(df_num_plo.corr(),2), annot = True, cmap='PiYG')
plt.title('Correlation matrix heatmap - with missing values')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.savefig("heatmap_mis.png")


# In[66]:


df_num_plo_fil = df_num_plo.fillna(df_num_plo.mean())


# In[67]:


plt.figure(figsize=(20,5))
ax = sns.heatmap(round(df_num_plo_fil.corr(),2), annot = True, cmap='PiYG')
plt.title('Correlation matrix heatmap - with imputation')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.savefig("heatmap_filled.png")


# In[68]:


#Select numeric features for pairwise investigation
df_num_sel = df_num[['accommodates', 'bathrooms', 'beds', 'availability_30', 'number_of_reviews', 'price', 'seattle']]


# In[69]:


df_num_sel2 = df_num[['accommodates', 'bathrooms', 'beds', 'seattle']]


# In[70]:


df_num_sel2.shape


# In[71]:


df_num_sel2.head(3)


# In[72]:


lf = lambda x: 'Seattle' if x == 1 else 'Boston'


# In[73]:


df_num_sel['city'] = df_num_sel.seattle.apply(lf)


# In[74]:


# sns.pairplot(df_num_sel, hue = "city") #script took excessive amount of time to run


# In[75]:


#Convert most IDs into object type
df[['listing_id', 'scrape_id']] = df[['listing_id', 'scrape_id']].astype('str')


# In[76]:


#Check missing data in the remaining dataset
plt.hist(np.sum(df.isnull())/df.shape[0], range=(0.4,0.6), bins=100)


# #### Drop rows

# In[77]:


#Drop rows with more than 40 missing values


# In[78]:


df = df[df['missing_values']<40]
df.shape


# In[79]:


df.to_csv('halved.csv')


# In[80]:


plt.hist(df.isnull().sum(axis=1), color = 'red')
plt.title('Number of rows by number of missing attributes per row')
plt.xlabel('Number of missing attributes in a row')
plt.ylabel('Number of rows having approximately the same number of missing attributes')


# In[101]:


plt.hist(df[df.seattle == 1].isnull().sum(axis=1), color = 'mediumblue', alpha = .5)
plt.hist(df[df.seattle == 0].isnull().sum(axis=1), color = 'darkorange', alpha = .8)
plt.title('Number of rows by number of missing attributes per row')
plt.xlabel('Number of missing attributes in a row')
plt.ylabel('Number of rows having approximately \n the same number of missing attributes')
plt.legend(['Seattle', 'Boston'])


# #### Handling missing numeric values

# In[82]:


#Assess number of missing values by numeric variable
df.select_dtypes(include='number').isnull().sum()


# In[83]:


col_to_fil = [
    'host_listings_count',
    'host_total_listings_count',
    'bathrooms',
    'bedrooms',
    'beds'
]


# In[84]:


for i in col_to_fil:
    df[i] = df[i].fillna(df[i].mean())


# In[85]:


df.select_dtypes(include='number').isnull().sum()


# In[88]:


room_type_val_b = df[df.seattle == 0].room_type.value_counts()
room_type_val_s = df[df.seattle == 1].room_type.value_counts()


(room_type_val_b/df[df.seattle == 0].shape[0]).plot(kind="bar", color='darkorange', alpha = .8);
(room_type_val_s/df[df.seattle == 1].shape[0]).plot(kind="bar", color='mediumblue', alpha = .5);
plt.legend(['Boston', 'Seattle'])
plt.title("Rooms per type by city");
plt.savefig('rooms_per_type_by_city.png')


# In[89]:


#Number of unique hosts by city
print ('----------------- \n Number of unique hosts by city:')
df.groupby('seattle')['host_id'].nunique()


# In[90]:


df.columns.sort_values()


# In[91]:


#Number of unique locations by city
print ('----------------- \n Number of unique listings by city:')
df.groupby('seattle')['listing_id'].nunique()


# In[94]:


cancellation_pol_b = df[df.seattle == 0].cancellation_policy.value_counts()
cancellation_pol_s = df[df.seattle == 1].cancellation_policy.value_counts()


(cancellation_pol_b/df[df.seattle == 0].shape[0]).plot(kind="bar", color='darkorange', alpha = .8);
(cancellation_pol_s/df[df.seattle == 1].shape[0]).plot(kind="bar", color='mediumblue', alpha = .5);
plt.legend(['Boston', 'Seattle'])
plt.title("Cancellation policy prevalence by city");
plt.savefig('cancellation_policy_by_city.png')


# In[ ]:


df.groupby(df['room_type'])


# In[95]:


#Distribution of number of reviews received
plt.figure(figsize=(20,10))
plt.hist(df['number_of_reviews'][df['seattle'] == 0], color='darkorange', alpha = .8, bins = 50)
plt.hist(df['number_of_reviews'][df['seattle'] == 1], color='mediumblue', alpha = .5, bins = 50)
plt.legend(['Boston', 'Seattle'])
plt.title("Distribution of number of reviews per listing by city");
plt.xlabel('Number of reviews recieved')
plt.ylabel('Number of listings')
plt.savefig('no_of_reviews_by_city.png')


# In[96]:


#Distribution of appartments by number of beds
plt.figure(figsize=(10,5))
plt.hist(df['beds'][df['seattle'] == 0], color='darkorange', alpha = .8, bins = 50)
plt.hist(df['beds'][df['seattle'] == 1], color='mediumblue', alpha = .5, bins = 50)
plt.legend(['Boston', 'Seattle'])
plt.title("Distribution of appartments by number of beds and by city");
plt.xlabel('Number of beds')
plt.ylabel('Number of appartments by bed number')
plt.savefig('no_of_beds_by_city.png')


# In[97]:


df_obj_uni = pd.DataFrame(df.select_dtypes(include='object').nunique())


# In[98]:


object_columns = df.select_dtypes(include='object').columns.sort_values()


# In[99]:


object_columns


# In[100]:


end_full = time.time()

t = end_full - start_full

print('------------ \n Time spent on running the full script: \n ', int(t/60), 'minutes', round(t%60,2), 'seconds')

