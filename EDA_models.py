#!/usr/bin/env python
# coding: utf-8

# # Load the relevant IMDb files and TMDB files, and conduct basic EDA and cleaning

# In[1]:


### Import various packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


title_basics = pd.read_csv('title.basics.tsv',sep = '\t') 


# In[3]:


## Get a general idea of what is in the dataset

title_basics.head


# In[4]:


title_basics.columns


# In[5]:


title_basics.iloc[1:5]


# In[6]:


## There are 269,209 tv series in the title_basics dataset. These titles will be used for this project.

title_basics['titleType'].value_counts()


# In[7]:


## Assign the dataframe tv_basics to equal the filtered dataset of title_basics that only include tv series

tv_basics = title_basics[title_basics['titleType'] == 'tvSeries']
len(tv_basics)


# In[8]:


tv_basics.iloc[1:5]


# In[80]:


## Now, import a second dataset: title_ratings

title_ratings = pd.read_csv('title.ratings.tsv',sep = '\t') 


# In[81]:


title_ratings


# In[13]:


## Clean up tv_basics by replacing all '\N' values with "NaN"

tv_basics.replace("\\N", np.nan, inplace=True)


# In[14]:


## Now, drop the NaN values from the tv_basics dataset
tv_basics_cleaned = tv_basics.dropna()


# In[15]:


tv_basics_cleaned


# In[83]:


## Remove duplicate or irrelevant columns 

tv_basics_short = tv_basics_cleaned[['tconst', 'primaryTitle', 'isAdult', 'endYear', 'runtimeMinutes', 'genres']]
tv_basics_short


# In[84]:


## Repeat the previous few steps for the title_ratings dataset, starting with replacing \N values with NaN

title_ratings.replace("\\N", np.nan, inplace=True)


# In[19]:


## Drop NaN values from title_ratings

title_ratings_cleaned = title_ratings.dropna()


# In[20]:


title_ratings_cleaned


# In[21]:


## Now, merge the two cleaned datasets: tv_basics and title_ratings_cleaned

tv_merged = pd.merge(tv_basics_short, title_ratings_cleaned, how="inner", left_on="tconst", right_on="tconst")


# In[22]:


tv_merged


# ## One-hot-encode required columns

# In[23]:


## Get an idea of what the genres column looks like

tv_merged['genres'].value_counts()


# In[24]:


## Split up the "genres" column into multiple genres and then do one-hot-encoding to create numerical columns
## for our upcoming predictive models.

tv_merged['genres_list'] = tv_merged['genres'].str.split(',')

# Create a one-hot encoding of the genres
tv_genre_enc = tv_merged['genres_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

# Combine the original DataFrame with the new one-hot encoded genre columns
df_final = tv_merged.drop('genres_list', axis=1).join(tv_genre_enc)

# Display the resulting DataFrame
df_final


# In[25]:


## Now, drop the original "genres" column

df_final = df_final.drop('genres', axis=1)
df_final


# ## Load in and merge more data to provide additional features for our predictive model

# In[26]:


## Let's load in a dataset from Kaggle, which was sourced from TMDB, in order to give us more columns in our dataset that 
## can help us recommend similar tv shows from a given tv show. 
## Source: https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows 

tmdb = pd.read_csv('TMDB_tv_dataset_v3.csv') 
tmdb


# In[27]:


tmdb.columns


# ## Drop duplicate rows and irrelevant columns

# In[28]:


## Does the id column of the tmdb dataset correspond to the tconst column of the df_final dataset? Answer: No

df_final[df_final['primaryTitle'] == 'Lucifer']


# In[29]:


## Only keep the columns that may be relevant for predicting similar tv shows. Also, remove any columns that 
## may provide duplicate information to avoid multicollinearity.

tmdb_short = tmdb[['name', 'number_of_seasons', 'number_of_episodes', 'original_language', 'overview', 'first_air_date', 
                   'last_air_date', 'in_production', 'popularity', 'type', 'status', 'tagline', 'created_by', 'networks', 
                   'origin_country', 'production_companies', 'episode_run_time']]
tmdb_short


# In[30]:


## Now, merge the 2 IMDB datasets (already merged in df_final) with the tmdb_short dataset

imdb_tmdb = pd.merge(df_final,tmdb_short, how="inner", left_on="primaryTitle", right_on="name")
imdb_tmdb


# In[31]:


## Drop NaN values from the imdb_tmdb dataset

merged_cleaned = imdb_tmdb.dropna()
merged_cleaned


# In[32]:


## Check for duplicate columns

merged_cleaned.columns


# In[33]:


## Duplicate columns identified: tconst, endYear, name

merged_shorter = merged_cleaned.drop(['tconst', 'endYear', 'name'], axis=1)
merged_shorter.columns


# In[34]:


## Double check if there are multiple rows with the same primaryTitle

merged_shorter['primaryTitle'].value_counts()


# In[35]:


## See what some observations look like that have the same primaryTitle

merged_shorter[merged_shorter['primaryTitle'] == 'The Twilight Zone']


# In[87]:


## Remove any columns with the same "primaryTitle" column, only keeping the first title

merged_unique = merged_shorter.drop_duplicates(subset='primaryTitle', keep='first')


# ## Create dummy columns for the merged dataset

# In[39]:


## Create dummy columns to ensure all columns are quantitative (aside from primaryTitle)
### Find which columns need to be dummified

pd.set_option('display.max_columns', None)
merged_unique.head()


# In[40]:


## Columns to be dummified: original_language, in_production, type, status, networks, origin_country

## Columns to be first split into a list and then dummified: overview, tagline, created_by (by ", "), 
## production_companies (by ", "), networks (by ", "), origin_country (by ", ")


# In[41]:


## Get dummy columns for the columns 'original_language', 'in_production', 'type', and 'status'

merged_dummified = pd.get_dummies(merged_unique, columns=['original_language', 'in_production', 'type', 'status'])
merged_dummified


# In[43]:


## Split into multiple words then dummify for the columns "created_by", "production_companies" and "origin_country"
## (dropping networks for simplification because many overlap with production companies)

merged_dummified['created_by_list'] = merged_dummified['created_by'].str.split(',')

created_by_enc = merged_dummified['created_by_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

merged_dummified_1 = merged_dummified.drop('created_by_list', axis=1).join(created_by_enc)


merged_dummified_1['production_comp_list'] = merged_dummified_1['production_companies'].str.split(',')

production_comp_enc = merged_dummified_1['production_comp_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

production_comp_enc = production_comp_enc.drop('Carter Bays', axis=1)

merged_dummified_2 = merged_dummified_1.drop('production_comp_list', axis=1).join(production_comp_enc)


merged_dummified_2['origin_country_list'] = merged_dummified_2['origin_country'].str.split(',')

origin_country_enc = merged_dummified_2['origin_country_list'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

merged_dummified_3 = merged_dummified_2.drop(['origin_country_list'], axis=1).join(origin_country_enc)


merged_dummified_3 = merged_dummified_3.drop(['created_by', 'production_companies', 'origin_country', 'networks'], axis=1)

merged_dummified_3


# In[44]:


## Now let's separate each word from the "overview" column of the dataframe and then one-hot-encode this column.
## We'll skip doing the same for the "tagline" column because there will be too much overlap, and instead we'll 
## just drop the tagline column.

merged_dummified_3['cleaned_overview'] = merged_dummified_3['overview'].str.replace(r'[^\w\s]', '', regex=True)
merged_dummified_3['cleaned_overview_list'] = merged_dummified_3['cleaned_overview'].str.split(' ')
overview_enc = merged_dummified_3['cleaned_overview_list'].apply(lambda x: pd.Series(1, index=pd.unique(x))).fillna(0)
overview_enc = overview_enc.drop(['Family', 'Drama', 'Crime', 'Mystery', 'Comedy', 'Western', 'Romance',
       'Fantasy', 'Musical', 'War', 'Short', 'Adult', 'popularity', 'networks',
       'ITV', 'WQED', 'BBC', 'Wellsville', 'HBO', 'CBC', 'Televisa', 'own',
       'Yellow', 'US', 'CA', 'IT'], axis=1)
merged_dummified_4 = merged_dummified_3.drop(['cleaned_overview_list'], axis=1).join(overview_enc)
merged_dummified_all = merged_dummified_4.drop(['cleaned_overview', 'overview', 'tagline'], axis=1)
merged_dummified_all


# In[45]:


## Now, let's create a single index that will take into account both the averageRating and numVotes from IMDB. 
## We can use this method: weighted rating (WR) = (v ÷ (v+3,000)) × R + (3,000 ÷ (v+3,000)) × 6.9

merged_dummified_all['weightedRating'] = (merged_dummified_all['numVotes'] / (merged_dummified_all['numVotes'] + 3000)) * merged_dummified_all['averageRating'] + (3000 / (merged_dummified_all['numVotes'] + 3000)) * 6.9

merged_dummified_clean = merged_dummified_all.drop(['numVotes', 'averageRating'], axis=1)
merged_dummified_clean


# In[46]:


## Let's double check what the type is of the column "first_air_date" and "last_air_date"
## Turns out it's a string. We'll need to change that

type(merged_dummified_clean.loc[58, 'first_air_date'])


# In[47]:


## Change the type of "first_air_date" and "last_air_date" to the Timestamp format

merged_dummified_clean['first_air_date'] = pd.to_datetime(merged_dummified_clean['first_air_date'], format='%Y-%m-%d')
print(type(merged_dummified_clean.loc[58, 'first_air_date']))

merged_dummified_clean['last_air_date'] = pd.to_datetime(merged_dummified_clean['last_air_date'], format='%Y-%m-%d')
print(type(merged_dummified_clean.loc[58, 'last_air_date']))


# ## Standardize required columns

# In[48]:


## Now let's standardize the following columns to avoid bias when applying our ML algorithms.
## Here are the columns requiring standardizing: runtimeMinutes, number_of_seasons, number_of_episodes, popularity, 
## episode_run_time

from sklearn.preprocessing import StandardScaler

for column in ['runtimeMinutes', 'number_of_seasons', 'number_of_episodes', 'popularity', 'episode_run_time']:
    merged_dummified_clean[column] = merged_dummified_clean[column].astype(float)
    
    # Standardize the column
    scaler = StandardScaler()
    merged_dummified_clean[column] = scaler.fit_transform(merged_dummified_clean[[column]])

print(np.mean(merged_dummified_clean['runtimeMinutes']))
print(np.std(merged_dummified_clean['runtimeMinutes']))

print(np.mean(merged_dummified_clean['popularity']))
print(np.std(merged_dummified_clean['popularity']))


# In[49]:


## We also need to standardize the first_air_date and last_air_date columns.

print(min(merged_dummified_clean['first_air_date']))
print(max(merged_dummified_clean['first_air_date']))


# In[50]:


## Standardize the "first_air_date" and "last_air_date" columns

# Convert to a Unix timestamp
merged_dummified_clean['unix_first_air_date'] = merged_dummified_clean['first_air_date'].apply(lambda x: x.timestamp())

# Now standardize
scaler = StandardScaler()
merged_dummified_clean['standard_first_air_date'] = scaler.fit_transform(merged_dummified_clean[['unix_first_air_date']])

# Convert to a Unix timestamp
merged_dummified_clean['unix_last_air_date'] = merged_dummified_clean['last_air_date'].apply(lambda x: x.timestamp())

# Now standardize
scaler = StandardScaler()
merged_dummified_clean['standard_last_air_date'] = scaler.fit_transform(merged_dummified_clean[['unix_last_air_date']])


# In[51]:


merged_final = merged_dummified_clean.drop(['unix_first_air_date', 'unix_last_air_date', 'first_air_date', 'last_air_date'], axis=1)


# ## Apply PCA to the dataset

# In[53]:


## There are nearly 16,000 columns, so we need to apply PCA to the dataset.

from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
pca_data = pca.fit_transform(merged_final.iloc[:, 2:])


# In[99]:


## By reducing the dataset to 1000 PCA components, we retain 93% of the variance.

pca.explained_variance_ratio_.cumsum()


# In[100]:


## Change the datatype of the "primaryTitle" column and reset the index

array = merged_final.loc[:, 'primaryTitle']
series = pd.Series(array)
series = series.reset_index(drop=True)


# In[101]:


series


# In[102]:


## Convert pca_data to a dataframe called "pca_dataframe"

pca_dataframe = pd.DataFrame(pca_data)


# In[103]:


## Add the "primaryTitle" column back to the pca_dataframe

pca_dataframe['Title'] = series
pca_dataframe


# In[104]:


## Export the final, cleaned dataframe as a csv

pca_dataframe.to_csv('final_df')


# # We can now conduct various machine learning algorithms on the smaller dataset.

# ## 1) K-means clustering

# In[105]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=500, random_state=42)  # Let's assume 500 clusters
kmeans.fit(pca_dataframe.iloc[:, :-1])

pca_dataframe['Cluster'] = kmeans.labels_


# In[106]:


pca_dataframe.head()


# In[107]:


### Let's examine which titles are in the same cluster. We can pick a random cluster -- cluster #211.

pca_dataframe[pca_dataframe['Cluster'] == 211]


# In[108]:


## Looks like only 1 title in a cluster. We may need to reduce the amount of clusters so that each show is 
## part of a cluster.
## Let's redo this clustering so that there are just 100 clusters.

kmeans = KMeans(n_clusters=100, random_state=42)  # Let's assume 100 clusters
kmeans.fit(pca_dataframe.iloc[:, :-2])

pca_dataframe['Cluster'] = kmeans.labels_


# In[109]:


### Let's examine which titles are in the same cluster. We can pick a random cluster -- cluster #3.

pca_dataframe[pca_dataframe['Cluster'] == 3]


# In[110]:


pca_dataframe['Cluster'].value_counts()


# In[111]:


### Since we don't know the optimal number of clusters, we can try hierarchical agglomeration clustering instead.


# ## 2) Hierarchical Agglomeration

# In[112]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linked = linkage(pca_dataframe.iloc[:, :-2], method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

cluster_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = cluster_model.fit_predict(pca_dataframe.iloc[:, :-2])

unique, counts = np.unique(labels, return_counts=True)
print("Cluster Sizes:", dict(zip(unique, counts)))


# ## 3) Equal-size Spectral Clustering

# In[113]:


## Source: https://towardsdatascience.com/equal-size-spectral-clustering-cce65c6f9ba3
## https://github.com/anamabo/Equal-Size-Spectral-Clustering/tree/main/source_code 


# In[114]:


import spectral_equal_size_clustering
import visualisation

clustering = spectral_equal_size_clustering.SpectralEqualSizeClustering(nclusters=100,
                                         nneighbors=13,
                                         equity_fraction=0.9,
                                         seed=1234
                                         )


# In[115]:


from scipy.spatial.distance import pdist, squareform
distance_matrix = squareform(pdist(pca_dataframe.iloc[:, :-2], metric='euclidean'))


# In[116]:


labels = clustering.fit(distance_matrix)


# In[117]:


pca_dataframe['ESSC Cluster'] = labels


# In[118]:


pca_dataframe


# In[119]:


## Check if the clusters are relatively equal-sized. For the most part, they are. 

pca_dataframe['ESSC Cluster'].value_counts()


# In[120]:


cluster = pca_dataframe[pca_dataframe['Title'] == "Gossip Girl"]['ESSC Cluster']
int(cluster.iloc[0])


# In[121]:


pca_dataframe[pca_dataframe['ESSC Cluster'] == 28]


# # Create the final recommendation function to predict similar tv shows

# In[122]:


import difflib

def find_similar_shows(title):
    if title in pca_dataframe['Title'].values:
        cluster = pca_dataframe[pca_dataframe['Title'] == title]['ESSC Cluster']
        cluster_int = int(cluster.iloc[0])
        print(pca_dataframe[pca_dataframe['ESSC Cluster'] == cluster_int]['Title'])
    else:
        column_values = pca_dataframe['Title'].tolist()
        closest_match = difflib.get_close_matches(title, column_values, n=1)
        if closest_match:
            cluster = pca_dataframe[pca_dataframe['Title'] == closest_match[0]]['ESSC Cluster']
            cluster_int = int(cluster.iloc[0])
            print(pca_dataframe[pca_dataframe['ESSC Cluster'] == cluster_int]['Title'])
        else:
            print("Sorry, this show is not in our database.")


# In[123]:


find_similar_shows('Scandal')


# In[ ]:




