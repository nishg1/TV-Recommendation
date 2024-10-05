#!/usr/bin/env python
# coding: utf-8

# In[15]:


import difflib
import pandas as pd

final_df = pd.read_csv('final_df')

def find_similar_shows(title):
    if title in final_df['Title'].values:
        cluster = final_df[final_df['Title'] == title]['ESSC Cluster']
        cluster_int = int(cluster.iloc[0])
        recommendations = final_df[final_df['ESSC Cluster'] == cluster_int]['Title']
        if not recommendations.empty:
            return recommendations.tolist()
        else:
            return "No similar shows found"

    else:
        column_values = final_df['Title'].tolist()
        closest_match = difflib.get_close_matches(title, column_values, n=1)
        if closest_match:
            cluster = final_df[final_df['Title'] == closest_match[0]]['ESSC Cluster']
            cluster_int = int(cluster.iloc[0])
            recommendations2 = final_df[final_df['ESSC Cluster'] == cluster_int]['Title']
            if not recommendations2.empty:
                return recommendations2.tolist()  # Convert Series to a list
            else:
                return "No similar shows found"
        else:
            return "No similar shows found"


# In[ ]:




