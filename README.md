# TV-Recommendation

<img width="1440" alt="Screen Shot 2024-10-05 at 4 39 13 PM" src="https://github.com/user-attachments/assets/b3d17718-e196-4e7a-adfa-0b0c153a0b43">

### Author: Nishka Govil

### Corresponding Website: https://govilnishka.pythonanywhere.com/

### About

This TV recommendation system generates a list of similar tv shows, in no particular order, given a user's input tv show. If the user made a spelling mistake in typing out the show, the website will generate similar shows to a title that best matches the inputted value. If the title is not in the database, the website will list "No similar shows found."

### Models Used

1. K-means clustering
2. Hierarchical algglomeration
3. Equal-size spectral clustering (source: https://towardsdatascience.com/equal-size-spectral-clustering-cce65c6f9ba3, https://github.com/anamabo/Equal-Size-Spectral-Clustering/tree/main/source_code)

### Required Setup:

#### Download the required datasets and external sources. These files were too large to upload to this repository.
Download "title.basics.tsv.gz" and "title.ratings.tsv.gz" from https://datasets.imdbws.com/. These corresponding tsv's are referenced as "title.basics.tsv" and "title.ratings.tsv", respectively, in the file EDA_models.py. 

Download the dataset from https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows. This is referenced in the file EDA_models.py as "TMDB_tv_dataset_v3.csv"

Download all files from the source_code folder from https://github.com/anamabo/Equal-Size-Spectral-Clustering/tree/main/source_code. This is used for the "Equal-Size Spectral Clustering model" that ultimately generates the tv recommendations in the file EDA_models.py.

#### Generate a file titled "final_df"
The file Rec_Algorithm.py references a file called "final_df." This file was generated from the line "pca_dataframe.to_csv('final_df')" in the file EDA_models.py. Thus, run the file EDA_models.py in its entirety before running Rec_Algorithm.py.

### Files
**EDA_models.py:** This file conducts EDA and data cleaning on two IMDB datasets and a Kaggle dataset sourced from TMDB. Once the data sources were merged and cleaned, various models were applied to the final dataset in order to construct TV recommendations from a user's inputted TV show.

**Rec_Algorithm.py:** This file isolates the final recommendation algorithm from the EDA_models.py file and is referenced in application.py to build the website.

**application.py:** This file uses Flask to create a website where a user can input a TV show to receive a list of recommended TV shows.

**templates/index.html:** This HTML file provides the content for the home page of the website. 

**templates/recommendations.html:** This HTML file provides the content for the recommendations page of the website.

**Acknowledgement:** Portions of the code from templates/index.html and templates/recommendations.html were generated via ChatGPT. I do not claim to have full proficiency in HTML.








