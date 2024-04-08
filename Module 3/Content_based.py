import pandas as pd
import numpy as np
import requests
from io import StringIO
import streamlit as st
from scipy.sparse import load_npz
from ast import literal_eval
import scipy
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import sklearn

st.cache_data
def load_data_via_requests(url):
    r = requests.get(url)
    if r.status_code == 200:
        # Use StringIO from io module
        data = StringIO(r.text)
        df = pd.read_csv(data)
        return df
    else:
        st.error(f"Failed to download data. Status code: {r.status_code}")
        return pd.DataFrame()

url_final = 'https://drive.google.com/uc?export=download&id=1LwQ0qtjIvIRKSOjuzTYXGpXoF1mOv9Xp'
url_rating = 'https://drive.google.com/uc?export=download&id=17QweXIk6u8KHEnA6pqxdlcwXLuybaZyE'
movies_df = load_data_via_requests(url_final)
rating = load_data_via_requests(url_rating)


    # Now you can access movies_df here or wherever you need it
tfidf_matrix = load_npz("tfidf_matrix.npz")
item_ids = movies_df['movieId'].tolist()


def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile


def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles


def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]

    if type(interactions_person_df['movieId']) == np.int64:
        ids = []
        ids.extend([interactions_person_df['movieId']])

    else:
        ids = interactions_person_df['movieId'].values

    user_item_profiles = get_item_profiles(interactions_person_df['movieId'])
    user_item_strengths = np.array(interactions_person_df['rating'].astype(float)).reshape(-1, 1)

    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(np.asarray(user_item_strengths_weighted_avg))
    return user_profile_norm


def build_users_profiles(new_user_df,movies_df):
    interactions_indexed_df = new_user_df[new_user_df['movieId'] .isin(movies_df['movieId'])].set_index('userId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, new_user_profile, topn=1000):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(new_user_profile[person_id], tfidf_matrix)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, user_profile, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id, user_profile)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['movieId', 'recRating']) \
            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='movieId',
                                                          right_on='movieId')[['recRating', 'movieId', 'title']]

        return recommendations_df
