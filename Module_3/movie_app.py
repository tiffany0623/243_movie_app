
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

#import models
from Content_based import ContentBasedRecommender, build_users_profiles
from Collaborative_Filtering import CFRecommender
from Hybrid import HybridRecommender

# Set the page's title and icon
st.set_page_config(
    page_title="Popcorn Pick",
    page_icon="🍿️",
    layout="wide",)


# Cover
header_image_url = 'https://drive.google.com/uc?export=view&id=1P7LlemoRhIZE-agKbaTtnrssYACEJ7Ox'
response = requests.get(header_image_url)
st.image(response.content)

# Background color
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-color: #eed9c4;
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""



st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom layout to show tagline aligned to the left
## Header
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Amatic+SC:wght@400;700&display=swap');

#     .header-text {
#         font-weight: bold;
#         font-size: 120px;
#         font-family: 'Amatic SC', cursive;
#         color: #F4BC07;
#     }
#     </style>
#     <h1 class='header-text'>Popcorn Picks 🍿</h1>
# """, unsafe_allow_html=True)

st.markdown("""
    <style>
    @keyframes pop {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.1);
        }
    }
    .popcorn-animation {
        display: inline-block;
        animation: pop 1s ease infinite;
    }
    .animated-title {
        font-weight: bold;
        font-size: 120px;
        font-family: 'Amatic SC', cursive;
        color: #F4BC07;
    }
    </style>
    <h1 class="animated-title"><span class="popcorn-animation">🍿</span> Popcorn Pick <span class="popcorn-animation">🍿</span></h1>
""", unsafe_allow_html=True)

## Introduction
st.markdown("""
    <p style='color: #660000; font-size: 22px;'>
        Welcome to Popcorn Picks-Your Ultimate Guide to Unforgettable Movie Nights! <br>
        Dive into a world where films meet your mood, curated just for you.
             <br>
             
    </p>
""", unsafe_allow_html=True)

st.markdown("---")



# Import Data
@st.cache_data
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





# Load datasets 
movies_df = load_data_via_requests('https://drive.google.com/uc?export=download&id=1LwQ0qtjIvIRKSOjuzTYXGpXoF1mOv9Xp')
rating = load_data_via_requests('https://drive.google.com/uc?export=download&id=17QweXIk6u8KHEnA6pqxdlcwXLuybaZyE')

# Add a new 'imdbUrl' column to movies_df
movies_df['imdbUrl'] = 'https://www.imdb.com/title/tt' + movies_df['imdbId'].apply(lambda x: f"{int(x):07d}") + '/'

ratings_with_titles_df = rating.merge(movies_df, on='movieId', how='left')
# Custom UI
# title - pick movies
st.markdown("""
            <h3 style='font-weight:bold; font-size:35px; font-family:Roboto, Arial; color: #330000;'>
            Rate Movies
            </h3>""", 
            unsafe_allow_html=True)
#951512
st.markdown("""
    <style>
    .rating-text {
        color: #660000;  
        font-size: 16px;  /* Slightly Larger Font Size */
        font-family: 'Comic Sans MS', cursive, sans-serif;  /* A Fun Font */
        background-color: #DB9D35;  /* Dark Background for Contrast */
        padding: 10px;  /* Some Padding for Aesthetic Spacing */
        border-radius: 10px;  /* Rounded Corners */
        border: 2px solid #F4BC07;  
        margin-bottom: 20px; /* Space below the box */
    }
    </style>
    <p class="rating-text">🌟 Time to be the critic! Rate the movies below 🎬<br>
    "1": 😡 (That was... unfortunate)<br>
    "2": 🤨 (Could've been worse)<br>
    "3": 😌 (It was okay, I guess)<br>
    "4": 😄 (Really liked it!)<br>
    "5": 🥳 (Mind. Blown. Loved it!)<br>
    🤷‍♂️ If you haven't watch this movie, please do not make a selection. <br>
    🍿️ Need more options? Refresh the page to discover new movies tailored just for you! </p>
""", unsafe_allow_html=True)



# Custom CSS for the UI
st.markdown(
    """
    <style>
    body {
        background-color: #eed9c4;  /*  background color */
        color: #FFD700;            /* Text color  */
        font-family: 'Comic Sans MS', cursive;
    }
    .movie-container {
        margin-right: 20px; /* Add margin to create space between movies */
        margin-bottom: 40px; /* Add bottom margin to ensure space above the select box */
    }
    .stButton>button {
        background-color: #F4BC07; /* Remove the background color of the button */
        color: #FFFFFF;
        border-radius: 10px;
    }
    .stSelectbox {
        color: #FFD700;            /* Selectbox text color  */
        background-color: #eed9c4;  /* Selectbox background color */
        height: 40px;
    }
    .stRating {
        font-size: 24px;
    }
    .stSelectbox > div:first-child {
        background-color: #eed9c4;   /* Selectbox open button background color */
    }
    .stSelectbox > div:last-child {
        background-color: #eed9c4;   /* Selectbox dropdown background color */
    }
    .poster-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        width: 180px;  /* Adjust the width as needed */
        height: 220px;  /* Fixed height for the container */
        justify-content: space-between;
        margin-bottom:30px;
    }
    .poster {
        max-width: 90%;  /* Max width for the poster */
        max-height: 80%;  /* Max height for the poster */
    }
    .caption {
        background-color: #eed9c4;
        height: 30px;  /* Fixed height for the caption */
        overflow: hidden; /* Hide overflow text */
    }
    .header-text {
        text-align: left;
    }
    .text-container {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define function - randomly pick movies
import random
def top_1000_movies_ids(df):
    
    # Assume the file has a column named 'movieId' for the movie identifier
    # Group by 'movieId', count the number of ratings for each movie,
    # and sort the result in descending order of rating count
    top_movies = df.groupby('movieId').size().sort_values(ascending=False).head(1000)

    # Get the movie IDs as a list
    top_movie_ids = top_movies.index.tolist()

    # Return the list of top 1000 movie IDs
    return top_movie_ids


popular = top_1000_movies_ids(rating)
n_random_movies=20
selected_movie_ids = random.sample(popular, n_random_movies)
selected_movies = movies_df[movies_df['movieId'].isin(selected_movie_ids)]



# Initialize the watchlist in the session state if it doesn't exist
if 'my_watchlist' not in st.session_state:
    st.session_state['my_watchlist'] = []





# Create a session state to store selected movies and posters (showing only top 1000 movies to rate)
if 'selected_movies' not in st.session_state:
    st.session_state['selected_movies'] = selected_movies


# Sidebar for the watchlist
st.sidebar.title("My Watchlist")
watchlist_area = st.sidebar.empty()  # This area will be updated with the watchlist

if 'rec_movies' not in st.session_state:
    st.session_state['rec_movies'] = pd.DataFrame()

selectbox_width = 180  # Adjust the width of the selectbox as needed
image_width = selectbox_width
user_ratings = {}
n_movies_to_display_per_row = 5  # Number of movies to display in one row

cnt = 0  # Counter for total movies displayed to manage the break across rows

for row in range(2):  # For creating two rows
    columns = st.columns(n_movies_to_display_per_row)  # Creates 5 columns for each row
    
    for i in range(n_movies_to_display_per_row):
        # Calculate movie index based on row and column
        movie_index = row * n_movies_to_display_per_row + i
        if movie_index >= len(st.session_state.selected_movies):
            break  # Exit if no more movies to display

        movie = st.session_state.selected_movies.iloc[movie_index]
        movie_id = movie['movieId']
        movie_name =movie['title']
        poster_url = movie['imageURL']

        with columns[i]:  # Use columns within the current row
            st.markdown(f"<div class='movie-container'>", unsafe_allow_html=True)
            if poster_url is not None:  # Check if poster_url is not empty
                st.image(poster_url, caption=movie_name, width=image_width)

            # Rating logic
            rating = st.selectbox("", ("Rate the movie", "1", "2", "3", "4", "5"), key=f"{movie_name}_rating")
            if rating != 'Rate the movie':
                user_ratings[movie_id] = rating
        st.markdown("</div>", unsafe_allow_html=True)
    

        cnt += 1  # Increment the total movie counter
        if cnt >= n_movies_to_display_per_row * 2:  # Check if the total count reaches 10 (5 per row * 2 rows)
            break

print(user_ratings)

# Recommendation List
st.markdown("---")

st.markdown("""
            <h3 style='font-weight:bold; font-size:35px; font-family:Roboto, Arial; color: #330000;'>
            Recommend Movies
            </h3>""", 
            unsafe_allow_html=True)

def custom_css():
    # Inject custom CSS
    st.markdown("""
    <style>
    /* Custom text color */
    .custom-text {
        color: #660000; /* Gold color */
    }
    
    /* Custom progress bar color */
    /* The specific class names may change with Streamlit updates */
    .stProgress > div > div > div > div {
        background-color: #F4BC07 !important;
    }
    </style>
    """, unsafe_allow_html=True)

import time
def progress_bar():
    custom_css()  # Apply custom styles

    
    progress_text = st.markdown("<p class='custom-text'>Personalizing your Recommendations...</p>", unsafe_allow_html=True)
    my_bar = st.progress(0)

    for percent_complete in range(101):
        time.sleep(0.1)
        my_bar.progress(percent_complete)
    
    # pass
    progress_text.empty()
    my_bar.empty()

def create_ratings_df(user_ratings, user_ID=987654321):
    # Prepare the data for the new DataFrame
    data = {
        "userId": [user_ID] * len(user_ratings),# Repeat user_ID for each entry
        "movieId": list(user_ratings.keys()),  # Movie IDs as Uid
        "rating": list(user_ratings.values())  # User ratings
    }
    # Create and return the new DataFrame
    new_user_df = pd.DataFrame(data)
    return new_user_df

def svd_pred(user_interactions_df):
    users_items_pivot_matrix_df = user_interactions_df.pivot(index='userId',
                                                             columns='movieId',
                                                             values='rating').fillna(0)
    users_items_pivot_matrix_df = users_items_pivot_matrix_df.apply(pd.to_numeric)

    users_items_pivot_matrix = users_items_pivot_matrix_df.to_numpy()
    users_ids = list(users_items_pivot_matrix_df.index)

    users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)

    U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k=7)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / \
                                      (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())

    cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=users_items_pivot_matrix_df.columns,
                               index=users_ids).transpose()

    return cf_preds_df

def display_movies(rec_movies):
    n_movies_to_display_per_row = 5
    for row in range(2):  # Create rows of movies
        columns = st.columns(n_movies_to_display_per_row)
        for i in range(n_movies_to_display_per_row):
            movie_index = row * n_movies_to_display_per_row + i
            if movie_index >= len(rec_movies):
                break
            movie = rec_movies.iloc[movie_index]
            with columns[i]:
                display_movie_details(movie)

# Function to display movie details including Add to Watchlist button and IMDb link
def display_movie_details(movie):
    movie_id = movie['movieId']
    movie_name = movie['title']
    poster_url = movie.get('imageURL', '')
    imdb_link = movie['imdbUrl']  
    
    with st.container():
        if poster_url:
            st.image(poster_url, width=180, caption=movie_name)
        
        if st.button(f"⭐ Add to Watchlist", key=f"add_{movie_id}"):
            movie_info = {"title": movie_name, "imdbUrl": imdb_link}
            # Check if movie already in watchlist to prevent duplicates
            if movie_info not in st.session_state['my_watchlist']:
                st.session_state['my_watchlist'].append(movie_info)  # Add movie info to watchlist
                st.success(f"'{movie_name}' added to watchlist!")
            else:
                st.error(f"'{movie_name}' is already in the watchlist.")
            
            # Update the sidebar watchlist display
            update_watchlist_area()
        
        # Display the IMDb link
        st.markdown(f"[{movie_name}]({imdb_link})", unsafe_allow_html=True)

# Function to update and display the watchlist in the sidebar
def update_watchlist_area():
    # Empty previous content in the sidebar area dedicated for the watchlist
    watchlist_area = st.sidebar.container()
    if st.session_state['my_watchlist']:
        # Display each movie as a clickable link to its IMDb page
        for movie_info in st.session_state['my_watchlist']:
            watchlist_area.markdown(f"[{movie_info['title']}]({movie_info['imdbUrl']})", unsafe_allow_html=True)
    else:
        watchlist_area.write("Your watchlist is currently empty.")




#Debug  
#watchlist_area.write(st.session_state['my_watchlist'])




# Button to trigger recommendations
if st.button("Recommend"):
    # Check if the user has rated at least two movie
    if  len(user_ratings.values()) < 2:
        st.markdown("<p style='color:#F4BC07; font-size: 18px;'><b>Please rate at least two movie before proceeding.</b></p>",
                    unsafe_allow_html=True)
    else:

        progress_bar()
        #st.write("<span style='color: #FFBF00;font-size: 18px;'><b>Personalizing your Recommendations...</b></span>", unsafe_allow_html=True)


        new_user_df = create_ratings_df(user_ratings, user_ID=987654321)

        #CB
        new_user_profile = build_users_profiles(new_user_df,movies_df)
        content_based_recommender_model = ContentBasedRecommender(movies_df)
        #cb_recommend_df = content_based_recommender_model.recommend_items(user_id=987654321, user_profile=new_user_profile)

        #CF
        total_interactions_df = pd.concat([ratings_with_titles_df[['movieId', 'userId', 'rating']], new_user_df],axis=0)
        cf_preds_df = svd_pred(total_interactions_df)

        cf_recommender_model = CFRecommender(cf_preds_df, movies_df)
        #cf_recommend_df = cf_recommender_model.recommend_items(user_id=987654321,user_profile=new_user_profile)

        #Hybrid
        hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, movies_df,
                                                     cb_ensemble_weight=0.47, cf_ensemble_weight=0.53)
        Hybrid_recommend_df = hybrid_recommender_model.recommend_items(user_id=987654321,user_profile=new_user_profile)

        movei_rec_id = Hybrid_recommend_df['movieId'].values
        rec_movies = movies_df[movies_df['movieId'].isin(movei_rec_id)]
        st.session_state['rec_movies'] = rec_movies  # Store the recommendations

rec_movies = st.session_state.get('rec_movies', pd.DataFrame())

if not rec_movies.empty:
    st.write("<span style='color: #660000;font-size: 18px;'><b>Here is your personalized movie list! 🎉🎉</b></span>", unsafe_allow_html=True)
    display_movies(rec_movies)

