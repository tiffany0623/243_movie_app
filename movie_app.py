import streamlit as st
import pandas as pd
import requests
from io import StringIO

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
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amatic+SC:wght@400;700&display=swap');

    .header-text {
        font-weight: bold;
        font-size: 120px;
        font-family: 'Amatic SC', cursive;
        color: #F4BC07;
    }
    </style>
    <h1 class='header-text'>Popcorn Picks 🍿</h1>
""", unsafe_allow_html=True)

## Introduction
st.markdown("<p style=#9C9D9F; font-size: 18px; \
            '>Welcome to Popcorn Picks, the <b>free movie recommendation system</b> \
            that suggests films based on your interest.</p>",
            unsafe_allow_html=True)

st.markdown("---")


# Import Data

# cb_url= 'https://docs.google.com/spreadsheets/d/1doXKqSMVT65TVW4XgDAXVm3O0_BO9jXThYdJT568peM/edit?usp=sharing'
# rating_url= 'https://docs.google.com/spreadsheets/d/1HGClAHQv-uLvBjqqo1mDGNgYhRNnRpDzhvnsk0x6R8Q/edit?usp=sharing'
# movies_url= 'https://docs.google.com/spreadsheets/d/1qVl3gDVHUy7JdUz0fw74zFQc-RDE0_zRJ9iWIh1Nhtk/edit?usp=sharing'


# csv_cb_url = cb_url.replace('/edit?usp=sharing', '/export?format=csv')
# csv_rating_url= rating_url.replace('/edit?usp=sharing','/export?format=csv')
# csv_movies_url=movies_url.replace('/edit?usp=sharing','/export?format=csv')



url_final = 'https://drive.google.com/uc?export=download&id=1LwQ0qtjIvIRKSOjuzTYXGpXoF1mOv9Xp'
url_rating = 'https://drive.google.com/uc?export=download&id=17QweXIk6u8KHEnA6pqxdlcwXLuybaZyE'


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

def main():
    
    global rating
    global final_combined_df

    url_final = 'https://drive.google.com/uc?export=download&id=1LwQ0qtjIvIRKSOjuzTYXGpXoF1mOv9Xp'
    url_rating = 'https://drive.google.com/uc?export=download&id=17QweXIk6u8KHEnA6pqxdlcwXLuybaZyE'
    
    final_combined_df = load_data_via_requests(url_final)
    rating = load_data_via_requests(url_rating)


if __name__ == "__main__":
   main()

# final_combined_df = load_data_via_requests(url_final)
# rating = load_data_via_requests(url_rating)


# Custom UI
# title - pick movies
st.markdown("""
            <h3 style='font-weight:bold; font-size:35px; font-family:Roboto, Arial; color: #330000;'>
            Rate Movies
            </h3>""", 
            unsafe_allow_html=True)

st.markdown("<p style=#9C9D9F; font-size: 18px; \
            '>Please rate the 10 movies below. <br>\
            \"0\": ❓ (Have not watched yet) <br>\
            \"1\": 😠 (1 Star) <br>\
            \"2\": 😕 (2 Stars) <br>\
            \"3\": 😐 (3 Stars) <br>\
            \"4\": 😃 (4 Stars) <br>\
            \"5\": 😍 (5 Stars)\
            </p>",
            unsafe_allow_html=True)


# Custom CSS for the UI
st.markdown(
    """
    <style>
    body {
        background-color: #eed9c4;  /* White background color */
        color: #FFD700;            /* Text color (Gold) */
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
        color: #FFD700;            /* Selectbox text color (Gold) */
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
selected_movies = final_combined_df[final_combined_df['movieId'].isin(selected_movie_ids)]


# Create a session state to store selected movies and posters (showing only top 1000 movies to rate)
if 'selected_movies' not in st.session_state:
    st.session_state['selected_movies'] = selected_movies


# Assuming initialization and setup are done earlier
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
        movie_name =movie['title']
        poster_url = movie['imageURL']

        with columns[i]:  # Use columns within the current row
            st.markdown(f"<div class='movie-container'>", unsafe_allow_html=True)
            if poster_url is not None:  # Check if poster_url is not empty
                st.image(poster_url, caption=movie_name, width=image_width)

            # Rating logic
            rating = st.selectbox("", ("Rate the movie","0", "1", "2", "3", "4", "5"), key=f"{movie_name}_rating")
            if rating != 'Rate the movie':
                user_ratings[movie_name] = rating
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

    for percent_complete in range(6):
        time.sleep(0.1)
        my_bar.progress(percent_complete)
    
    pass
    #progress_text.empty()
    #my_bar.empty()


# Button to trigger recommendations
if st.button("Recommend"):
    # Check if the user has rated at least one movie
    if  len(user_ratings.values()) == 0:
        st.markdown("<p style='color:#F4BC07; font-size: 18px;'><b>Please rate at least one movie before proceeding.</b></p>",
                    unsafe_allow_html=True)
    else:
        progress_bar()
        #st.write("<span style='color: #FFBF00;font-size: 18px;'><b>Personalizing your Recommendations...</b></span>", unsafe_allow_html=True)

        # Every user is first time user
        first_time_user = True

        recommended_movie_ids,recommended_movie_names, recommended_movie_posters = \
            recommend(1111111, user_ratings, first_time_user, n_movies_to_recommend=10)

        if len(recommended_movie_names) != 0:

            if len(recommended_movie_names) < 5:
                n_movies_to_display = len(recommended_movie_names)
            
            # I am displaying only 5 movies in one row
            n_movies_to_display = 5

            cnt = 0

            st.write("<span style='color: #00BFFF;font-size: 18px;'><b>Here are a few Recommendations..........</b></span>",\
                      unsafe_allow_html=True)
            columns = st.columns(n_movies_to_display)
            
            for i, movie_name in enumerate(recommended_movie_names):
                if cnt == 5:
                    break
                cnt += 1
                try:
                    with columns[i]:
                        st.markdown("<div class='movie-container'>", unsafe_allow_html=True)
                        st.image(recommended_movie_posters[i], caption=movie_name, output_format="PNG", width=200)
                        st.markdown("</div>", unsafe_allow_html=True)
                except:
                    st.markdown(f"![{movie_name}]({recommended_movie_posters[i]})")

                # columns[i].image(recommended_movie_posters[i], caption=movie_name, output_format="PNG", width=150)
        else:
            st.markdown("<p style='color:#FFBF00; font-size: 18px;'><b>Sorry, We couldn't find any recommendations for you.</b></p>",
                        unsafe_allow_html=True)
            
            
    

