import streamlit as st
import pandas as pd
import re

from difflib import get_close_matches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD


st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 Hybrid Movie Recommendation System")

st.markdown(
    """
    This system combines:

    - **Content-Based Filtering**
    - **Collaborative Filtering (SVD)**

    to generate personalized movie recommendations.
    """
)

st.markdown("---")


# Load and Prepare Models

@st.cache_resource
def prepare_models():

    # Load datasets
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies['genres'] = movies['genres'].str.replace(
        '|',
        ' ',
        regex=False
    )

    # Content-Based Filtering

    movies_cb = movies[['movieId', 'title', 'genres']].copy()

    # Combine title and genres
    movies_cb['content'] = (
        movies_cb['title']
        + ' '
        + movies_cb['genres']
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        stop_words='english'
    )

    tfidf_matrix = tfidf.fit_transform(
        movies_cb['content']
    )

    # Cosine Similarity
    cosine_sim = cosine_similarity(
        tfidf_matrix
    )

    # Title Preprocessing

    def preprocess_title(title):

        title = re.sub(r'[.,]', '', title)

        title = title.strip().lower()

        return title

    movies_cb['processed_title'] = (
        movies_cb['title'].apply(preprocess_title)
    )

    # Collaborative Filtering

    reader = Reader(
        rating_scale=(0.5, 5)
    )

    data = Dataset.load_from_df(
        ratings[['userId', 'movieId', 'rating']],
        reader
    )

    trainset = data.build_full_trainset()

    svd_model = SVD()

    svd_model.fit(trainset)

    return (
        movies,
        movies_cb,
        cosine_sim,
        svd_model,
        preprocess_title
    )

# Load Models Once

(
    movies,
    movies_cb,
    cosine_sim,
    svd_model,
    preprocess_title
) = prepare_models()

# Hybrid Recommendation Function

def hybrid_recommend(
    user_id,
    movie_title,
    top_n=10,
    w_content=0.5,
    w_collab=0.5
):

    # Clean title
    movie_title = preprocess_title(movie_title)

    # Get all processed titles
    titles = movies_cb['processed_title'].tolist()

    # Find closest matching movie
    matches = get_close_matches(
        movie_title,
        titles,
        n=1,
        cutoff=0.5
    )

    # If movie not found
    if not matches:
        return []

    matched_title = matches[0]

    # Get movie index
    idx = movies_cb[
        movies_cb['processed_title'] == matched_title
    ].index[0]

    # Get similarity scores
    sim_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Sort by similarity
    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie itself
    sim_scores = sim_scores[1:51]

    recommendations = []

    # Generate hybrid scores
    for i, content_score in sim_scores:

        movie_id = movies_cb.iloc[i]['movieId']

        title = movies_cb.iloc[i]['title']

        genres = movies_cb.iloc[i]['genres']

        # Collaborative prediction
        collab_score = svd_model.predict(
            user_id,
            movie_id
        ).est

        # Normalize collaborative score
        collab_score_norm = collab_score / 5.0

        # Final hybrid score
        hybrid_score = (
            w_content * content_score
            + w_collab * collab_score_norm
        )

        recommendations.append({
            "Movie Title": title,
            "Genres": genres,
            "Recommendation Score": round(hybrid_score, 3)
        })

    # Sort final results
    recommendations = sorted(
        recommendations,
        key=lambda x: x["Recommendation Score"],
        reverse=True
    )

    return recommendations[:top_n]

# Sidebar

st.sidebar.header("⚙️ Recommendation Settings")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    value=1
)

movie_name = st.sidebar.selectbox(
    "Select a Movie",
    movies['title'].tolist()
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=10
)

# Weight Tuning

st.sidebar.subheader("Hybrid Weights")

w_content = st.sidebar.slider(
    "Content-Based Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

w_collab = round(1.0 - w_content, 1)

st.sidebar.write(
    f"Collaborative Weight: {w_collab}"
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Developed using Streamlit, TF-IDF, Cosine Similarity, and SVD."
)


# Recommendation Button

if st.button("🎯 Get Recommendations"):

    results = hybrid_recommend(
        user_id=user_id,
        movie_title=movie_name,
        top_n=top_n,
        w_content=w_content,
        w_collab=w_collab
    )

    if results:

        st.subheader("🎥 Recommended Movies")

        results_df = pd.DataFrame(results)

        results_df.index = (
            results_df.index + 1
        )

        # table
        st.dataframe(
            results_df,
            use_container_width=True
        )

        # Message
        st.success(
            "Recommendations generated successfully!"
        )

    else:

        st.error(
            "Movie not found!"
        )


st.markdown("---")

st.caption(
    "Hybrid Movie Recommendation System | "
    "Content-Based + Collaborative Filtering"
)