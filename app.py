# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import os

# ---------------------------
# Recommendation System Class
# ---------------------------
class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        self.movies_df = pd.DataFrame()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.qualified_movies = pd.DataFrame()

    def clean_title_text(self, title):
        if not isinstance(title, str):
            return ''
        text = re.sub(r"\([^)]*\)", "", title)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def load_imdb_data(self, source):
        # source can be path / file-like / DataFrame
        if isinstance(source, str):
            df = pd.read_csv(source)
        elif hasattr(source, "read"):
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise ValueError("Unsupported source type for load_imdb_data")

        # minimal expected columns — fill missing with None
        expected_cols = ['orig_title', 'enhanced_content', 'score', 'vote_count', 'date_x', 'genre', 'crew', 'orig_lang', 'country']
        for c in expected_cols:
            if c not in df.columns:
                df[c] = None

        # keep a copy
        self.movies_df = df.copy()
        # prepare qualified_movies (used by searches)
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        if self.movies_df is None or self.movies_df.empty:
            raise ValueError("No movies_df loaded")
        working_df = self.movies_df.copy()
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        # build indices: title -> index
        # ensure titles are unique keys (drop duplicates keeps first)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    # helpers for app display
    def get_content_recommendations(self, title, n=10):
        if self.indices is None or self.cosine_sim is None:
            return (None, {}, None)
        if title not in self.indices:
            # simple substring fuzzy attempt
            matches = [t for t in self.indices.index if isinstance(t, str) and title.lower() in t.lower()]
            if not matches:
                return (None, {}, None)
            names = matches[:5]
            return ("choose", {'names': pd.Series(names)}, None)
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx]
        movie_indices = [i for i, _ in sim_scores[:n]]
        recs = self.qualified_movies.iloc[movie_indices]
        movie_info = self.qualified_movies.iloc[idx].to_dict()
        return ("ok", movie_info, recs)

    def display_centralized_results(self, df_or_dict, header, query, n=10):
        lines = []
        lines.append(f"=== {header} ({query}) ===")
        if isinstance(df_or_dict, dict):
            # single movie
            title = df_or_dict.get('orig_title', 'Unknown')
            year = df_or_dict.get('date_x', '')
            score = df_or_dict.get('score', '')
            lines.append(f"• {title} — Rating: {score}")
        else:
            # dataframe
            for _, x in df_or_dict.head(n).iterrows():
                title = x.get('orig_title', 'Unknown')
                year = x.get('date_x', '')
                score = x.get('score', '')
                lines.append(f"• {title} ({year}) — Rating: {score}")
        return "\n".join(lines)

    # simple search helpers (use qualified_movies)
    def search_by_genre(self, genre, n=10):
        if 'genre' not in self.qualified_movies.columns:
            return pd.DataFrame()
        matches = self.qualified_movies[self.qualified_movies['genre'].astype(str).str.contains(genre, case=False, na=False)]
        return matches.head(n)

    def search_by_crew(self, crew, n=10):
        if 'crew' not in self.qualified_movies.columns:
            return pd.DataFrame()
        matches = self.qualified_movies[self.qualified_movies['crew'].astype(str).str.contains(crew, case=False, na=False)]
        return matches.head(n)

    def search_by_year(self, year, n=10):
        self.qualified_movies['year'] = pd.to_datetime(self.qualified_movies['date_x'], errors='coerce').dt.year
        matches = self.qualified_movies[self.qualified_movies['year'] == year]
        return matches.head(n)

    def search_by_language(self, language, n=10):
        if 'orig_lang' not in self.qualified_movies.columns:
            return pd.DataFrame()
        matches = self.qualified_movies[self.qualified_movies['orig_lang'].astype(str).str.contains(language, case=False, na=False)]
        return matches.head(n)

    def get_top_movies_by_rating(self, n=20):
        if 'score' not in self.qualified_movies.columns:
            return self.qualified_movies.head(n)
        return self.qualified_movies.sort_values(by='score', ascending=False).head(n)


# ----------------- Streamlit App -----------------
def main():
    st.set_page_config(page_title="ENHANCED IMDB RECOMMENDER", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")

    st.markdown("✨ CENTRALIZED RESULTS: All searches now show comprehensive movie information!")
    st.write("")

    # ---------------- Sidebar upload / reset ----------------
    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")

    # Try to load bundled joblib first (minimal addition)
    local_joblib_path = "recommender.joblib"
    recommender = None
    if os.path.exists(local_joblib_path):
        st.sidebar.success(f"Found bundled model: {local_joblib_path}")
        try:
            recommender = joblib.load(local_joblib_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load bundled model: {e}")
            recommender = None

    if st.sidebar.button("🔄 Reset All Records"):
        for k in list(st.session_state.keys()):
            if k.startswith("choices_") or k.startswith("confirmed_") or k in ("last_query",):
                try:
                    del st.session_state[k]
                except:
                    pass
        st.experimental_rerun()

    # If no bundled recommender, fall back to uploaded CSV
    if recommender is None:
        if not uploaded_file:
            st.sidebar.info("No bundled model found. Upload imdb_movies.csv in the sidebar or place 'recommender.joblib' next to this app.")
            st.warning("Please upload imdb_movies.csv in the sidebar.")
            return

        recommender = IMDBContentBasedRecommendationSystem()
        try:
            recommender.load_imdb_data(uploaded_file)
            recommender.build_content_based_system()
        except Exception as e:
            st.error(f"Failed to load/build dataset: {e}")
            return

    # Main menu
    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title (Content-based recommendations)",
        "2️⃣ Search by Genre (Top-rated movies in genre)",
        "3️⃣ Search by Crew Member (Movies with specific actor/director)",
        "4️⃣ Search by Similarity (Find movies similar to a given movie)",
        "5️⃣ Search by Year (Movies from a specific year)",
        "6️⃣ Search by Language",
        "7️⃣ Top Rated",
        "🔟 Exit"
    ])

    # 1. Title
    if option.startswith("1️⃣"):
        st.subheader("Search by Movie Title (Content-based recommendations)")
        title = st.text_input("🎬 Enter a movie title:")
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            cleaned = recommender.clean_title_text(title)
            status, movie_info, recs = recommender.get_content_recommendations(cleaned, n=n_recs)
            if status is None:
                st.error("❌ No matches found.")
            elif status == "choose":
                choices = movie_info['names'].tolist() if 'names' in movie_info else []
                choice = st.selectbox("Did you mean one of these?", choices)
                if st.button("Confirm Selection"):
                    st.write(recommender.display_centralized_results({'orig_title': choice}, "Selection", choice, n=1))
            elif status == "ok":
                st.code(recommender.display_centralized_results(movie_info, "Content Recommendations", title, n_recs), language="text")

    # 2. Genre
    elif option.startswith("2️⃣"):
        st.subheader("Search by Genre")
        genre = st.text_input("🎭 Enter a genre:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Genre"):
            res = recommender.search_by_genre(genre, n=n_results)
            st.code(recommender.display_centralized_results(res, "Genre Search", genre, n=n_results), language="text")

    # 3. Crew
    elif option.startswith("3️⃣"):
        st.subheader("Search by Crew Member (actor/director)")
        crew = st.text_input("👥 Enter crew member name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Crew"):
            res = recommender.search_by_crew(crew, n=n_results)
            st.code(recommender.display_centralized_results(res, "Crew Search", crew, n=n_results), language="text")

    # 4. Similarity
    elif option.startswith("4️⃣"):
        st.subheader("Find movies similar to a given movie")
        movie_title = st.text_input("🎬 Enter a base movie title:")
        n_sim = st.slider("📊 Number of similar movies", 1, 20, 5)
        if st.button("Find Similar Movies"):
            status, movie_info, recs = recommender.get_content_recommendations(recommender.clean_title_text(movie_title), n=n_sim)
            if status == "ok" and recs is not None:
                st.code(recommender.display_centralized_results(recs, "Similar Movies", movie_title, n=n_sim), language="text")
            else:
                st.error("No similar movies found or ambiguous title.")

    # 5. Year
    elif option.startswith("5️⃣"):
        st.subheader("Search by Year")
        year = st.number_input("📅 Enter year:", min_value=1800, max_value=2100, value=2000)
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Year"):
            res = recommender.search_by_year(int(year), n=n_results)
            st.code(recommender.display_centralized_results(res, "Year Search", year, n=n_results), language="text")

    # 6. Language
    elif option.startswith("6️⃣"):
        st.subheader("Search by Language")
        language = st.text_input("🗣️ Enter language:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Language"):
            res = recommender.search_by_language(language, n=n_results)
            st.code(recommender.display_centralized_results(res, "Language Search", language, n=n_results), language="text")

    # 7. Top Rated
    elif option.startswith("7️⃣"):
        st.subheader("Top Rated Movies")
        n_results = st.slider("📊 Number of results", 1, 50, 20)
        if st.button("Show Top Rated Movies"):
            res = recommender.get_top_movies_by_rating(n=n_results)
            st.code(recommender.display_centralized_results(res, "Top Rated Movies", f"Top {n_results} Movies", n=n_results), language="text")

    # Exit
    elif option.startswith("🔟"):
        st.write("Exiting — close the browser tab or choose another option.")

if __name__ == "__main__":
    main()
