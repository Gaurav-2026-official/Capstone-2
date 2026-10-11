import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RestaurantRecommender:
    """Content-based recommender using restaurant metadata and TF-IDF cosine similarity."""
    def __init__(self, restaurants: pd.DataFrame):
        self.restaurants = restaurants.reset_index(drop=True).copy()
        # Keep the loader's column mapping, but ensure filter columns are string-safe.
        for col in ['name', 'location', 'cuisines', 'online_order', 'book_table']:
            if col in self.restaurants.columns:
                self.restaurants[col] = self.restaurants[col].astype('string').fillna('')
        text = self.restaurants.get('search_text', pd.Series('', index=self.restaurants.index)).fillna('').astype(str)
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=60000)
        self.matrix = self.vectorizer.fit_transform(text) if len(text) else None

    def recommend(self, query='', location='', cuisine='', max_cost=None, min_rating=0.0,
                  online_order=None, book_table=None, top_n=10, exclude_name='') -> pd.DataFrame:
        df = self.restaurants.copy()
        mask = pd.Series(True, index=df.index)
        if location and location != 'All':
            mask &= df['location'].str.contains(re.escape(location), case=False, na=False)
        if cuisine:
            mask &= df['cuisines'].str.contains(re.escape(cuisine), case=False, na=False)
        if max_cost is not None:
            mask &= df['cost_for_two'].isna() | (df['cost_for_two'] <= max_cost)
        if min_rating > 0:
            mask &= df['rating'].fillna(0) >= min_rating
        if online_order in ('Yes','No'):
            mask &= df['online_order'].str.casefold().eq(online_order.casefold())
        if book_table in ('Yes','No'):
            mask &= df['book_table'].str.casefold().eq(book_table.casefold())
        if exclude_name:
            mask &= df['name'].str.casefold().ne(exclude_name.casefold())
        candidates = df.loc[mask].copy()
        if candidates.empty:
            return candidates
        # Popularity-aware baseline score. Ratings/votes are not treated as ground-truth relevance.
        candidates['rating_score'] = candidates['rating'].fillna(0).clip(0,5) / 5.0
        candidates['popularity_score'] = np.log1p(candidates['votes'].fillna(0))
        pop_max = candidates['popularity_score'].max()
        candidates['popularity_score'] = candidates['popularity_score'] / pop_max if pop_max else 0
        candidates['similarity_score'] = 0.0
        q = (query or '').strip()
        if q and self.matrix is not None:
            query_vec = self.vectorizer.transform([q])
            sims = cosine_similarity(query_vec, self.matrix).ravel()
            candidates['similarity_score'] = candidates.index.map(lambda i: float(sims[i]))
            candidates['recommendation_score'] = (0.60*candidates['similarity_score'] + 0.30*candidates['rating_score'] + 0.10*candidates['popularity_score'])
        else:
            candidates['recommendation_score'] = (0.75*candidates['rating_score'] + 0.25*candidates['popularity_score'])
        return candidates.sort_values(['recommendation_score','votes'], ascending=False).head(int(top_n)).reset_index(drop=True)

    def similar_to(self, restaurant_name, top_n=10) -> pd.DataFrame:
        matches = self.restaurants.index[self.restaurants['name'].str.casefold().eq(restaurant_name.casefold())].tolist()
        if not matches or self.matrix is None:
            return self.restaurants.iloc[0:0].copy()
        idx = matches[0]
        sims = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        out = self.restaurants.copy()
        out['similarity_score'] = sims
        out = out[out.index != idx].sort_values('similarity_score', ascending=False).head(top_n)
        return out.reset_index(drop=True)
