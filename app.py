from pathlib import Path
import sys
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_restaurants
from src.recommender import RestaurantRecommender

st.set_page_config(page_title="Gaurav's Restaurant Recommendation System", page_icon="🍽️", layout="wide")
st.title("Gaurav's Restaurant Recommendation System")
# st.caption("Content-based restaurant discovery using the supplied Zomato Bengaluru dataset. Historical data, not live restaurant availability.")

@st.cache_data(show_spinner="Loading and preparing the dataset…")
def get_data():
    return load_restaurants()

@st.cache_resource(show_spinner="Building recommendation index…")
def get_model(df):
    return RestaurantRecommender(df)

try:
    df = get_data()
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.info("Ensure data/raw/zomato.csv.zip or data/raw/zomato.csv exists in the project folder.")
    st.stop()
model = get_model(df)

with st.sidebar:
    st.header("Your preferences")
    locations = sorted([x for x in df['location'].dropna().unique() if str(x).strip()])
    location = st.selectbox("Area", ["All"] + locations)
    cuisine_options = sorted({part.strip() for value in df['cuisines'].fillna('') for part in str(value).split(',') if part.strip()})
    cuisine = st.selectbox("Cuisine", ["Any"] + cuisine_options)
    budget = st.number_input("Maximum cost for two (₹)", min_value=0, max_value=20000, value=1500, step=100)
    min_rating = st.slider("Minimum rating", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
    order = st.selectbox("Online ordering", ["Any", "Yes", "No"])
    table = st.selectbox("Table booking", ["Any", "Yes", "No"])
    top_n = st.slider("Number of recommendations", 5, 25, 10)

query = st.text_input("What are you looking for?", placeholder="e.g. family dinner, biryani, quiet cafe, North Indian Chinese")
tab1, tab2 = st.tabs(["Recommendations", "Explore dataset"])
# tab1, tab2, tab3 = st.tabs(["Recommendations", "Explore dataset", "About & limitations"])  --old version
with tab1:
    recs = model.recommend(query=query, location=location, cuisine='' if cuisine == 'Any' else cuisine,
                           max_cost=budget if budget > 0 else None, min_rating=min_rating,
                           online_order=None if order == 'Any' else order,
                           book_table=None if table == 'Any' else table, top_n=top_n)
    st.subheader(f"Recommended restaurants ({len(recs)})")
    if recs.empty:
        st.warning("No matches found. Try widening the area, cuisine, budget or rating filters.")
    else:
        for _, r in recs.iterrows():
            with st.container(border=True):
                c1,c2,c3 = st.columns([3,2,1])
                c1.markdown(f"**{r.get('name','Unknown')}**")
                c1.caption(f"{r.get('location','')} · {r.get('rest_type','')}")
                c2.write(f"Cuisine: {r.get('cuisines','Not listed') or 'Not listed'}")
                c2.write(f"Cost for two: ₹{r['cost_for_two']:,.0f}" if pd.notna(r.get('cost_for_two')) else "Cost: Not listed")
                rating = r.get('rating')
                c3.metric("Rating", f"{rating:.1f}/5" if pd.notna(rating) else "N/A")
                c3.caption(f"Votes: {int(r.get('votes',0)):,}")
                if r.get('url'):
                    c1.markdown(f"[Open listing]({r['url']})")
with tab2:
    a,b,c,d = st.columns(4)
    a.metric("Listings after deduplication", f"{len(df):,}")
    b.metric("Areas", f"{df['location'].nunique():,}")
    c.metric("Cuisine entries", f"{df['cuisines'].nunique():,}")
    d.metric("Online order: Yes", f"{(df['online_order'].str.casefold() == 'yes').sum():,}")
    st.markdown("#### Top areas by listing count")
    st.bar_chart(df['location'].replace('', 'Unknown').value_counts().head(15))
    st.markdown("#### Rating distribution")
    st.bar_chart(df['rating'].dropna().round(1).value_counts().sort_index())
    st.markdown("#### Cost for two distribution")
    st.bar_chart(df['cost_for_two'].dropna().clip(upper=5000).round(-2).value_counts().sort_index())
