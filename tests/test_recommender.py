import pandas as pd
from src.recommender import RestaurantRecommender

def sample_data():
    return pd.DataFrame([
        {"name":"Curry House","location":"Indiranagar","cuisines":"North Indian, Curry","rest_type":"Casual Dining","dish_liked":"Biryani","listed_type":"Delivery","listed_city":"Indiranagar","search_text":"Curry House Indiranagar North Indian Curry Biryani","rating":4.4,"votes":120,"cost_for_two":800,"online_order":"Yes","book_table":"No"},
        {"name":"Pasta Corner","location":"Koramangala","cuisines":"Italian, Pizza","rest_type":"Cafe","dish_liked":"Pasta","listed_type":"Delivery","listed_city":"Koramangala","search_text":"Pasta Corner Koramangala Italian Pizza Pasta","rating":4.0,"votes":80,"cost_for_two":900,"online_order":"Yes","book_table":"Yes"},
        {"name":"Budget Curry","location":"Indiranagar","cuisines":"North Indian","rest_type":"Quick Bites","dish_liked":"Thali","listed_type":"Dine-out","listed_city":"Indiranagar","search_text":"Budget Curry Indiranagar North Indian Thali","rating":3.8,"votes":50,"cost_for_two":300,"online_order":"No","book_table":"No"},
    ])

def test_filters_and_results():
    rec = RestaurantRecommender(sample_data())
    result = rec.recommend(location='Indiranagar', max_cost=500, top_n=10)
    assert len(result) == 1
    assert result.iloc[0]['name'] == 'Budget Curry'

def test_query_returns_ranked_results():
    rec = RestaurantRecommender(sample_data())
    result = rec.recommend(query='Italian pasta', top_n=2)
    assert len(result) == 2
    assert result.iloc[0]['name'] == 'Pasta Corner'
