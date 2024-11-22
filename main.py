import os
os.environ['GDAL_CONFIG'] = '/usr/bin/gdal-config'

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import geopandas as gpd
from textblob import TextBlob



@st.cache_data
def load_data():
    try:
        listings = pd.read_csv('listings.csv.gz', compression='gzip')
        calendar = pd.read_csv('calendar.csv.gz', compression='gzip')
        reviews = pd.read_csv('reviews.csv.gz', compression='gzip')
        neighborhoods = pd.read_csv('neighbourhoods.csv')
        geojson = gpd.read_file('neighbourhoods.geojson')
        return listings, calendar, reviews, neighborhoods, geojson
    except ImportError as e:
        st.error(f"Error loading GeoPandas or Fiona modules: {e}")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


listings, calendar, reviews, neighborhoods, neighborhoods_geo = load_data()

if listings is not None:
    listings['price'] = listings['price'].replace('[\$,]', '', regex=True).astype(float)
    calendar['price'] = calendar['price'].replace('[\$,]', '', regex=True).astype(float)
    average_price_neighborhood = listings.groupby('neighbourhood_cleansed')['price'].mean().round(2).sort_values(ascending=False)
    property_type_distribution = listings.groupby(['neighbourhood_cleansed', 'property_type']).size().unstack(
        fill_value=0)

    def filter_listings_by_neighborhood(neighborhood, room_type='Entire home/apt'):
        filtered = listings[
            (listings['neighbourhood_cleansed'] == neighborhood) &
            (listings['room_type'] == room_type)
            ]
        return filtered

    # Sentiment analysis function based on AI code. See Section 1 of AI Report.

    def analyze_sentiment(comments):
        sentiments = comments.dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
        return pd.cut(sentiments, bins=[-1, -0.1, 0.1, 1], labels=["Negative", "Neutral", "Positive"])

    reviews = reviews.merge(
        listings[['id', 'neighbourhood_cleansed', 'name']],
        left_on='listing_id',
        right_on='id',
        how='left'
    )

    price_data = listings.groupby('neighbourhood_cleansed').agg({
        'price': 'mean',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    price_data['neighbourhood_cleansed'] = price_data['neighbourhood_cleansed'].str.strip().str.lower()
    neighborhoods_geo['neighbourhood'] = neighborhoods_geo['neighbourhood'].str.strip().str.lower()
    reviews['neighbourhood_cleansed'] = reviews['neighbourhood_cleansed'].str.strip().str.lower()

    st.title("Boston Airbnb: Explore Listings & Trends")

    st.header("Sentiment Analysis of Reviews")
    neighborhoods_display_reviews = [n.title() for n in reviews['neighbourhood_cleansed'].dropna().unique()]
    neighborhoods_mapping_reviews = {n.title(): n for n in reviews['neighbourhood_cleansed'].dropna().unique()}

    selected_neighborhood_reviews = st.selectbox(
        "Select a neighborhood to analyze sentiment:",
        options=neighborhoods_display_reviews
    )

    selected_neighborhood_lower = neighborhoods_mapping_reviews[selected_neighborhood_reviews]
    neighborhood_reviews = reviews[reviews['neighbourhood_cleansed'] == selected_neighborhood_lower]

    if not neighborhood_reviews.empty:
        sentiments = analyze_sentiment(neighborhood_reviews['comments'])
        sentiment_counts = sentiments.value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)

   # Bar chart for sentiment distribution based on AI-generated code. See Section 3 of AI Report.

        fig, ax = plt.subplots(figsize=(6, 4))
        sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'], ax=ax)
        ax.set_title("Sentiment Distribution", fontsize=16)
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        st.pyplot(fig)
    else:
        st.write("No reviews available for this neighborhood.")


    st.header("Neighborhood Price Insights")
    neighborhoods_display_price = [n.title() for n in average_price_neighborhood.index]
    neighborhoods_mapping_price = {n.title(): n for n in average_price_neighborhood.index}

    selected_neighborhood = st.selectbox(
        "Choose a neighborhood to view average prices:",
        options=["All"] + neighborhoods_display_price,
        index=0
    )

    if selected_neighborhood == "All":
        st.bar_chart(average_price_neighborhood)
    else:
        selected_neighborhood_lower = neighborhoods_mapping_price[selected_neighborhood]
        avg_price = average_price_neighborhood.get(selected_neighborhood_lower, None)
        if pd.notnull(avg_price):
            st.write(f"The average listing price in **{selected_neighborhood}** is **${avg_price:.2f}**.")
        else:
            st.write("No data available for the selected neighborhood.")

    st.header("Top 5 Most Expensive Neighborhoods")
    top_5_neighborhoods = average_price_neighborhood.head(5)
    st.write(top_5_neighborhoods)
    st.bar_chart(top_5_neighborhoods)

    st.header("Property Types Across Neighborhoods")
    neighborhoods_display_property = [n.title() for n in property_type_distribution.index]
    neighborhoods_mapping_property = {n.title(): n for n in property_type_distribution.index}

    selected_neighborhoods_display = st.multiselect(
        "Select neighborhoods to compare property types:",
        options=neighborhoods_display_property,
        default=neighborhoods_display_property[:3]
    )

    if selected_neighborhoods_display:
        selected_neighborhoods_lower = [neighborhoods_mapping_property[n] for n in selected_neighborhoods_display]
        try:
            filtered_distribution = property_type_distribution.loc[selected_neighborhoods_lower]
            st.bar_chart(filtered_distribution.T)
        except KeyError as e:
            st.error(f"Selected neighborhoods not found in the data: {e}")
    else:
        st.write("Please select at least one neighborhood.")

    st.header("Property Type Distribution Across Neighborhoods (Pivot Table)")
    property_pivot = listings.pivot_table(
        index='neighbourhood_cleansed',
        columns='property_type',
        values='id',
        aggfunc='count',
        fill_value=0
    )
    st.write(property_pivot)
    # Map Uses Recommendations from AI. See Section 4 of AI Report.
    st.header("Listing Locations on the Map")
    map_data = listings[['latitude', 'longitude', 'price']].dropna()
    price_min = int(map_data['price'].min())
    price_max = int(map_data['price'].max())
    price_range = st.slider("Filter by price range ($):", price_min, price_max, (50, 500))

    filtered_map_data = map_data[
        (map_data['price'] >= price_range[0]) & (map_data['price'] <= price_range[1])
    ]

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=42.3601,
            longitude=-71.0589,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_map_data,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=100,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                'GeoJsonLayer',
                data=neighborhoods_geo,
                get_fill_color='[100, 150, 200, 50]',
                get_line_color='[255, 255, 255, 200]',
                line_width_min_pixels=1,
                pickable=False,
            ),
        ],
        tooltip={
            "html": "<b>Price:</b> ${price}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    ))

    st.header("Neighborhood Price Choropleth")
    st.write("Visualize average listing prices across neighborhoods using a choropleth map.")

    price_data['price_category'] = pd.cut(
        price_data['price'],
        bins=[0, 100, 200, 300, 400, 500, price_data['price'].max()],
        labels=[1, 2, 3, 4, 5, 6]
    ).astype(int)

    color_map = {
        1: [50, 205, 50],
        2: [173, 255, 47],
        3: [255, 255, 0],
        4: [255, 165, 0],
        5: [255, 69, 0],
        6: [255, 0, 0]
    }

    price_data['color'] = price_data['price_category'].map(color_map)
    price_data['color'] = price_data['color'].apply(lambda x: x if isinstance(x, list) else [128, 128, 128])

    try:
        merged_geo = neighborhoods_geo.merge(
            price_data,
            left_on='neighbourhood',
            right_on='neighbourhood_cleansed',
            how='left'
        )

        merged_geo['color'] = merged_geo['color'].apply(lambda x: x if isinstance(x, list) else [128, 128, 128])
        merged_geo['price'] = merged_geo['price'].apply(lambda x: round(x, 2) if pd.notnull(x) else None)

        # Choropleth map setup based on AI-generated code. See Section 2 of AI Report.

        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            data=merged_geo,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="[color[0], color[1], color[2], 200]",
            get_line_color="[255, 255, 255, 150]",
            line_width_min_pixels=1,
            auto_highlight=True,
        )

        choropleth_deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=42.3601,
                longitude=-71.0589,
                zoom=11,
                pitch=50
            ),
            layers=[geojson_layer],
            tooltip={
                "html": "<b>Neighborhood:</b> {neighbourhood}<br/>"
                        "<b>Average Price:</b> ${price}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
        )

        st.pydeck_chart(choropleth_deck)
    except Exception as e:
        st.error(f"Error rendering the choropleth map: {e}")

    st.markdown(
        "Dive into Boston's Airbnb data with this interactive tool. Discover trends, compare neighborhoods, and explore property types."
    )
else:
    st.error("Data could not be loaded. Please check the data files and try again.")

