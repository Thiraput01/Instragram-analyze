import streamlit as st
import joblib
import pandas as pd

# Title and description
st.write("""# Follow Predictor""")
st.write("This is a simple web app to predict the follow count of a user based on some features.")

# Sidebar header
st.sidebar.header('User Input Features')

# Load the model and scaler
model = joblib.load('follow_predict/random_forest_model.pkl')  # Load the trained Random Forest model
scaler = joblib.load('follow_predict/scaler.pkl')  # Load the scaler that was used for training

def user_input():
    # Slider inputs for each feature with unique keys
    impressions = st.sidebar.slider('Impressions', min_value=0, max_value=50000, value=25000, key='impressions')
    from_home = st.sidebar.slider('From Home', min_value=0, max_value=20000, value=10000, key='from_home')
    from_hashtags = st.sidebar.slider('From Hashtags', min_value=0, max_value=20000, value=10000, key='from_hashtags')
    from_explore = st.sidebar.slider('From Explore', min_value=0, max_value=20000, value=10000, key='from_explore')
    from_other = st.sidebar.slider('From Other', min_value=0, max_value=5000, value=2500, key='from_other')
    saves = st.sidebar.slider('Saves', min_value=0, max_value=2000, value=1000, key='saves')
    shares = st.sidebar.slider('Shares', min_value=0, max_value=100, value=50, key='shares')
    likes = st.sidebar.slider('Likes', min_value=0, max_value=1000, value=500, key='likes')
    profile_visits = st.sidebar.slider('Profile Visits', min_value=0, max_value=500, value=250, key='profile_visits')

    # Display user inputs
    st.write("## Selected Input Values:")
    st.write(f"Impressions: {impressions}")
    st.write(f"From Home: {from_home}")
    st.write(f"From Hashtags: {from_hashtags}")
    st.write(f"From Explore: {from_explore}")
    st.write(f"From Other: {from_other}")
    st.write(f"Saves: {saves}")
    st.write(f"Shares: {shares}")
    st.write(f"Likes: {likes}")
    st.write(f"Profile Visits: {profile_visits}")
    return impressions, from_home, from_hashtags, from_explore, from_other, saves, shares, likes, profile_visits

def model_output(impressions, from_home, from_hashtags, from_explore, from_other, saves, shares, likes, profile_visits):
    # Prepare the input data
    input_data = [[impressions, from_home, from_hashtags, from_explore, from_other, saves, shares, likes, profile_visits]]
    
    # Normalize the input data using the scaler
    input_scaled = scaler.transform(input_data)  # Use the scaler that was used during training
    
    # Make predictions using the model
    predicted_output = model.predict(input_scaled)
    
    # Return the predicted output without inverse_transform
    return predicted_output


# Get user input
impressions, from_home, from_hashtags, from_explore, from_other, saves, shares, likes, profile_visits = user_input()

# Button to trigger prediction
if st.button('Predict'):
    output = model_output(impressions, from_home, from_hashtags, from_explore, from_other, saves, shares, likes, profile_visits)
    st.write(f"Predicted Follow Count: {output[0]}")
else:
    st.write("Click the 'Predict' button to get the prediction.")
