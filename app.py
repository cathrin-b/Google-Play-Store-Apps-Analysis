import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Google Play Store Analysis", layout="wide", page_icon="📱")

# --- 2. DATA CLEANING ---
@st.cache_data
def process_data(file_path):
    df = pd.read_csv(file_path)
    # Remove anomaly row
    df = df[df['Category'] != '1.9']

    # Clean Reviews, Installs, and Price
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Installs'] = df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    # Clean Size
    def convert_size(size):
        size = str(size)
        if 'M' in size: return float(size.replace('M', ''))
        if 'k' in size: return float(size.replace('k', '')) / 1024
        return np.nan

    df['Size'] = df['Size'].apply(convert_size)
    df['Size'] = df['Size'].fillna(df['Size'].median())

    # Drop missing values and duplicates
    df = df.dropna(subset=['Rating', 'Type', 'Content Rating', 'Installs', 'Price'])
    df = df.drop_duplicates(subset=['App'])
    
    return df

# --- 3. ML MODEL TRAINING ---
@st.cache_resource
def train_model(df):
    le_cat = LabelEncoder()
    df['Category_Encoded'] = le_cat.fit_transform(df['Category'])
    
    le_content = LabelEncoder()
    df['Content_Rating_Encoded'] = le_content.fit_transform(df['Content Rating'])

    X = df[['Category_Encoded', 'Reviews', 'Size', 'Installs', 'Price', 'Content_Rating_Encoded']]
    y = df['Rating']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_cat, le_content

# --- 4. LOAD DATA AND MODEL ---
DATA_FILE = 'googleplaystore..CSV.csv' 

if os.path.exists(DATA_FILE):
    df_clean = process_data(DATA_FILE)
    model, le_cat, le_content = train_model(df_clean)
else:
    st.error(f"Dataset not found! Please ensure '{DATA_FILE}' is in the same folder.")
    st.stop()

# --- 5. SIDEBAR (LOGO & NAVIGATION) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d0/Google_Play_Arrow_logo.svg", width=60)
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Select a Page", ["Data Insights", "Rating Predictor", "About Project"])

# --- 6. PAGE: DATA INSIGHTS ---
if page == "Data Insights":
    st.title("📊 Google Play Store App Insights")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Apps", len(df_clean))
    kpi2.metric("Avg Rating", f"{df_clean['Rating'].mean():.2f} ⭐")
    kpi3.metric("Top Category", df_clean['Category'].mode()[0])

    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 App Categories")
        cat_counts = df_clean['Category'].value_counts().head(15).reset_index()
        cat_counts.columns = ['Category_Name', 'Count'] 
        fig_cat = px.bar(cat_counts, x='Category_Name', y='Count', color='Category_Name',
                         labels={'Category_Name': 'Category', 'Count': 'Number of Apps'})
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        st.subheader("Price vs. Rating")
        paid_apps = df_clean[df_clean['Price'] > 0]
        if not paid_apps.empty:
            fig_price = px.scatter(paid_apps, x="Price", y="Rating", size="Installs", 
                                 color="Category", hover_name="App", log_x=True)
            st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Installation Volume per Content Rating")
    fig_inst = px.box(df_clean, x="Content Rating", y="Installs", color="Content Rating", log_y=True)
    st.plotly_chart(fig_inst, use_container_width=True)

# --- 7. PAGE: RATING PREDICTOR ---
elif page == "Rating Predictor":
    st.title("🤖 Rating Prediction Model")
    st.write("Fill in the details below and click 'Predict Rating' to see the result.")

    with st.form(key="prediction_form"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            input_cat = st.selectbox("App Category", le_cat.classes_)
            input_rev = st.number_input("Number of Reviews", min_value=0, value=1000)
            input_size = st.number_input("App Size (MB)", min_value=0.1, value=15.0)
            
        with col_b:
            input_inst = st.number_input("Total Installs", min_value=0, value=10000)
            input_price = st.number_input("Price ($)", min_value=0.0, value=0.0)
            input_content = st.selectbox("Content Rating", le_content.classes_)

        submitted = st.form_submit_button("Predict Rating")

    if submitted:
        cat_encoded = le_cat.transform([input_cat])[0]
        content_encoded = le_content.transform([input_content])[0]
        features = np.array([[cat_encoded, input_rev, input_size, input_inst, input_price, content_encoded]])
        prediction = model.predict(features)[0]
        
        st.divider()
        st.success(f"### Predicted Rating: {prediction:.2f} ⭐")
        st.balloons()

# --- 8. PAGE: ABOUT PROJECT ---
elif page == "About Project":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Objective
    This application is designed to analyze the Google Play Store dataset to identify key success factors for mobile apps. 
    By combining **Data Exploration** and **Machine Learning**, we provide a tool that predicts app ratings based on 
    market parameters.
    """)

    

    st.markdown("""
    ### Technical Overview
    - **Language:** Python
    - **UI Framework:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly Express
    - **ML Model:** Random Forest Regressor (Scikit-Learn)
    
    ### How it Works
    The model was trained on over 8,000 unique apps. It uses **Feature Importance** to weigh how much an app's 
    Category, Size, and Pricing impact its final User Rating.
    
    ---
    ### 📜 Copyright & Credits
    - **Developer:** Cathrin Prasalya
    - **Dataset Source:** Kaggle (Google Play Store Apps by Lava18)
    - **Year:** 2024
    
    **Disclaimer:** This project is for educational purposes. Google Play and the Google Play logo are trademarks 
    of Google LLC.
    """)

# --- 9. FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 | Google Play Store Analysis -Developed by Cathrin Prasalya")