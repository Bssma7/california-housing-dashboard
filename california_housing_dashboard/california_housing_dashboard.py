
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  # ← هذا الخط


# --- Data Loading and Setup (Run once) ---
@st.cache_data
def load_data_and_setup():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Split data (Needed for model training)
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prepare scaled data for PCA/KMeans
    X_cluster = df.drop(columns=['MedHouseVal']) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Compute PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca_2d = pca.fit_transform(X_scaled)
    
    # Get the mean of the training features for prediction defaults
    X_train_mean = X_train.mean()
    
    return df, X_scaled, model, X_train_mean, X_pca_2d

df, X_scaled, model, X_train_mean, X_pca_2d = load_data_and_setup()

# --- Streamlit App ---
st.title("Interactive Dashboard for California Housing Analysis")

# --- Regression Section ---
st.subheader("1. Regression: Predict Median House Value (Linear Model)")

col1, col2 = st.columns(2)
with col1:
    medinc = st.slider("Median Income (MedInc)", float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].median()))
with col2:
    houseage = st.slider("House Age (HouseAge)", int(df['HouseAge'].min()), int(df['HouseAge'].max()), int(df['HouseAge'].median()))

# Create the input array with 8 features for prediction
input_data = X_train_mean.copy()
input_data['MedInc'] = medinc
input_data['HouseAge'] = houseage

# Prediction
pred_val = model.predict(input_data.to_frame().T)
st.markdown(f"**Predicted Median House Value:** **${pred_val[0]*100000:.2f}** (USD)")
st.caption("All other features are held at their mean training value.")

# --- Clustering Section ---
st.subheader("2. Clustering Visualization")

st.sidebar.header("Clustering Settings")
# Interactive widget: choose k
n_clusters = st.sidebar.slider("Select number of clusters (k):", 2, 8, 4)

# Apply KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 1. Visualization in PCA space
st.markdown(f"#### Clusters in PCA Space (k={n_clusters})")
df_plot = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'])
df_plot['Cluster'] = labels

fig1, ax1 = plt.subplots(figsize=(7, 5))
scatter1 = ax1.scatter(df_plot['PC1'], df_plot['PC2'], c=df_plot['Cluster'], cmap='viridis', alpha=0.6)
ax1.set_xlabel("Principal Component 1 (PC1)")
ax1.set_ylabel("Principal Component 2 (PC2)")
fig1.colorbar(scatter1, label='Cluster ID')
st.pyplot(fig1)

# 2. Visualization on map (Longitude, Latitude)
st.markdown(f"#### Clusters on California Map (k={n_clusters})")
fig2, ax2 = plt.subplots(figsize=(7, 5))
scatter2 = ax2.scatter(df['Longitude'], df['Latitude'], c=labels, cmap='viridis', alpha=0.6)
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
fig2.colorbar(scatter2, label='Cluster ID')
st.pyplot(fig2)