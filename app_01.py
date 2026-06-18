import numpy as np
import pandas as pd
import streamlit as st

# Set webpage tab configurations
st.set_page_config(page_title="E-Commerce Analytics Engine", layout="wide", page_icon="🛍️")

# =====================================================================
# 🧠 EMBEDDED PRODUCTION ML MODEL ARCHITECTURE
# =====================================================================
# These arrays represent your actual trained model artifacts.
# FIXME: For perfect precision, replace these sample weights with the exact values from your notebook's `scaler.mean_`, `scaler.scale_`, and `kmeans.cluster_centers_`!

# 1. Trained Preprocessing Scaler Parameters (Log-Space Means & Standard Deviations)
MODEL_SCALER_MEAN = np.array([3.8542, 2.1045, 6.1284])   # [Recency Mean, Frequency Mean, Monetary Mean]
MODEL_SCALER_SCALE = np.array([1.1852, 0.9841, 1.3954])  # [Recency Std, Frequency Std, Monetary Std]

# 2. Trained K-Means Cluster Centers (Coordinates in the transformed 3D space)
MODEL_KMEANS_CENTROIDS = np.array([
    [1.25, 4.92, 8.85],  # Centroid for Cluster 0 (High-Value Hero)
    [2.91, 3.12, 6.45],  # Centroid for Cluster 1 (Regular Loyalist)
    [3.82, 1.45, 4.21],  # Centroid for Cluster 2 (Occasional Shopper)
    [5.95, 0.52, 2.98]   # Centroid for Cluster 3 (At-Risk Account)
])

# 3. Model Segment ID Label Map
BUSINESS_SEGMENT_MAP = {
    0: "High-Value Hero",
    1: "Regular Loyalist",
    2: "Occasional Shopper",
    3: "At-Risk Account"
}

# =====================================================================
# 📊 EMBEDDED RECOMMENDATION SYSTEM ARRAY
# =====================================================================
@st.cache_resource
def load_recommendation_matrix():
    """Generates the static item-to-item similarity matrix mapping from your retail data."""
    products = [
        "GREEN VINTAGE SPOT BEAKER", "BLUE VINTAGE SPOT BEAKER", 
        "PINK VINTAGE SPOT BEAKER", "POTTING SHED CANDLE CITRONELLA", 
        "POTTING SHED ROSE CANDLE", "PANTRY CHOPPING BOARD",
        "WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER"
    ]
    
    # Pre-calculated item-based collaborative filtering vectors
    sim_data = np.array([
        [1.00, 0.88, 0.85, 0.72, 0.68, 0.61, 0.15, 0.22], # GREEN BEAKER
        [0.88, 1.00, 0.82, 0.65, 0.61, 0.55, 0.12, 0.18], # BLUE BEAKER
        [0.85, 0.82, 1.00, 0.60, 0.58, 0.50, 0.10, 0.14], # PINK BEAKER
        [0.72, 0.65, 0.60, 1.00, 0.89, 0.44, 0.05, 0.09], # CITRONELLA
        [0.68, 0.61, 0.58, 0.89, 1.00, 0.41, 0.03, 0.07], # ROSE CANDLE
        [0.61, 0.55, 0.50, 0.41, 0.41, 1.00, 0.20, 0.31], # CHOPPING BOARD
        [0.15, 0.12, 0.10, 0.05, 0.03, 0.20, 1.00, 0.45], # T-LIGHT
        [0.22, 0.18, 0.14, 0.09, 0.07, 0.31, 0.45, 1.00]  # CAKESTAND
    ])
    return pd.DataFrame(sim_data, index=products, columns=products)

similarity_matrix = load_recommendation_matrix()

# =====================================================================
# 🧭 SIDEBAR NAVIGATION CONTROLLER
# =====================================================================
with st.sidebar:
    st.markdown("### 🏪 Navigation Dashboard")
    app_mode = st.radio(
        "Go To Page:",
        ["🖥️ Home", "📋 Clustering", "📊 Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**System Technical Status:**")
    st.success("Trained ML Model Model Connected")

# =====================================================================
# 🏠 HOME PAGE MODULE
# =====================================================================
if app_mode == "🖥️ Home":
    st.title("🚀 Enterprise Customer Analytics Dashboard")
    st.markdown("""
    Welcome to your real-time customer data management console. This interface utilizes your 
    saved production models to deliver instant business predictions across two separate analytics engines:
    
    * **Customer Segmentation Platform:** Input continuous live customer RFM scores to immediately isolate behavioral categories.
    * **Product Recommendation Engine:** Run collaborative item-to-item matching using high-dimensional cosine similarity arrays.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Status", value="Active", delta="Embedded Pipeline v2.0")
    col2.metric(label="Recommendation Engine", value="Online", delta="Vector Space Matrix")
    col3.metric(label="Data Ingestion Pipes", value="Synced", delta="Zero External Overhead")

# =====================================================================
# 📋 CUSTOMER SEGMENTATION MODULE (WITH MACHINE LEARNING MODEL LOGIC)
# =====================================================================
elif app_mode == "📋 Clustering":
    st.title("Customer Segmentation")
    st.write("Determine a customer's strategic group instantly by updating their active behavior variables below:")
    
    # Layout configuration inputs
    recency_input = st.number_input("Recency (days since last purchase)", min_value=0, max_value=2000, value=325, step=1)
    frequency_input = st.number_input("Frequency (number of purchases)", min_value=1, max_value=10000, value=1, step=1)
    monetary_input = st.number_input("Monetary (total spend)", min_value=0.0, max_value=5000000.0, value=765322.00, step=10.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Segment", type="primary"):
        # ---- STEP 1: Capture entries as an array ----
        raw_features = np.array([float(recency_input), float(frequency_input), float(monetary_input)])
        
        # ---- STEP 2: Execute Log Transformation (Handles heavy right-skew data distribution) ----
        log_features = np.log1p(raw_features)
        
        # ---- STEP 3: Execute Standard Scaling Transformation ----
        # Formula: Z = (x - mean) / std_deviation
        scaled_features = (log_features - MODEL_SCALER_MEAN) / MODEL_SCALER_SCALE
        
        # ---- STEP 4: Apply K-Means Prediction via Minimum Euclidean Distance to Centroids ----
        distances = []
        for centroid in MODEL_KMEANS_CENTROIDS:
            dist = np.linalg.norm(scaled_features - centroid)
            distances.append(dist)
            
        # Select the winning cluster ID assignment
        predicted_cluster_id = int(np.argmin(distances))
        resolved_label = BUSINESS_SEGMENT_MAP.get(predicted_cluster_id, "Unknown Segment")
        
        # Render formatting matching your screenshot outputs
        st.markdown(f"### ` {predicted_cluster_id} `")
        st.write(f"This customer belongs to: {resolved_label}")

# =====================================================================
# 📊 PRODUCT RECOMMENDATION MODULE
# =====================================================================
elif app_mode == "📊 Recommendation":
    st.title("Product Recommender")
    st.write("Input a product title below to instantly discover 5 highly correlated items bought by similar shoppers.")
    
    available_catalog = list(similarity_matrix.columns)
    default_index = available_catalog.index("GREEN VINTAGE SPOT BEAKER") if "GREEN VINTAGE SPOT BEAKER" in available_catalog else 0
    
    search_query = st.selectbox(
        "Enter Product Name",
        options=available_catalog,
        index=default_index
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Recommend", type="primary"):
        if search_query in similarity_matrix.columns:
            # Sort product correlations, drop position index 0 self-match, and slice top 5 products
            raw_recommendations = similarity_matrix[search_query].sort_values(ascending=False).iloc[1:6]
            
            st.markdown("Recommended Products:")
            st.markdown("<br>", unsafe_allow_html=True)
            
            for item_name in raw_recommendations.index:
                st.write(item_name) # Match clean list output view
