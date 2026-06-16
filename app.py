import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import gdown

# Set webpage tab configurations
st.set_page_config(page_title="E-Commerce Analytics Engine", layout="wide", page_icon="🛍️")

import os
import pickle
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

# Set webpage tab configurations
st.set_page_config(page_title="E-Commerce Analytics Engine", layout="wide", page_icon="🛍️")

# =====================================================================
# 💾 ROBUST BYPASS-LINK ASSET DOWNLOAD PIPELINE
# =====================================================================
def download_large_gdrive_file(file_id, destination):
    """Downloads large binaries from GDrive, bypassing the large-file warning page."""
    if not os.path.exists(destination):
        with st.spinner(f"📥 Downloading required deployment asset: {destination}..."):
            # The 'confirm=t' flag tells Google to skip the "cannot scan for viruses" HTML screen
            url = f"https://docs.google.com/uc?export=download&id={file_id}&confirm=t"
            
            # Pretend to be a clean browser request to avoid catching an anti-bot HTML page
            request_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            try:
                req = urllib.request.Request(url, headers=request_headers)
                with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                st.error(f"❌ Network download error on {destination}: {e}")

# Initialize global memory pointers
loaded_scaler = None
loaded_kmeans = None
similarity_matrix = None

# Step 1: Run the raw binary download streaming sequence
try:
    if "gdrive_file_ids" in st.secrets:
        download_large_gdrive_file(st.secrets["gdrive_file_ids"]["scaler_id"], "scaler.pkl")
        download_large_gdrive_file(st.secrets["gdrive_file_ids"]["kmeans_id"], "kmeans_model.pkl")
        download_large_gdrive_file(st.secrets["gdrive_file_ids"]["matrix_id"], "recommendation_matrix.pkl")
    else:
        st.error("❌ 'gdrive_file_ids' section missing from Streamlit Cloud Secrets Manager configuration!")
except Exception as e:
    st.error(f"❌ Download handshake execution failed: {e}")

# High-efficiency resource caching to protect server memory limits
@st.cache_resource
def load_production_artifacts():
    if os.path.exists('scaler.pkl') and os.path.exists('kmeans_model.pkl') and os.path.exists('recommendation_matrix.pkl'):
        # Check if the file is tiny (corrupted HTML is usually under 50KB, a true matrix is MBs)
        if os.path.getsize('recommendation_matrix.pkl') < 50000:
            st.error("❌ The downloaded asset is too small. Google Drive is still serving an HTML page instead of raw bytes.")
            return None, None, None
            
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        with open('recommendation_matrix.pkl', 'rb') as f:
            matrix = pickle.load(f)
        return scaler, kmeans, matrix
    return None, None, None

# Step 2: Try reading the binary assets into memory
try:
    loaded_scaler, loaded_kmeans, similarity_matrix = load_production_artifacts()
except Exception as e:
    st.error(f"❌ Error loading pickle files into memory: {e}")

# Step 3: Global Safety Circuit Breaker
if loaded_scaler is None or loaded_kmeans is None or similarity_matrix is None:
    st.warning("⚠️ **Application is paused:** The required ML models or recommendation matrices could not be loaded.")
    st.info("💡 **How to resolve this right now:** Make sure your Google Drive files are set to **'Anyone with the link'** and check that you haven't copied the whole URL into Streamlit secrets, just the string of letters/numbers!")
    st.stop()
    
# =====================================================================
# 🧭 SIDEBAR NAVIGATION CONTROLLER (Matching UI Screenshot)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏪 Navigation Dashboard")
    # Using streamlit native radio styled creatively or option_menu alternative
    app_mode = st.radio(
        "Go To Page:",
        ["🖥️ Home", "📋 Clustering", "📊 Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**System Technical Status:**")
    st.success("All Core ML Models Connected")

# =====================================================================
# 🏠 HOME PAGE NAVIGATION MODULE
# =====================================================================
if app_mode == "🖥️ Home":
    st.title("🚀 Enterprise Customer Analytics Dashboard")
    st.markdown("""
    Welcome to your real-time customer data management console. This interface utilizes your 
    saved production models to deliver instant business predictions across two separate analytics engines:
    
    * **Customer Segmentation Platform:** Input continuous live customer RFM scores to immediately isolate behavioral categories.
    * **Product Recommendation Engine:** Run collaborative item-to-item matching using high-dimensional cosine similarity arrays.
    """)
    
    # Render operational metadata matrix metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Status", value="Active", delta="K-Means v1.0")
    col2.metric(label="Recommendation Engine", value="Online", delta="Cosine Space")
    col3.metric(label="Data Ingestion Pipes", value="Synced", delta="Streamlit Secrets Connected")

# =====================================================================
# 📋 CUSTOMER SEGMENTATION MODULE (Matching UI Screenshot)
# =====================================================================
elif app_mode == "📋 Clustering":
    st.title("Customer Segmentation")
    st.write("Determine a customer's strategic group instantly by updating their active behavior variables below:")
    
    # Controlled input numerical boxes matching screenshot behavior
    recency_input = st.number_input("Recency (days since last purchase)", min_value=0, max_value=1000, value=325, step=1)
    frequency_input = st.number_input("Frequency (number of purchases)", min_value=1, max_value=5000, value=1, step=1)
    monetary_input = st.number_input("Monetary (total spend)", min_value=0.0, max_value=1000000.0, value=765322.00, step=10.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Segment", type="primary"):
        # 1. Transform raw entries into a structured DataFrame
        raw_entry_df = pd.DataFrame([{'Recency': recency_input, 'Frequency': frequency_input, 'Monetary': monetary_input}])
        
        # 2. Re-execute the exact mathematical log transformation log1p sequence
        log_transformed_entry = np.log1p(raw_entry_df)
        
        # 3. Apply the training standard scaling transformation parameters
        scaled_entry = loaded_scaler.transform(log_transformed_entry)
        
        # 4. Extract cluster ID prediction assignment
        predicted_cluster_id = int(loaded_kmeans.predict(scaled_entry)[0])
        
        # 5. Business logic label dictionary mapping rules
        business_labels = {
            0: "High-Value Hero",
            1: "Regular Loyalist",
            2: "Occasional Shopper",
            3: "At-Risk Account"
        }
        resolved_label = business_labels.get(predicted_cluster_id, "Undefined Cluster")
        
        # Render clean output block mirroring the uploaded sample screenshot
        st.markdown(f"### ` {predicted_cluster_id} `")
        st.success(f"**This customer belongs to:** {resolved_label}")

# =====================================================================
# 📊 PRODUCT RECOMMENDATION MODULE (Matching UI Screenshot)
# =====================================================================
elif app_mode == "📊 Recommendation":
    st.title("Product Recommender")
    st.write("Input a product title below to instantly discover 5 highly correlated items bought by similar shoppers.")
    
    # Setup interactive selection query search component box
    try:
        available_catalog = list(similarity_matrix.columns)
        default_index = available_catalog.index("GREEN VINTAGE SPOT BEAKER") if "GREEN VINTAGE SPOT BEAKER" in available_catalog else 0
        
        search_query = st.selectbox(
            "Enter Product Name",
            options=available_catalog,
            index=default_index
        )
    except NameError:
        # Fallback text entry input structure if matrix is broken
        search_query = st.text_input("Enter Product Name", value="GREEN VINTAGE SPOT BEAKER")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Recommend", type="primary"):
        if 'similarity_matrix' in globals() or 'similarity_matrix' in locals():
            if search_query in similarity_matrix.columns:
                # Isolate product rows, drop self correlation at position index 0, and slice top 5 products
                raw_recommendations = similarity_matrix[search_query].sort_values(ascending=False).iloc[1:6]
                
                st.markdown("#### **Recommended Products:**")
                st.markdown("---")
                
                # Loop out items inside clean styled text representations mimicking card boundaries
                for i, (item_name, similarity_score) in enumerate(raw_recommendations.items(), 1):
                    st.markdown(f"✨ **{item_name}** *(Match Confidence Score: {similarity_score:.1%})*")
            else:
                st.error("The specified product name value was not located within our current interaction history logs.")
        else:
            st.error("Critical System Dependency Error: Recommendation Matrix asset uninitialized.")
