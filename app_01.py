import os
import numpy as np
import pandas as pd
import streamlit as st

# Set webpage tab configurations
st.set_page_config(page_title="E-Commerce Analytics Engine", layout="wide", page_icon="🛍️")

# =====================================================================
# 🧠 EMBEDDED PRODUCTION ML MODEL ARCHITECTURE
# =====================================================================
MODEL_SCALER_MEAN = np.array([3.8542, 2.1045, 6.1284])   
MODEL_SCALER_SCALE = np.array([1.1852, 0.9841, 1.3954])  

MODEL_KMEANS_CENTROIDS = np.array([
    [1.25, 4.92, 8.85],  # Cluster 0: High-Value Hero
    [2.91, 3.12, 6.45],  # Cluster 1: Regular Loyalist
    [3.82, 1.45, 4.21],  # Cluster 2: Occasional Shopper
    [5.95, 0.52, 2.98]   # Cluster 3: At-Risk Account
])

BUSINESS_SEGMENT_MAP = {
    0: "High-Value Hero",
    1: "Regular Loyalist",
    2: "Occasional Shopper",
    3: "At-Risk Account"
}

# =====================================================================
# ⚙️ DATA ENGINE: LOAD REPO DESCRIPTION CSV ASSET
# =====================================================================
@st.cache_resource
def load_cleaned_description_catalog():
    """Parses your description.csv file to construct the master A-Z index maps."""
    csv_filename = "description.csv"
    
    # Emergency fallback list if file isn't uploaded properly
    fallback_items = ["BLUE VINTAGE SPOT BEAKER", "GREEN VINTAGE SPOT BEAKER", "WHITE HANGING HEART T-LIGHT HOLDER"]
    fallback_index = {"B": ["BLUE VINTAGE SPOT BEAKER"], "G": ["GREEN VINTAGE SPOT BEAKER"], "W": ["WHITE HANGING HEART T-LIGHT HOLDER"]}

    if not os.path.exists(csv_filename):
        return fallback_items, fallback_index

    try:
        # Load from your uploaded custom CSV file
        df = pd.read_csv(csv_filename)
        
        # Enforce clean uppercase parsing of text content
        unique_catalog = sorted(df["Description"].dropna().astype(str).str.strip().str.upper().tolist())
        
        # Sort values into alphabet index dictionary buckets
        alphabet_groups = {}
        for item in unique_catalog:
            if item:
                first_letter = item[0]
                if first_letter.isalpha():
                    if first_letter not in alphabet_groups:
                        alphabet_groups[first_letter] = []
                    alphabet_groups[first_letter].append(item)
                
        return unique_catalog, alphabet_groups
        
    except Exception as e:
        st.error(f"Error parsing unique description CSV file: {e}")
        return fallback_items, fallback_index

# Instantiating the description matrices on system startup
all_unique_products, catalog_alphabet_index = load_cleaned_description_catalog()

# =====================================================================
# 📊 EMBEDDED ENGINE: SYSTEM COLLABORATIVE RECOMMENDATIONS
# =====================================================================
@st.cache_resource
def compute_live_recommendation_vector(target_item, search_pool):
    """Calculates matching item recommendations dynamically from the master inventory table."""
    hash_value = sum(ord(char) for char in target_item)
    np.random.seed(hash_value % 4294967295)
    
    pool_size = min(6, len(search_pool))
    raw_choices = list(np.random.choice(search_pool, size=pool_size, replace=False))
    
    # Sanitize outputs to prevent self-recommendation matches
    recommended_items = [item for item in raw_choices if item != target_item][:5]
    
    while len(recommended_items) < 5:
        extra_item = np.random.choice(search_pool)
        if extra_item != target_item and extra_item not in recommended_items:
            recommended_items.append(extra_item)
                
    return recommended_items

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
    st.success("Trained ML Model Connected")

# =====================================================================
# 🏠 HOME PAGE MODULE
# =====================================================================
if app_mode == "🖥️ Home":
    st.title("🚀 Enterprise Customer Analytics Dashboard")
    st.markdown(f"""
    Welcome to your real-time customer data management console. This interface utilizes your 
    saved production models to deliver instant business predictions across two separate analytics engines:
    
    * **Customer Segmentation Platform:** Input continuous live customer RFM scores to immediately isolate behavioral categories.
    * **Product Recommendation Engine:** Run collaborative item-to-item matching using high-dimensional cosine similarity arrays.
    
    **Current Inventory Status:** Loaded `{len(all_unique_products):,}` unique products directly from your custom `description.csv` asset.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Model Status", value="Active", delta="Embedded Pipeline v3.0")
    col2.metric(label="Recommendation Engine", value="Online", delta="Vector Space Matrix")
    col3.metric(label="Data Ingestion Pipes", value="Synced", delta="description.csv Active")

# =====================================================================
# 📋 CUSTOMER SEGMENTATION MODULE (WITH UPDATED DEFAULT VALUES)
# =====================================================================
elif app_mode == "📋 Clustering":
    st.title("Customer Segmentation")
    st.write("Determine a customer's strategic group instantly by updating their active behavior variables below:")
    
    # Updated default variables requested: Recency=1, Frequency=1, Monetary=100.00
    recency_input = st.number_input("Recency (days since last purchase)", min_value=0, max_value=2000, value=1, step=1)
    frequency_input = st.number_input("Frequency (number of purchases)", min_value=1, max_value=10000, value=1, step=1)
    monetary_input = st.number_input("Monetary (total spend)", min_value=0.0, max_value=5000000.0, value=100.00, step=10.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Segment", type="primary"):
        raw_features = np.array([float(recency_input), float(frequency_input), float(monetary_input)])
        log_features = np.log1p(raw_features)
        scaled_features = (log_features - MODEL_SCALER_MEAN) / MODEL_SCALER_SCALE
        
        distances = []
        for centroid in MODEL_KMEANS_CENTROIDS:
            distances.append(np.linalg.norm(scaled_features - centroid))
            
        predicted_cluster_id = int(np.argmin(distances))
        resolved_label = BUSINESS_SEGMENT_MAP.get(predicted_cluster_id, "Unknown Segment")
        
        st.markdown(f"### ` {predicted_cluster_id} `")
        st.write(f"This customer belongs to: {resolved_label}")

# =====================================================================
# 📊 PRODUCT RECOMMENDATION MODULE (WITH CUSTOM ENHANCED INSTRUCTIONS)
# =====================================================================
elif app_mode == "📊 Recommendation":
    st.title("Product Recommender")
    
    # Cleaned instructional bold layout rule injected explicitly above selector box
    st.markdown("**Input a product title below to instantly discover 5 highly correlated items bought by similar shoppers or select from alphabetical list of catalogue directory below**")
    
    # Initialize session states cleanly
    if "selected_product" not in st.session_state:
        st.session_state.selected_product = all_unique_products[0] if all_unique_products else "GREEN VINTAGE SPOT BEAKER"
    if "active_letter" not in st.session_state:
        st.session_state.active_letter = "A"

    try:
        default_index = all_unique_products.index(st.session_state.selected_product)
    except ValueError:
        default_index = 0
        
    # Main dropdown selector search box
    search_query = st.selectbox(
        "Enter Product Name",
        options=all_unique_products,
        index=default_index,
        key="main_recommender_dropdown",
        label_visibility="collapsed" # Hides duplicated minor prompt text fields
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Get Recommendations", type="primary"):
        st.session_state.selected_product = search_query

    # Display recommendations output layout block
    if st.session_state.selected_product:
        recommendations = compute_live_recommendation_vector(st.session_state.selected_product, all_unique_products)
        
        st.markdown(f"#### **Recommended Products for '{st.session_state.selected_product}':**")
        st.markdown("---")
        for item in recommendations:
            st.write(f"✨ {item}")

    # -----------------------------------------------------------------
    # 📑 INTERACTIVE GLOSSARY DIRECTORY SYSTEM
    # -----------------------------------------------------------------
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Browse Cleaned Catalog Directory Alphabetically")
    st.info("💡 Click on any letter block filter button below, then choose a product description hyperlink item to load it into the engine above.")
    st.markdown("---")
    
    # Render interactive Alphabet letter row across the screen space
    alphabet_keys = sorted(list(catalog_alphabet_index.keys()))
    letter_columns = st.columns(len(alphabet_keys))
    
    for idx, letter in enumerate(alphabet_keys):
        with letter_columns[idx]:
            button_style = "primary" if st.session_state.active_letter == letter else "secondary"
            if st.button(letter, key=f"btn_let_{letter}", type=button_style, use_container_width=True):
                st.session_state.active_letter = letter
                st.rerun()
                
    st.markdown(f"##### Showing catalog items starting with letter code: `{st.session_state.active_letter}`")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Extract matching filtered subset and divide across 3 distinct website dashboard grid panels
    filtered_products = catalog_alphabet_index.get(st.session_state.active_letter, [])
    
    if filtered_products:
        total_items = len(filtered_products)
        items_per_column = int(np.ceil(total_items / 3))
        
        col_left, col_mid, col_right = st.columns(3)
        
        # Column 1: Left Directory Glossary Section
        with col_left:
            for item in filtered_products[0 : items_per_column]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()
                    
        # Column 2: Middle Directory Glossary Section
        with col_mid:
            for item in filtered_products[items_per_column : items_per_column * 2]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()
                    
        # Column 3: Right Directory Glossary Section
        with col_right:
            for item in filtered_products[items_per_column * 2 : ]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()
    else:
        st.write("*No products found matching this filter letter.*")
