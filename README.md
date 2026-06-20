# Shop Spectrum: E-Commerce Customer Analytics & Personalized Recommendation Engine

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end unsupervised machine learning ecosystem that transforms raw, fragmented retail transactional data into highly actionable customer segments and real-time cross-selling recommendations. Powered by an enterprise RFM clustering engine and item-based collaborative filtering, this project bridges the gap between raw database ledgers and customer personalization.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Core Architecture & Workflow](#-core-architecture--workflow)
- [Dataset & Rigorous Preprocessing](#-dataset--rigorous-preprocessing)
- [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [Machine Learning Architecture](#-machine-learning-architecture)
  - [Customer Segmentation (RFM + K-Means)](#1-customer-segmentation-rfm--k-means)
  - [Collaborative Filtering Recommender](#2-collaborative-filtering-recommender)
- [Streamlit Dashboard UI Features](#-streamlit-dashboard-ui-features)
- [Installation & Local Deployment](#-installation--local-deployment)
- [Project Directory Structure](#-project-directory-structure)
- [Model Performance Summary](#-model-performance-summary)

---

## 🎯 Project Overview

Modern e-commerce platforms generate massive operational transaction matrices but struggle to maximize Customer Lifetime Value (LTV) due to two structural issues:
1. **Customer Monolithism:** Treating all buyers with identical marketing strategies, leading to inefficient ad spend and preventable churn.
2. **Catalog Staticity:** Missing cross-selling funnels during the checkout experience by not analyzing look-alike purchasing patterns.

**Shop Spectrum** solves this by delivering a dual-engine architecture:
* An **Unsupervised Customer Segmentation Engine** that breaks down customer bases into distinct tactical cohorts based on transactional histories.
* An **Item-Based Collaborative Filtering Engine** that maps co-purchase associations to drive an instantaneous cross-selling utility funnel.

---

## ⚙️ Core Architecture & Workflow

The pipeline ingests raw flat ledgers and outputs optimized interactive user interfaces:

1.  **Ingestion & Surgical Cleansing:** Outlier pruning, transaction mapping, and string normalization.
2.  **Feature Engineering Layer:** Extrapolating `LineTotal` and distilling data into customer-level RFM vectors.
3.  **Algorithmic Modeling Pipeline:** Math transformations ($\log(x+1)$ scaling), silhouette grid-searches, and transpose matrix multiplications ($C \cdot C^T$).
4.  **Production Asset Serialization:** Packaging objects using binary vector file pickles (`.pkl`).
5.  **Interactive Streamlit Execution UI:** A high-performance dashboard that incorporates 3D CSS physics styling, dynamic typography decryption rendering, and neon animated progress trackers.

---

## 📊 Dataset & Rigorous Preprocessing

The model uses a standard retail transactional ledger consisting of **541,909 raw records**. The clean data pipeline implements strict validation guardrails:

* **Missing Data Strategy:** Identified and eliminated 135,080 unlogged guest checkout accounts missing a structural `CustomerID` to secure deterministic clustering attributes.
* **Ledger Pruning:** Removed financial administrative noise variables such as Manual Adjustments (`M`), Postage fees (`POST`), and Bank Charges (`BANK CHARGES`) which warp true product interaction profiles.
* **Cancellation Filtering:** Isolated and eliminated transactional cancellations (records flagged with a negative quantity or an invoice code prefixed with `C`).
* **String Uniformity:** Applied character standardization across product descriptions using multi-space compression and non-ASCII character filtration to prevent string token fragmentation.

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory data analysis followed a rigorous Univariate, Bivariate, and Multivariate (UBM) inspection framework:

* **Geographic Concentration:** Found that gross revenue generation was heavily concentrated within the United Kingdom domestic infrastructure.
* **Temporal Micro-Trends:** Revealed an operational peak in mid-week trading (specifically Thursdays), with systemic drops over weekend periods.
* **Volume Relationships:** Bivariate log-space scatter charts confirmed an intense linear correlation coefficient ($0.90$) between overall invoice quantities and gross transaction values, validating that volume-driven inventory turnover drives company growth over premium item markups.

---

## 🧠 Machine Learning Architecture

### 1. Customer Segmentation (RFM + K-Means)
To extract clean geometric groups, the transaction history was summarized into an individual customer-level profile matrix:
* **Recency ($R$):** Days elapsed from the customer's last invoice relative to the dataset horizon.
* **Frequency ($F$):** Aggregate count of valid invoices initialized by the account.
* **Monetary ($M$):** Cumulative capital spend transacted across the customer life cycle.

Because RFM variables exhibit extreme right-skew distributions, features were log-transformed ($\log(x+1)$) and globally standardized via `StandardScaler` to ensure equal variance during distance computations.


```text

Raw RFM Matrix ──> Log Transform [log1p] ──> StandardScaler ──> K-Means++ Optimization Loop

```

Three clustering methodologies were cross-evaluated:
* **K-Means Clustering:** Seeding with `k-means++` across parameter sets $K \in [2, 8]$. The Elbow Curve (Within-Cluster Sum of Squares) combined with Silhouette Analysis verified that a $4$-cluster configuration yielded the most stable boundaries.
* **DBSCAN:** Deployed to analyze non-spherical density layouts and flag extreme whale investors as unclustered system noise.
* **Agglomerative Hierarchical Clustering:** Utilized Ward linkage matrices to verify structure configurations.

#### Assigned Business Archetypes:
* **High-Value Hero (Cluster 0):** Highly recent, high-frequency, premium-capital spend investors.
* **Regular Loyalist (Cluster 1):** Dependable shoppers maintaining steady, predictable interaction loops.
* **Occasional Shopper (Cluster 2):** Low-frequency buyers showing seasonal or sporadic engagement.
* **At-Risk Account (Cluster 3):** Stalled accounts marked by prolonged periods of inactivity.

### 2. Collaborative Filtering Recommender
The cross-selling model transforms transactional logs into a broad user-item interaction pivot matrix:
$$A \in \mathbb{R}^{U \times I}$$
where $U$ represents unique customers, $I$ contains individual product `StockCode` columns, and cells hold the sum total of items purchased.

Product vectors are cross-multiplied using **Cosine Similarity**:
$$\text{similarity}(\mathbf{A}_i, \mathbf{A}_j) = \frac{\mathbf{A}_i \cdot \mathbf{A}_j}{\|\mathbf{A}_i\| \|\mathbf{A}_j\|}$$

The pre-computed similarity matrix is exposed to a lightweight production wrapper function. When queried with an item ID, it isolates the designated vector row, sorts similarity weights in descending order, and resolves the top 5 matching indexes into product descriptions.

---

## 🎨 Streamlit Dashboard UI Features

The interface uses an advanced, highly responsive front-end layer embedded directly into Streamlit components:

* **Kinetic Color Headers:** Title segments render through a dynamic multi-color letter reveal, shifting through a smooth pulsing gradient matrix.
* **Interactive 3D Tilt Panels:** Result cards utilize CSS perspective matrices to execute realistic 3D tilt axes rotations and corner-shape morphing upon hover.
* **Cipher Decryption Text Widgets:** Selected cluster metrics display via an animated Javascript typography matrix scramble that settles cleanly into the final text result.
* **Neon Progress Glow Tracks:** Relative modeling confidence indicators are backed by real-time animated breathing color cycles.

---

## 🚀 Installation & Local Deployment

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment manager (`venv` or `conda`)

### Step-by-Step Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/shop-spectrum.git](https://github.com/yourusername/shop-spectrum.git)
   cd shop-spectrum

```

2. **Establish and Activate a Virtual Environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Prepare Underlying CSV Datasets:**
Ensure your cleaned catalog metadata asset `description.csv` is correctly positioned within the root directory file structure.
5. **Launch the Production UI Dashboard Server:**
```bash
streamlit run app.py

```


The local server will instantiate and map inside your active browser window at `http://localhost:8501`.

---

## 📂 Project Directory Structure

```text
├── data/
│   └── Data.csv                # Raw, uncleaned transactions ledger source
├── description.csv             # Cleaned unique index product catalog metadata
├── Shop_Spectrum_Working.ipynb  # Comprehensive R&D development notebook file
├── app.py                      # Main production UI deployment script
├── requirements.txt            # Package dependencies manifest
└── README.md                   # System documentation documentation

```

---

## 📈 Model Performance Summary

| Algorithm Archetype | Metric Tracked | Optimal Configuration | Core Strategic Output |
| --- | --- | --- | --- |
| **K-Means Clustering** | Silhouette Coefficient / WCSS | $K = 4$ Clusters | Behavioral Cohort Isolation Matrix |
| **Cosine Similarity** | Angular Vector Geometry | Item-to-Item $I \times I$ | Automated Look-alike Cross-Selling Suggestions |
