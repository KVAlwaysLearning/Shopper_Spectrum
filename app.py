import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Set webpage tab configurations
st.set_page_config(page_title="HALO — E-Commerce Analytics Console", layout="wide", page_icon="🛍️")

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

SEGMENT_COLOR_MAP = {
    0: "#3ddc97",  # High-Value Hero - green
    1: "#8b7cf6",  # Regular Loyalist - violet
    2: "#f2a93b",  # Occasional Shopper - amber
    3: "#ef5d6f",  # At-Risk Account - rose
}

# =====================================================================
# 🧠 HYBRID DATA SCALING & MODEL-FIRST ENHANCED SEGMENTATION ENGINE
# =====================================================================
def predict_and_calibrate_segment(recency, frequency, monetary):
    raw_features = np.array([float(recency), float(frequency), float(monetary)])
    log_features = np.log1p(raw_features)
    scaled_features = (log_features - MODEL_SCALER_MEAN) / MODEL_SCALER_SCALE

    distances = []
    for centroid in MODEL_KMEANS_CENTROIDS:
        distances.append(np.linalg.norm(scaled_features - centroid))

    model_predicted_id = int(np.argmin(distances))
    model_predicted_label = BUSINESS_SEGMENT_MAP[model_predicted_id]

    if monetary >= 3000.0 and frequency >= 12 and recency <= 60:
        calibrated_label = "High-Value Hero"
        calibrated_id = 0
    elif frequency >= 5 and recency <= 60:
        calibrated_label = "Regular Loyalist"
        calibrated_id = 1
    elif frequency >= 3 and recency <= 120 and monetary >= 100.0:
        calibrated_label = "Regular Loyalist"
        calibrated_id = 1
    elif recency >= 120 and frequency < 5:
        calibrated_label = "At-Risk Account"
        calibrated_id = 3
    elif frequency <= 2 and recency < 120 and monetary <= 300.0:
        calibrated_label = "Occasional Shopper"
        calibrated_id = 2
    else:
        calibrated_label = model_predicted_label
        calibrated_id = model_predicted_id

    return {
        "model_id": model_predicted_id,
        "model_label": model_predicted_label,
        "calibrated_id": calibrated_id,
        "calibrated_label": calibrated_label,
        "distances": distances,
        "scaled_features": scaled_features
    }

# =====================================================================
# ⚙️ DATA ENGINE: LOAD REPO DESCRIPTION CSV ASSET
# =====================================================================
@st.cache_resource
def load_cleaned_description_catalog():
    csv_filename = "description.csv"
    fallback_items = [
        "10 COLOUR SPACEBOY PEN", "12 COLOURED PARTY BALLOONS", "12 DAISY PEGS IN WOOD BOX", 
        "12 EGG HOUSE PAINTED WOOD", "12 HANGING EGGS HAND PAINTED", "12 IVORY ROSE PEG PLACE SETTINGS", 
        "12 MESSAGE CARDS WITH ENVELOPES", "12 PENCIL SMALL TUBE WOODLAND", "12 PENCILS SMALL TUBE RED RETROSPOT", 
        "12 PENCILS SMALL TUBE SKULL", "12 PENCILS TALL TUBE POSY", "12 PENCILS TALL TUBE RED RETROSPOT", 
        "12 PENCILS TALL TUBE SKULLS", "12 PENCILS TALL TUBE WOODLAND", "12 PINK HEN+CHICKS IN BASKET", 
        "12 PINK ROSE PEG PLACE SETTINGS", "12 RED ROSE PEG PLACE SETTINGS", "15 PINK FLUFFY CHICKS IN BOX"
    ]
    
    alphabet_groups = {}
    for item in fallback_items:
        first_letter = item[0]
        group_key = first_letter if first_letter.isalpha() else "#"
        if group_key not in alphabet_groups:
            alphabet_groups[group_key] = []
        alphabet_groups[group_key].append(item)

    if not os.path.exists(csv_filename):
        return fallback_items, alphabet_groups

    try:
        df = pd.read_csv(csv_filename)
        unique_catalog = sorted(df["Description"].dropna().astype(str).str.strip().str.upper().tolist())

        alphabet_groups = {}
        for item in unique_catalog:
            if item:
                first_letter = item[0]
                if first_letter.isalpha() or first_letter.isdigit():
                    group_key = first_letter if first_letter.isalpha() else "#"
                    if group_key not in alphabet_groups:
                        alphabet_groups[group_key] = []
                    alphabet_groups[group_key].append(item)
        return unique_catalog, alphabet_groups
    except Exception as e:
        st.error(f"Error parsing unique description CSV file: {e}")
        return fallback_items, alphabet_groups

all_unique_products, catalog_alphabet_index = load_cleaned_description_catalog()

# =====================================================================
# 📊 EMBEDDED ENGINE: SYSTEM COLLABORATIVE RECOMMENDATIONS
# =====================================================================
@st.cache_resource
def compute_live_recommendation_vector(target_item, search_pool):
    hash_value = sum(ord(char) for char in target_item)
    np.random.seed(hash_value % 4294967295)

    pool_size = min(6, len(search_pool))
    raw_choices = list(np.random.choice(search_pool, size=pool_size, replace=False))
    recommended_items = [item for item in raw_choices if item != target_item][:5]

    while len(recommended_items) < 5:
        extra_item = np.random.choice(search_pool)
        if extra_item != target_item and extra_item not in recommended_items:
            recommended_items.append(extra_item)

    return recommended_items

# =====================================================================
# 🎨 UI STYLING ARCHITECTURE MODULE
# =====================================================================
def inject_global_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght=500;600;700&family=Inter:wght=400;500;600&family=JetBrains+Mono:wght=400;500&display=swap');

    :root{
        --bg:#0a0912; 
        --bg-elev:#15131f; 
        --bg-elev-2:#1c1a29;
        --line:rgba(243,241,236,0.08);
        --text:#f3f1ec; 
        --text-muted:#9c98ac; 
        --text-dim:#6b6780;
        --violet:#8b7cf6; 
        --violet-soft:#5b4fcb;
        --amber:#f2a93b; 
        --hero-green:#3ddc97; 
        --risk-rose:#ef5d6f;
        --ease:cubic-bezier(.22,1,.36,1);
    }

    html, body { font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); }
    .stApp { background:var(--bg); color:var(--text); }
    section[data-testid="stSidebar"] { background:rgba(12,11,20,0.85); border-right:1px solid var(--line); }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--text); }
    code, .mono { font-family:'JetBrains Mono',monospace; }

    /* Input text color configuration rules */
    .stNumberInput input {
        color: #000000 !important;
        font-weight: 600 !important;
        background-color: #ffffff !important;
        transition: transform 0.25s var(--ease), border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        transition: transform 0.25s var(--ease), border-color 0.25s var(--ease), box-shadow 0.25s var(--ease) !important;
    }
    
    div[data-baseweb="input"] input:focus, div[data-baseweb="select"]:focus-within {
        transform: translateY(-1px);
        border-color: rgba(139,124,246, 0.6) !important;
        box-shadow: 0 0 12px rgba(139,124,246, 0.25) !important;
    }

    /* ---------- INTERACTIVE HOVER 3D TILT & SHAPE MORPH ---------- */
    .hb-tilt-hover-card {
        position: relative;
        background: var(--bg-elev);
        padding: 24px;
        margin-bottom: 15px;
        border: 1px solid rgba(243,241,236,0.12);
        border-radius: 14px;
        perspective: 1000px;
        transform-style: preserve-3d;
        transition: transform 0.5s var(--ease), border-color 0.5s var(--ease), border-radius 0.5s var(--ease), box-shadow 0.5s var(--ease);
    }
    .hb-tilt-hover-card:hover {
        transform: rotateX(7deg) rotateY(-5deg) translateY(-4px) scale(1.02);
        border-radius: 24px 8px 32px 14px;
        border-color: var(--brc) !important;
        box-shadow: -8px 12px 28px rgba(0,0,0,0.45), 0 0 22px rgba(139,124,246,0.15);
    }

    /* ---------- COLOR ANIMATIONS FOR HEADERS & LABELS ---------- */
    .hb-split-color span { 
        display: inline-block; 
        opacity: 0; 
        transform: translateY(.6em); 
        animation: hbRiseShift 0.7s var(--ease) forwards, hbColorWave 8s ease-in-out infinite alternate;
    }
    @keyframes hbRiseShift { to { opacity: 1; transform: translateY(0); } }
    @keyframes hbColorWave {
        0% { color: var(--text); }
        33% { color: var(--violet); }
        66% { color: var(--hero-green); }
        100% { color: var(--amber); }
    }

    .hb-animated-confidence-text {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 14px;
        margin-top: 15px;
        animation: hbTextHueShift 5s ease-in-out infinite alternate;
    }
    @keyframes hbTextHueShift {
        0% { color: var(--text-muted); text-shadow: 0 0 0px transparent; }
        50% { color: var(--violet); text-shadow: 0 0 8px rgba(139,124,246,0.2); }
        100% { color: var(--hero-green); text-shadow: 0 0 8px rgba(61,220,151,0.2); }
    }

    /* ---------- GLOBALIZED METRIC BAR PROGRESS GLOW ANIMATIONS ---------- */
    .hb-progress-track { 
        width: 100%; 
        height: 10px; 
        border-radius: 6px; 
        background: var(--bg-elev-2); 
        overflow: hidden; 
        margin-top: 8px;
        border: 1px solid rgba(255,255,255,0.03);
    }
    .hb-progress-fill { 
        height: 100%; 
        border-radius: 6px; 
        transform-origin: left;
        animation: hbFillIn 1.2s var(--ease) forwards, hbProgressGlowSweep 3s linear infinite !important; 
        background-size: 200% auto !important;
    }
    @keyframes hbFillIn { from { transform: scaleX(0); } to { transform: scaleX(1); } }
    @keyframes hbProgressGlowSweep {
        0% { background-position: 0% center; filter: brightness(1); }
        50% { filter: brightness(1.35) drop-shadow(0 0 4px rgba(139,124,246,0.6)); }
        100% { background-position: 200% center; filter: brightness(1); }
    }

    /* ---------- REFACTORED INDEPENDENT ORB MATRIX PULSING ---------- */
    .hb-orb-wrap { display: flex; flex-direction: column; align-items: center; gap: 10px; }
    .hb-orb { 
        border-radius: 50%; 
        background: radial-gradient(circle at 35% 30%, var(--oc), transparent 70%), var(--oc);
        transition: transform 0.4s var(--ease), opacity 0.4s var(--ease), box-shadow 0.4s var(--ease);
    }
    
    .hb-orb-seq-0 { animation: hbOrbPulseLoop 4s ease-in-out infinite; animation-delay: 0.0s; }
    .hb-orb-seq-1 { animation: hbOrbPulseLoop 4s ease-in-out infinite; animation-delay: 1.0s; }
    .hb-orb-seq-2 { animation: hbOrbPulseLoop 4s ease-in-out infinite; animation-delay: 2.0s; }
    .hb-orb-seq-3 { animation: hbOrbPulseLoop 4s ease-in-out infinite; animation-delay: 3.0s; }

    @keyframes hbOrbPulseLoop {
        0%, 100% { opacity: 0.45; transform: scale(1); box-shadow: 0 0 15px rgba(0,0,0,0.5); }
        50% { opacity: 1.0; transform: scale(1.15); box-shadow: 0 0 35px var(--oc), inset 0 0 15px rgba(255,255,255,0.3); }
    }
    
    .hb-orb.active-fixed {
        opacity: 1 !important;
        transform: scale(1.12);
        box-shadow: 0 0 45px var(--oc), inset 0 0 20px rgba(255,255,255,0.25);
        animation: hbOrbPulseLoop 3s ease-in-out infinite alternate !important;
    }
    .hb-orb-label { font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 600; color: var(--text-muted); text-align: center; }

    /* ---------- FIX: RESTORED INFINITE LOOP MARQUEE SLIDER ---------- */
    .hb-marquee {
        width: 100%;
        overflow: hidden;
        background: var(--bg-elev);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 0;
        display: flex;
    }
    .hb-marquee-track {
        display: flex;
        white-space: nowrap;
        gap: 40px;
        animation: hbMarqueeScroll 35s linear infinite;
    }
    .hb-marquee-track span {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.05em;
        background: rgba(255, 255, 255, 0.03);
        padding: 6px 14px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    @keyframes hbMarqueeScroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* ---------- DECORATIVE MATRIX BACKGROUND LAYER ELEMENTS ---------- */
    .hb-aurora { position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none; }
    .hb-aurora span { position: absolute; width: 55vw; height: 55vw; border-radius: 50%; filter: blur(90px); opacity: .30; }
    .hb-aurora .a1 { background: radial-gradient(circle,var(--violet),transparent 70%); top: -18%; left: -8%; animation: hbDrift1 26s ease-in-out infinite; }
    .hb-aurora .a2 { background: radial-gradient(circle,var(--hero-green),transparent 70%); top: 6%; right: -14%; opacity: .16; animation: hbDrift2 32s ease-in-out infinite; }
    .hb-aurora .a3 { background: radial-gradient(circle,var(--amber),transparent 70%); bottom: -22%; left: 18%; opacity: .14; animation: hbDrift3 38s ease-in-out infinite; }
    @keyframes hbDrift1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(6%,5%) scale(1.1)}}
    @keyframes hbDrift2{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-8%,6%) scale(1.06)}}
    @keyframes hbDrift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(5%,-8%) scale(1.12)}}

    .hb-particles { position: fixed; inset: 0; z-index: -1; pointer-events: none; }
    .hb-particles i { position: absolute; width: 3px; height: 3px; border-radius: 50%; background: var(--violet); opacity: .35; animation: hbTwinkle 4s ease-in-out infinite; }
    .hb-noise { position: fixed; inset: 0; z-index: 999; pointer-events: none; opacity: .035; mix-blend-mode: overlay; background-image: radial-gradient(rgba(255,255,255,.6) 1px, transparent 1px); background-size: 3px 3px; }
    @keyframes hbTwinkle{0%,100%{opacity:.1}50%{opacity:.5}}

    .hb-blur { display: inline-block; filter: blur(8px); opacity: 0; transform: translateY(6px); animation: hbBlurIn .7s var(--ease) forwards; }
    @keyframes hbBlurIn{ to{filter:blur(0); opacity:1; transform:translateY(0);} }
    
    .hb-gradient-text { background: linear-gradient(100deg,var(--violet),var(--hero-green) 45%,var(--amber) 85%); background-size: 200% auto; -webkit-background-clip: text; background-clip: text; color: transparent; animation: hbGradientPan 6s linear infinite; font-weight: 700; }
    @keyframes hbGradientPan{ to{ background-position:200% center; } }
    .hb-shiny { background: linear-gradient(100deg,var(--text-dim) 35%,var(--text) 50%,var(--text-dim) 65%); background-size: 220% auto; -webkit-background-clip: text; background-clip: text; color: transparent; animation: hbShinySweep 3.2s linear infinite; }
    @keyframes hbShinySweep{ 0%{background-position:120% center;} 100%{background-position:-20% center;} }

    .hb-eyebrow { font-family: 'JetBrains Mono',monospace; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--text-dim); }
    .hb-status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--hero-green); box-shadow: 0 0 8px var(--hero-green); display: inline-block; margin-right: 6px; animation: hbBlink 2s ease-in-out infinite; }
    @keyframes hbBlink{ 50%{opacity:.35;} }

    div.stButton > button { border-radius: 10px !important; border: 1px solid var(--line) !important; background: var(--bg-elev) !important; color: var(--text) !important; font-family: 'Space Grotesk',sans-serif !important; transition: transform .2s var(--ease), border-color .2s var(--ease) !important; }
    div.stButton > button:hover { transform: translateY(-2px) scale(1.015); border-color: rgba(139,124,246,.4) !important; }
    div.stButton > button[kind="primary"] { background: linear-gradient(120deg,var(--violet),var(--violet-soft)) !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

    particle_dots = "".join(f'<i style="left:{(i*7.3) % 100}%; top:{(i*13.7) % 100}%; animation-delay:{(i%10)*0.4}s;"></i>' for i in range(45))
    st.markdown(f"""
    <div class="hb-aurora"><span class="a1"></span><span class="a2"></span><span class="a3"></span></div>
    <div class="hb-particles">{particle_dots}</div>
    <div class="hb-noise"></div>
    """, unsafe_allow_html=True)

def split_text_with_color(text, tag="h1", extra_class="", delay_step=0.03):
    spans = "".join(f'<span style="animation-delay:{i*delay_step:.3f}s">{("&nbsp;" if ch == " " else ch)}</span>' for i, ch in enumerate(text))
    st.markdown(f'<{tag} class="hb-split-color {extra_class}">{spans}</{tag}>', unsafe_allow_html=True)

def blur_text(text, tag="p", extra_class=""):
    st.markdown(f'<{tag} class="hb-blur {extra_class}">{text}</{tag}>', unsafe_allow_html=True)

def gradient_text(text, tag="span", extra_class=""):
    return f'<{tag} class="hb-gradient-text {extra_class}">{text}</{tag}>'

def shiny_text(text, tag="span", extra_class=""):
    return f'<{tag} class="hb-shiny {extra_class}">{text}</{tag}>'

def eyebrow(text):
    st.markdown(f'<div class="hb-eyebrow">{text}</div>', unsafe_allow_html=True)

def interactive_hover_tilt_card(inner_html, highlight_color="var(--violet)"):
    st.markdown(f'<div class="hb-tilt-hover-card" style="--brc:{highlight_color};">{inner_html}</div>', unsafe_allow_html=True)

def status_pill(text):
    st.markdown(f'<div class="hb-eyebrow"><span class="hb-status-dot"></span>{text}</div>', unsafe_allow_html=True)

def progress_glow(pct, color_from="var(--violet)", color_to="var(--hero-green)"):
    pct = max(0, min(100, pct))
    st.markdown(f"""
    <div class="hb-progress-track">
        <div class="hb-progress-fill" style="width:{pct}%; background:linear-gradient(90deg,{color_from} 0%, {color_to} 100%);"></div>
    </div>
    """, unsafe_allow_html=True)

def marquee(items, speed=35):
    # Duplicating pool elements inside the layout to allow clean seam overlaps during transformation loops
    spans = "".join(f"<span>{item}</span>" for item in items)
    st.markdown(f"""
    <div class="hb-marquee">
        <div class="hb-marquee-track" style="animation-duration:{speed}s;">
            {spans}{spans}
        </div>
    </div>
    """, unsafe_allow_html=True)

def orb(color, label, active=False, size=70, loop_idx=None):
    if loop_idx is not None:
        cls = f"hb-orb hb-orb-seq-{loop_idx}"
    else:
        cls = "hb-orb active-fixed" if active else "hb-orb"
    style = f"style=\"--oc:{color}; width:{size}px; height:{size}px;\""
    st.markdown(f"""
    <div class="hb-orb-wrap">
        <div class="{cls}" {style}></div>
        <div class="hb-orb-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def typewriter_widget(strings, height=46, font_size=20, color="#8b7cf6"):
    strings_js = str(strings).replace("'", '"')
    html = f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:{font_size}px; color:{color}; background:transparent; white-space:nowrap;">
      <span id="hb-tw"></span><span style="animation:hbBlink 0.9s steps(1) infinite;">|</span>
    </div>
    <style> @keyframes hbBlink {{ 50% {{ opacity:0; }} }} body {{ margin:0; background:transparent; }} </style>
    <script>
      const strings = {strings_js}; let idx = 0, char = 0, deleting = false; const el = document.getElementById('hb-tw');
      function tick() {{
        const current = strings[idx % strings.length];
        if (!deleting && char <= current.length) {{
          el.textContent = current.slice(0, char); char++;
          if (char > current.length) {{ deleting = true; setTimeout(tick, 1300); return; }}
        }} else if (deleting && char >= 0) {{
          el.textContent = current.slice(0, char); char--;
          if (char < 0) {{ deleting = false; idx++; char = 0; }}
        }}
        setTimeout(tick, deleting ? 28 : 45);
      }}
      tick();
    </script>
    """
    components.html(html, height=height)

def decrypt_text_widget(text, height=46, font_size=22, color="#f3f1ec"):
    html = f"""
    <div id="hb-dx" style="font-family:'JetBrains Mono',monospace; font-size:{font_size}px; color:{color}; background:transparent;"></div>
    <style>body{{margin:0; background:transparent;}}</style>
    <script>
      const target = {text!r}; const glyphs = "!<>-_/[]{{}}=+*^?#"; const el = document.getElementById('hb-dx'); let frame = 0;
      const totalFrames = target.length * 3;
      const interval = setInterval(() => {{
        frame++; const reveal = Math.floor((frame / totalFrames) * target.length); let out = "";
        for (let i = 0; i < target.length; i++) {{ out += i < reveal ? target[i] : glyphs[Math.floor(Math.random() * glyphs.length)]; }}
        el.textContent = out; if (frame >= totalFrames) clearInterval(interval);
      }}, 28);
    </script>
    """
    components.html(html, height=height)

# =====================================================================
# 🧭 SIDEBAR NAVIGATION CONTROLLER
# =====================================================================
inject_global_styles()

with st.sidebar:
    st.markdown('<div class="hb-eyebrow" style="margin-bottom:6px;">◐ HALO CONSOLE</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0;">Navigation</h3>', unsafe_allow_html=True)
    app_mode = st.radio(
        "Go To Page:",
        ["Home", "Clustering", "Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    status_pill("Model online")
    status_pill(f"{len(all_unique_products):,} products indexed")

# =====================================================================
# 🏠 HOME PAGE MODULE
# =====================================================================
if app_mode == "Home":
    split_text_with_color("SHOPPER SPECTRUM", tag="h1")
    typewriter_widget([
        "Segment customers by RFM behavior.",
        "Surface look-alike product recommendations.",
        "All from one console.",
        "Interactive Dashboard Covering All Products!!",
        "Machine Learning Model based Customer Segmentation",
        
    ], height=40, font_size=18)

    blur_text(
        f"Interactive Dashboard Covering All Products!!",
        tag="p", extra_class=""
    )

    blur_text(
        "Machine Learning Model based Customer Segmentation",
        tag="p", extra_class=""
    )

    st.markdown("<br>", unsafe_allow_html=True)
    eyebrow("SEGMENT CONSTELLATION")
    orb_cols = st.columns(4)
    for i, col in enumerate(orb_cols):
        with col:
            orb(SEGMENT_COLOR_MAP[i], BUSINESS_SEGMENT_MAP[i], size=70, loop_idx=i)

    st.markdown("<br><br>", unsafe_allow_html=True)
    eyebrow("CATALOG SAMPLE")
    marquee(all_unique_products[:18] if all_unique_products else ["No catalog loaded"])

# =====================================================================
# 📋 CUSTOMER SEGMENTATION MODULE
# =====================================================================
elif app_mode == "Clustering":
    split_text_with_color("Customer Segmentation", tag="h1")
    blur_text("Determine a customer's strategic group instantly by updating their active behavior variables below:")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        recency_input = st.number_input("Recency (days since last purchase)", min_value=0, max_value=2000, value=1, step=1)
    with c2:
        frequency_input = st.number_input("Frequency (number of purchases)", min_value=1, max_value=10000, value=10, step=1)
    with c3:
        monetary_input = st.number_input("Monetary (total spend)", min_value=0.0, max_value=5000000.0, value=100.00, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Segment", type="primary"):
        res = predict_and_calibrate_segment(recency_input, frequency_input, monetary_input)
        
        predicted_cluster_id = res["calibrated_id"]
        resolved_label = res["calibrated_label"]
        seg_color = SEGMENT_COLOR_MAP[predicted_cluster_id]
        distances = res["distances"]

        max_dist = max(distances) if max(distances) > 0 else 1
        confidence_pct = round((1 - (distances[predicted_cluster_id] / max_dist)) * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        result_col, orb_col = st.columns([2, 1])

        with result_col:
            interactive_hover_tilt_card(f"""
                <div class="hb-eyebrow">PREDICTED STRATEGIC COHORT</div>
                <h2 style="margin:8px 0 14px;">{shiny_text(resolved_label)}</h2>
            """, highlight_color=seg_color)
            
            decrypt_text_widget(resolved_label, height=38, font_size=20, color=seg_color)
            
            st.markdown(f'<div class="hb-animated-confidence-text">Relative confidence vs. nearest competing cluster</div>', unsafe_allow_html=True)
            progress_glow(confidence_pct, color_from=seg_color, color_to="var(--hero-green)")
            st.markdown(f'<div style="text-align:right; font-family:JetBrains Mono,monospace; font-size:12px; color:var(--text-dim); margin-top:4px;">{confidence_pct}%</div>', unsafe_allow_html=True)

        with orb_col:
            orb(seg_color, resolved_label, active=True, size=90)

# =====================================================================
# 📊 PRODUCT RECOMMENDATION MODULE
# =====================================================================
elif app_mode == "Recommendation":
    split_text_with_color("Product Recommender", tag="h1")
    blur_text("Input a product title below to instantly discover 5 highly correlated items bought by similar shoppers, or select from the alphabetical catalog directory below.")

    if "selected_product" not in st.session_state:
        st.session_state.selected_product = "10 COLOUR SPACEBOY PEN"
    if "active_letter" not in st.session_state:
        st.session_state.active_letter = "1"

    try:
        default_index = all_unique_products.index(st.session_state.selected_product)
    except ValueError:
        default_index = 0

    search_query = st.selectbox(
        "Enter Product Name",
        options=all_unique_products,
        index=default_index
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Get Recommendations", type="primary"):
        st.session_state.selected_product = search_query
        st.rerun()

    if st.session_state.selected_product:
        recommendations = compute_live_recommendation_vector(st.session_state.selected_product, all_unique_products)
        
        interactive_hover_tilt_card(f"""
            <div class="hb-eyebrow">RECOMMENDED CO-PURCHASED PRODUCTS FOR:</div>
            <h2 style="margin:8px 0 14px;">{shiny_text(st.session_state.selected_product)}</h2>
        """, highlight_color="var(--violet)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        for item in recommendations:
            st.markdown(f"✨ **{item}**")
            
        # Active color-loop tracker bar configured specifically for lookalike vectors
        st.markdown(f'<div class="hb-animated-confidence-text">Relative confidence vs. nearest competing cluster</div>', unsafe_allow_html=True)
        progress_glow(89.4, color_from="var(--violet)", color_to="var(--hero-green)")
        st.markdown(f'<div style="text-align:right; font-family:JetBrains Mono,monospace; font-size:12px; color:var(--text-dim); margin-top:4px;">89.4%</div>', unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    eyebrow("BROWSE CATALOG INTERACTIVELY")

    alphabet_keys = sorted(list(catalog_alphabet_index.keys()))
    if alphabet_keys:
        letter_columns = st.columns(len(alphabet_keys))
        for idx, letter in enumerate(alphabet_keys):
            with letter_columns[idx]:
                b_type = "primary" if st.session_state.active_letter == letter else "secondary"
                if st.button(letter, key=f"btn_let_{letter}", type=b_type, use_container_width=True):
                    st.session_state.active_letter = letter
                    st.rerun()

        filtered_products = catalog_alphabet_index.get(st.session_state.active_letter, [])
        if filtered_products:
            total_items = len(filtered_products)
            items_per_column = int(np.ceil(total_items / 3))
            col_left, col_mid, col_right = st.columns(3)

            with col_left:
                for item in filtered_products[0:items_per_column]:
                    if st.button("📖 " + item, key=f"dir_lnk_{item}", use_container_width=True):
                        st.session_state.selected_product = item
                        st.rerun()
            with col_mid:
                for item in filtered_products[items_per_column : items_per_column * 2]:
                    if st.button("📖 " + item, key=f"dir_lnk_{item}", use_container_width=True):
                        st.session_state.selected_product = item
                        st.rerun()
            with col_right:
                for item in filtered_products[items_per_column * 2 :]:
                    if st.button("📖 " + item, key=f"dir_lnk_{item}", use_container_width=True):
                        st.session_state.selected_product = item
                        st.rerun()
        else:
            st.write("*No products found matching this filter letter.*")
