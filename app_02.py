import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Set webpage tab configurations
st.set_page_config(page_title="HALO — E-Commerce Analytics Console", layout="wide", page_icon="🛍️")

# =====================================================================
# 🧠 EMBEDDED PRODUCTION ML MODEL ARCHITECTURE  (UNCHANGED FROM ORIGINAL)
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
# ⚙️ DATA ENGINE: LOAD REPO DESCRIPTION CSV ASSET (UNCHANGED LOGIC)
# =====================================================================
@st.cache_resource
def load_cleaned_description_catalog():
    """Parses your description.csv file to construct the master A-Z index maps."""
    csv_filename = "description.csv"

    fallback_items = ["BLUE VINTAGE SPOT BEAKER", "GREEN VINTAGE SPOT BEAKER", "WHITE HANGING HEART T-LIGHT HOLDER"]
    fallback_index = {"B": ["BLUE VINTAGE SPOT BEAKER"], "G": ["GREEN VINTAGE SPOT BEAKER"], "W": ["WHITE HANGING HEART T-LIGHT HOLDER"]}

    if not os.path.exists(csv_filename):
        return fallback_items, fallback_index

    try:
        df = pd.read_csv(csv_filename)
        unique_catalog = sorted(df["Description"].dropna().astype(str).str.strip().str.upper().tolist())

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


all_unique_products, catalog_alphabet_index = load_cleaned_description_catalog()


# =====================================================================
# 📊 EMBEDDED ENGINE: SYSTEM COLLABORATIVE RECOMMENDATIONS (UNCHANGED LOGIC)
# =====================================================================
@st.cache_resource
def compute_live_recommendation_vector(target_item, search_pool):
    """Calculates matching item recommendations dynamically from the master inventory table."""
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
# 🎨 REACTBITS-STYLE COMPONENT LIBRARY (pure CSS + tiny inline JS, all
# defined right here in app.py — no external .js/.css files, nothing
# extra for Streamlit Cloud to install or serve).
# =====================================================================

def inject_global_styles():
    """One-time global <style> block: tokens + every animated CSS component."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --bg:#0a0912; --bg-elev:#15131f; --bg-elev-2:#1c1a29;
        --line:rgba(243,241,236,0.08);
        --text:#f3f1ec; --text-muted:#9c98ac; --text-dim:#6b6780;
        --violet:#8b7cf6; --violet-soft:#5b4fcb;
        --amber:#f2a93b; --hero-green:#3ddc97; --risk-rose:#ef5d6f;
        --ease:cubic-bezier(.22,1,.36,1);
    }

    html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
    .stApp{ background:var(--bg); color:var(--text); }
    section[data-testid="stSidebar"]{ background:rgba(12,11,20,0.85); border-right:1px solid var(--line); }
    h1,h2,h3,h4{ font-family:'Space Grotesk',sans-serif !important; }
    code, .mono{ font-family:'JetBrains Mono',monospace; }

    /* ---------- AURORA ANIMATED BACKDROP ---------- */
    .hb-aurora{ position:fixed; inset:0; z-index:-1; overflow:hidden; pointer-events:none; }
    .hb-aurora span{ position:absolute; width:55vw; height:55vw; border-radius:50%; filter:blur(90px); opacity:.30; }
    .hb-aurora .a1{ background:radial-gradient(circle,var(--violet),transparent 70%); top:-18%; left:-8%; animation:hbDrift1 26s ease-in-out infinite; }
    .hb-aurora .a2{ background:radial-gradient(circle,var(--hero-green),transparent 70%); top:6%; right:-14%; opacity:.16; animation:hbDrift2 32s ease-in-out infinite; }
    .hb-aurora .a3{ background:radial-gradient(circle,var(--amber),transparent 70%); bottom:-22%; left:18%; opacity:.14; animation:hbDrift3 38s ease-in-out infinite; }
    @keyframes hbDrift1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(6%,5%) scale(1.1)}}
    @keyframes hbDrift2{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(-8%,6%) scale(1.06)}}
    @keyframes hbDrift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(5%,-8%) scale(1.12)}}

    /* ---------- PARTICLE / STARFIELD DOTS (pure CSS) ---------- */
    .hb-particles{ position:fixed; inset:0; z-index:-1; pointer-events:none; }
    .hb-particles i{ position:absolute; width:3px; height:3px; border-radius:50%; background:var(--violet); opacity:.35; animation:hbTwinkle 4s ease-in-out infinite; }

    /* ---------- GRAIN OVERLAY ---------- */
    .hb-noise{ position:fixed; inset:0; z-index:999; pointer-events:none; opacity:.035; mix-blend-mode:overlay;
        background-image:radial-gradient(rgba(255,255,255,.6) 1px, transparent 1px); background-size:3px 3px; }
    @keyframes hbTwinkle{0%,100%{opacity:.1}50%{opacity:.5}}

    /* ---------- SPLIT TEXT (letter rise-in) ---------- */
    .hb-split span{ display:inline-block; opacity:0; transform:translateY(.6em); animation:hbRise .6s var(--ease) forwards; }
    @keyframes hbRise{ to{opacity:1; transform:translateY(0);} }

    /* ---------- BLUR TEXT ---------- */
    .hb-blur{ display:inline-block; filter:blur(8px); opacity:0; transform:translateY(6px); animation:hbBlurIn .7s var(--ease) forwards; }
    @keyframes hbBlurIn{ to{filter:blur(0); opacity:1; transform:translateY(0);} }

    /* ---------- GRADIENT TEXT ---------- */
    .hb-gradient-text{ background:linear-gradient(100deg,var(--violet),var(--hero-green) 45%,var(--amber) 85%);
        background-size:200% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
        animation:hbGradientPan 6s linear infinite; font-weight:700; }
    @keyframes hbGradientPan{ to{ background-position:200% center; } }

    /* ---------- SHINY TEXT (light sweep) ---------- */
    .hb-shiny{ background:linear-gradient(100deg,var(--text-dim) 35%,var(--text) 50%,var(--text-dim) 65%);
        background-size:220% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
        animation:hbShinySweep 3.2s linear infinite; }
    @keyframes hbShinySweep{ 0%{background-position:120% center;} 100%{background-position:-20% center;} }

    /* ---------- GLASS PANEL / CARD ---------- */
    .hb-glass{ background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
        border:1px solid var(--line); border-radius:14px; padding:20px 22px; backdrop-filter:blur(14px); }

    /* ---------- SPOTLIGHT CARD (CSS-only radial hover) ---------- */
    .hb-spotlight{ position:relative; border-radius:14px; border:1px solid var(--line);
        background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
        padding:20px 22px; overflow:hidden; transition:transform .3s var(--ease), border-color .3s var(--ease); }
    .hb-spotlight:hover{ transform:translateY(-3px); border-color:rgba(139,124,246,.35); }
    .hb-spotlight::before{ content:""; position:absolute; inset:0; opacity:0; transition:opacity .3s var(--ease);
        background:radial-gradient(220px circle at 50% 0%, rgba(139,124,246,.18), transparent 70%); }
    .hb-spotlight:hover::before{ opacity:1; }

    /* ---------- GRADIENT-BORDER CARD ---------- */
    .hb-border-card{ position:relative; border-radius:14px; padding:20px 22px;
        background:var(--bg-elev); }
    .hb-border-card::before{ content:""; position:absolute; inset:0; border-radius:14px; padding:1px;
        background:linear-gradient(120deg,var(--violet),transparent 40%, var(--hero-green) 80%);
        -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite:xor; mask-composite:exclude; }

    /* ---------- ORB (glowing pulsing segment marker) ---------- */
    .hb-orb-wrap{ display:flex; flex-direction:column; align-items:center; gap:10px; }
    .hb-orb{ border-radius:50%; opacity:.5; animation:hbFloat 5s ease-in-out infinite;
        box-shadow:0 0 24px var(--oc), inset 0 0 16px rgba(255,255,255,.15);
        background:radial-gradient(circle at 35% 30%, var(--oc), transparent 70%), var(--oc); }
    .hb-orb.active{ opacity:1; transform:scale(1.12); animation:hbPulse 1.8s ease-in-out infinite;
        box-shadow:0 0 50px var(--oc), inset 0 0 22px rgba(255,255,255,.25); }
    @keyframes hbFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
    @keyframes hbPulse{0%,100%{box-shadow:0 0 36px var(--oc), inset 0 0 18px rgba(255,255,255,.2)}50%{box-shadow:0 0 58px var(--oc), inset 0 0 26px rgba(255,255,255,.3)}}
    .hb-orb-label{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--text-muted); text-align:center; }

    /* ---------- STATUS DOT ---------- */
    .hb-status-dot{ width:7px; height:7px; border-radius:50%; background:var(--hero-green);
        box-shadow:0 0 8px var(--hero-green); display:inline-block; margin-right:6px;
        animation:hbBlink 2s ease-in-out infinite; }
    @keyframes hbBlink{ 50%{opacity:.35;} }

    /* ---------- PROGRESS / CONFIDENCE GLOW BAR ---------- */
    .hb-progress-track{ width:100%; height:8px; border-radius:6px; background:var(--bg-elev-2); overflow:hidden; }
    .hb-progress-fill{ height:100%; border-radius:6px; background:linear-gradient(90deg,var(--violet),var(--hero-green));
        animation:hbFillIn 1.1s var(--ease) forwards; transform-origin:left; }
    @keyframes hbFillIn{ from{ transform:scaleX(0); } to{ transform:scaleX(1); } }

    /* ---------- EYEBROW LABEL ---------- */
    .hb-eyebrow{ font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.14em;
        text-transform:uppercase; color:var(--text-dim); }

    /* ---------- MARQUEE ---------- */
    .hb-marquee{ overflow:hidden; -webkit-mask-image:linear-gradient(90deg,transparent,#000 10%,#000 90%,transparent);
        mask-image:linear-gradient(90deg,transparent,#000 10%,#000 90%,transparent); }
    .hb-marquee-track{ display:flex; gap:26px; width:max-content; animation:hbMarquee 26s linear infinite; }
    .hb-marquee-track span{ font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--text-dim); white-space:nowrap; }
    @keyframes hbMarquee{ from{transform:translateX(0);} to{transform:translateX(-50%);} }

    /* ---------- FADE-UP ENTRY (per-card stagger via nth-child) ---------- */
    .hb-fade{ opacity:0; transform:translateY(14px); animation:hbFadeUp .6s var(--ease) forwards; }
    @keyframes hbFadeUp{ to{ opacity:1; transform:translateY(0); } }

    /* ---------- STAR-BORDER PILL BADGE (rotating comet ring) ---------- */
    .hb-star-badge{ position:relative; display:inline-flex; padding:9px 18px; border-radius:999px;
        background:var(--bg-elev-2); font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13px;
        overflow:hidden; isolation:isolate; }
    .hb-star-badge::before{ content:""; position:absolute; inset:-40%; z-index:0;
        background:conic-gradient(from 0deg, transparent 0%, var(--violet) 8%, transparent 16%);
        animation:hbRotate 3.2s linear infinite; }
    .hb-star-badge span{ position:relative; z-index:1; background:var(--bg-elev-2); padding:2px 6px; border-radius:999px; }
    @keyframes hbRotate{ to{ transform:rotate(360deg); } }

    /* ---------- BUTTON OVERRIDES (magnetic hover feel) ---------- */
    div.stButton > button{
        border-radius:10px !important; border:1px solid var(--line) !important;
        background:var(--bg-elev) !important; color:var(--text) !important;
        font-family:'Space Grotesk',sans-serif !important; transition:transform .2s var(--ease), border-color .2s var(--ease) !important;
    }
    div.stButton > button:hover{ transform:translateY(-2px) scale(1.015); border-color:rgba(139,124,246,.4) !important; }
    div.stButton > button[kind="primary"]{ background:linear-gradient(120deg,var(--violet),var(--violet-soft)) !important; border:none !important; }

    @media (prefers-reduced-motion: reduce){
        *{ animation-duration:.001ms !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Fixed-position ambient backdrop, rendered once
    particle_dots = "".join(
        f'<i style="left:{(i*7.3) % 100}%; top:{(i*13.7) % 100}%; animation-delay:{(i%10)*0.4}s;"></i>'
        for i in range(45)
    )
    st.markdown(f"""
    <div class="hb-aurora"><span class="a1"></span><span class="a2"></span><span class="a3"></span></div>
    <div class="hb-particles">{particle_dots}</div>
    <div class="hb-noise"></div>
    """, unsafe_allow_html=True)


def split_text(text, tag="h1", extra_class="", delay_step=0.025):
    """Letter-by-letter rise-in headline."""
    spans = "".join(
        f'<span style="animation-delay:{i*delay_step:.3f}s">{("&nbsp;" if ch == " " else ch)}</span>'
        for i, ch in enumerate(text)
    )
    st.markdown(f'<{tag} class="hb-split {extra_class}">{spans}</{tag}>', unsafe_allow_html=True)


def blur_text(text, tag="p", extra_class=""):
    st.markdown(f'<{tag} class="hb-blur {extra_class}">{text}</{tag}>', unsafe_allow_html=True)


def gradient_text(text, tag="span", extra_class=""):
    return f'<{tag} class="hb-gradient-text {extra_class}">{text}</{tag}>'


def shiny_text(text, tag="span", extra_class=""):
    return f'<{tag} class="hb-shiny {extra_class}">{text}</{tag}>'


def eyebrow(text):
    st.markdown(f'<div class="hb-eyebrow">{text}</div>', unsafe_allow_html=True)


def glass_card_open(extra_class=""):
    st.markdown(f'<div class="hb-glass {extra_class}">', unsafe_allow_html=True)


def glass_card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def spotlight_card_html(inner_html):
    st.markdown(f'<div class="hb-spotlight">{inner_html}</div>', unsafe_allow_html=True)


def border_card_html(inner_html):
    st.markdown(f'<div class="hb-border-card">{inner_html}</div>', unsafe_allow_html=True)


def orb(color, label, active=False, size=70):
    cls = "hb-orb active" if active else "hb-orb"
    st.markdown(f"""
    <div class="hb-orb-wrap">
        <div class="{cls}" style="--oc:{color}; width:{size}px; height:{size}px;"></div>
        <div class="hb-orb-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def status_pill(text):
    st.markdown(f'<div class="hb-eyebrow"><span class="hb-status-dot"></span>{text}</div>', unsafe_allow_html=True)


def progress_glow(pct, color_from="var(--violet)", color_to="var(--hero-green)"):
    pct = max(0, min(100, pct))
    st.markdown(f"""
    <div class="hb-progress-track">
        <div class="hb-progress-fill" style="width:{pct}%; background:linear-gradient(90deg,{color_from},{color_to});"></div>
    </div>
    """, unsafe_allow_html=True)


def marquee(items, speed=26):
    spans = "".join(f"<span>{i}</span>" for i in (items + items))
    st.markdown(f"""
    <div class="hb-marquee"><div class="hb-marquee-track" style="animation-duration:{speed}s;">{spans}</div></div>
    """, unsafe_allow_html=True)


def star_badge(text):
    st.markdown(f'<div class="hb-star-badge"><span>{text}</span></div>', unsafe_allow_html=True)


def fade_grid(htmls, columns=3, stagger=0.08):
    """Render a row of glass cards with staggered fade-up entry, pure CSS."""
    cols = st.columns(columns)
    for i, (col, html) in enumerate(zip(cols * (len(htmls)//columns + 1), htmls)):
        with cols[i % columns]:
            st.markdown(
                f'<div class="hb-fade" style="animation-delay:{i*stagger:.2f}s">{html}</div>',
                unsafe_allow_html=True,
            )


def typewriter_widget(strings, height=46, font_size=20, color="#8b7cf6"):
    """Real sequential typewriter effect — tiny inline JS, lives inside this
    .py file as a string (no separate .js file), rendered in its own
    sandboxed iframe via components.html. Pure visual, no Streamlit state."""
    strings_js = str(strings).replace("'", '"')
    html = f"""
    <div style="font-family:'JetBrains Mono',monospace; font-size:{font_size}px; color:{color};
                background:transparent; white-space:nowrap;">
      <span id="hb-tw"></span><span style="animation:hbBlink 0.9s steps(1) infinite;">|</span>
    </div>
    <style>
      @keyframes hbBlink {{ 50% {{ opacity:0; }} }}
      body {{ margin:0; background:transparent; }}
    </style>
    <script>
      const strings = {strings_js};
      let idx = 0, char = 0, deleting = false;
      const el = document.getElementById('hb-tw');
      function tick() {{
        const current = strings[idx % strings.length];
        if (!deleting && char <= current.length) {{
          el.textContent = current.slice(0, char);
          char++;
          if (char > current.length) {{ deleting = true; setTimeout(tick, 1300); return; }}
        }} else if (deleting && char >= 0) {{
          el.textContent = current.slice(0, char);
          char--;
          if (char < 0) {{ deleting = false; idx++; char = 0; }}
        }}
        setTimeout(tick, deleting ? 28 : 45);
      }}
      tick();
    </script>
    """
    components.html(html, height=height)


def decrypt_text_widget(text, height=46, font_size=22, color="#f3f1ec"):
    """One-shot scramble-to-reveal text effect, same inline-JS approach as above."""
    html = f"""
    <div id="hb-dx" style="font-family:'JetBrains Mono',monospace; font-size:{font_size}px;
         color:{color}; background:transparent;"></div>
    <style>body{{margin:0; background:transparent;}}</style>
    <script>
      const target = {text!r};
      const glyphs = "!<>-_/[]{{}}=+*^?#";
      const el = document.getElementById('hb-dx');
      let frame = 0;
      const totalFrames = target.length * 3;
      const interval = setInterval(() => {{
        frame++;
        const reveal = Math.floor((frame / totalFrames) * target.length);
        let out = "";
        for (let i = 0; i < target.length; i++) {{
          out += i < reveal ? target[i] : glyphs[Math.floor(Math.random() * glyphs.length)];
        }}
        el.textContent = out;
        if (frame >= totalFrames) clearInterval(interval);
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
        ["🏠​ Home", "📋 Clustering", "📊 Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    status_pill("Model online")
    status_pill(f"{len(all_unique_products):,} products indexed")

# =====================================================================
# 🏠 HOME PAGE MODULE
# =====================================================================
if app_mode == "🏠​ Home":
    split_text("SHOPPER'S SPECTRUM", tag="h1")
    typewriter_widget([
        "Segment customers by RFM behavior.",
        "Surface look-alike product recommendations.",
        "All from one console.",
    ], height=40, font_size=18)

    blur_text(
        "Interactive Dashboard Covering All Products!!"
        ""
        ""
        "Machine Learning Model based Customer Segmentation",
        tag="p", extra_class=""
    )

    st.markdown("<br>", unsafe_allow_html=True)
    eyebrow("SEGMENT CONSTELLATION")
    orb_cols = st.columns(4)
    for i, col in enumerate(orb_cols):
        with col:
            orb(SEGMENT_COLOR_MAP[i], BUSINESS_SEGMENT_MAP[i], active=(i == 0))

    st.markdown("<br>", unsafe_allow_html=True)
    eyebrow("SYSTEM STATUS")
    fade_grid([
        f'<div class="hb-eyebrow">MODEL</div><h3 style="margin:6px 0 0;">{gradient_text("Active")}</h3><div style="color:var(--text-dim); font-size:13px; margin-top:4px;">Embedded pipeline v3.0</div>',
        f'<div class="hb-eyebrow">RECOMMENDATION ENGINE</div><h3 style="margin:6px 0 0;">{gradient_text("Online")}</h3><div style="color:var(--text-dim); font-size:13px; margin-top:4px;">Vector space matrix</div>',
        f'<div class="hb-eyebrow">DATA INGESTION</div><h3 style="margin:6px 0 0;">{gradient_text("Synced")}</h3><div style="color:var(--text-dim); font-size:13px; margin-top:4px;">description.csv active</div>',
    ], columns=3)

    st.markdown("<br><br>", unsafe_allow_html=True)
    eyebrow("CATALOG SAMPLE")
    marquee(all_unique_products[:18] if all_unique_products else ["No catalog loaded"])

# =====================================================================
# 📋 CUSTOMER SEGMENTATION MODULE
# =====================================================================
elif app_mode == "📋 Clustering":
    split_text("Customer Segmentation", tag="h1")
    blur_text("Determine a customer's strategic group instantly by updating their active behavior variables below:")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        recency_input = st.number_input("Recency (days since last purchase)", min_value=0, max_value=2000, value=1, step=1)
    with c2:
        frequency_input = st.number_input("Frequency (number of purchases)", min_value=1, max_value=10000, value=1, step=1)
    with c3:
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
        seg_color = SEGMENT_COLOR_MAP.get(predicted_cluster_id, "#8b7cf6")

        max_dist = max(distances) if max(distances) > 0 else 1
        confidence_pct = round((1 - (distances[predicted_cluster_id] / max_dist)) * 100, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        result_col, orb_col = st.columns([2, 1])

        with result_col:
            border_card_html(f"""
                <div class="hb-eyebrow">PREDICTED CLUSTER</div>
                <h2 style="margin:8px 0 2px;">{shiny_text('Cluster ' + str(predicted_cluster_id))}</h2>
                <div style="font-size:18px; margin-bottom:14px;">{gradient_text(resolved_label)}</div>
            """)
            decrypt_text_widget(resolved_label, height=38, font_size=20, color=seg_color)
            st.markdown(f'<div style="margin-top:10px; color:var(--text-dim); font-size:13px;">Relative confidence vs. nearest competing cluster</div>', unsafe_allow_html=True)
            progress_glow(confidence_pct, color_from=seg_color, color_to="var(--hero-green)")
            st.markdown(f'<div style="text-align:right; font-family:JetBrains Mono,monospace; font-size:12px; color:var(--text-dim); margin-top:4px;">{confidence_pct}%</div>', unsafe_allow_html=True)

        with orb_col:
            orb(seg_color, resolved_label, active=True, size=90)

        st.markdown("<br>", unsafe_allow_html=True)
        eyebrow("ALL SEGMENT DISTANCES")
        dist_cols = st.columns(4)
        for cid, col in enumerate(dist_cols):
            with col:
                is_winner = (cid == predicted_cluster_id)
                spotlight_card_html(f"""
                    <div class="hb-eyebrow">CLUSTER {cid}</div>
                    <div style="font-weight:600; margin:6px 0;">{BUSINESS_SEGMENT_MAP[cid]}</div>
                    <div class="mono" style="color:{SEGMENT_COLOR_MAP[cid] if is_winner else 'var(--text-dim)'}; font-size:13px;">
                        distance: {distances[cid]:.3f}{' ★' if is_winner else ''}
                    </div>
                """)

# =====================================================================
# 📊 PRODUCT RECOMMENDATION MODULE
# =====================================================================
elif app_mode == "📊 Recommendation":
    split_text("Product Recommender", tag="h1")
    blur_text("Input a product title below to instantly discover 5 highly correlated items bought by similar shoppers, or select from the alphabetical catalog directory below.")

    if "selected_product" not in st.session_state:
        st.session_state.selected_product = all_unique_products[0] if all_unique_products else "GREEN VINTAGE SPOT BEAKER"
    if "active_letter" not in st.session_state:
        st.session_state.active_letter = "A"

    try:
        default_index = all_unique_products.index(st.session_state.selected_product)
    except ValueError:
        default_index = 0

    search_query = st.selectbox(
        "Enter Product Name",
        options=all_unique_products,
        index=default_index,
        key="main_recommender_dropdown",
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get Recommendations", type="primary"):
        st.session_state.selected_product = search_query

    if st.session_state.selected_product:
        recommendations = compute_live_recommendation_vector(st.session_state.selected_product, all_unique_products)

        st.markdown("<br>", unsafe_allow_html=True)
        eyebrow("RECOMMENDED FOR")
        st.markdown(f'<h3 style="margin-top:4px;">{gradient_text(st.session_state.selected_product)}</h3>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        rec_cards = [
            f'<div style="font-size:14px; color:var(--text-dim); margin-bottom:6px;">✨ MATCH {i+1}</div><div style="font-weight:600;">{item}</div>'
            for i, item in enumerate(recommendations)
        ]
        cols = st.columns(5)
        for i, (col, html) in enumerate(zip(cols, rec_cards)):
            with col:
                st.markdown(f'<div class="hb-fade" style="animation-delay:{i*0.08:.2f}s">', unsafe_allow_html=True)
                spotlight_card_html(html)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    eyebrow("BROWSE CATALOG ALPHABETICALLY")
    st.markdown('<p style="color:var(--text-muted);">💡 Click a letter, then choose a product to load it into the engine above.</p>', unsafe_allow_html=True)
    st.markdown("---")

    alphabet_keys = sorted(list(catalog_alphabet_index.keys()))
    letter_columns = st.columns(len(alphabet_keys))

    for idx, letter in enumerate(alphabet_keys):
        with letter_columns[idx]:
            button_style = "primary" if st.session_state.active_letter == letter else "secondary"
            if st.button(letter, key=f"btn_let_{letter}", type=button_style, use_container_width=True):
                st.session_state.active_letter = letter
                st.rerun()

    st.markdown(f'<div class="hb-eyebrow" style="margin-top:14px;">SHOWING ITEMS STARTING WITH “{st.session_state.active_letter}”</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    filtered_products = catalog_alphabet_index.get(st.session_state.active_letter, [])

    if filtered_products:
        total_items = len(filtered_products)
        items_per_column = int(np.ceil(total_items / 3))

        col_left, col_mid, col_right = st.columns(3)

        with col_left:
            for item in filtered_products[0: items_per_column]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()

        with col_mid:
            for item in filtered_products[items_per_column: items_per_column * 2]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()

        with col_right:
            for item in filtered_products[items_per_column * 2:]:
                if st.button(f"📖 {item}", key=f"dir_lnk_{item}", use_container_width=True):
                    st.session_state.selected_product = item
                    st.rerun()
    else:
        st.write("*No products found matching this filter letter.*")
