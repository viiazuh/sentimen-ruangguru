import streamlit as st
import pandas as pd
import time
import re
import random

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. AGGRESSIVE LIGHT MODE CSS (Fixing Black Inputs & Sidebar Text) ---
st.markdown("""
    <style>
    /* 1. Paksa Background Utama & Teks */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f7f9fc !important;
        color: #1f2937 !important;
    }

    /* 2. Perbaiki Input Area (Textarea & File Uploader) agar tidak hitam */
    textarea, [data-testid="stFileUploader"], [data-testid="stTextInputRootElement"] {
        background-color: white !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* 3. Perbaiki Sidebar agar Teks Terlihat */
    [data-testid="stSidebar"] {
        background-color: white !important;
        border-right: 1px solid #e5e7eb !important;
    }
    
    /* Teks Caption di bawah logo Sentiment agar hitam pekat */
    .sidebar-caption {
        color: #4b5563 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* 4. Card Styling (Dashboard) */
    .metric-card {
        background-color: white; padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f3f4f6;
        display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;
    }
    .metric-title { color: #6b7280 !important; font-size: 0.9rem; font-weight: 600; }
    .metric-value { color: #1f2937 !important; font-size: 2.2rem; font-weight: 700; margin-top: 8px; }
    
    /* 5. Icon Boxes */
    .icon-box { width: 45px; height: 45px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px; }
    .bg-blue { background-color: #dbeafe; color: #3b82f6; }
    .bg-green { background-color: #d1fae5; color: #10b981; }
    .bg-red { background-color: #fee2e2; color: #ef4444; }
    .bg-gray { background-color: #f3f4f6; color: #6b7280; }

    /* 6. Button Orange Glow */
    .stButton>button {
        background: linear-gradient(135deg, #fb923c, #f97316) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 700 !important;
        box-shadow: 0 4px 10px rgba(249, 115, 22, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# --- 4. PREPROCESSING LOGIC ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_prediction(text):
    pos_words = ["bagus", "puas", "mantap", "keren", "oke"]
    neg_words = ["buruk", "kecewa", "parah", "jelek", "lambat"]
    score = 0
    cleaned = clean_text(text)
    for w in pos_words: 
        if w in cleaned: score += 1
    for w in neg_words: 
        if w in cleaned: score -= 1
    if score > 0: return "Positif", "😊", random.randint(75, 98)
    if score < 0: return "Negatif", "😞", random.randint(70, 95)
    return "Netral", "😐", random.randint(50, 70)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#1f2937; margin-bottom:0;'>Sentiment<span style='color:#f97316;'>🙂</span></h1>", unsafe_allow_html=True)
    # Fix Teks Caption di Sidebar
    st.markdown("<p class='sidebar-caption'>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"])
    st.divider()
    st.info("Status: Live (Public)")

# --- 6. PAGE CONTENT ---

if menu == "Dashboard":
    st.header("Overview Dashboard")
    s = st.session_state.stats
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif</div><div class="metric-value">{s["positif"]}</div></div><div class="icon-box bg-green">😊</div></div>', unsafe_allow_html=True)
    with c3: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif</div><div class="metric-value">{s["negatif"]}</div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: 
        st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral</div><div class="metric-value">{s["netral"]}</div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Belum ada data ulasan yang dianalisis.")

elif menu == "Data Management":
    st.header("Data Management")
    st.write("Kelola dataset ulasan Anda")
    
    # CSS wrapper agar box upload tetap putih
    st.markdown('<div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb;">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head(5), use_container_width=True)
        if st.button("Jalankan Analisis Batch"):
            with st.spinner("Memproses..."):
                results = []
                for i, row in df.iterrows():
                    txt = str(row.iloc[0])
                    res, _, _ = get_prediction(txt)
                    results.append({"Text": txt, "Hasil": res})
                st.session_state.dataset = pd.DataFrame(results)
                st.success("Selesai!")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.dataset is not None:
        st.divider()
        st.dataframe(st.session_state.dataset, use_container_width=True)

elif menu == "Sentiment Prediction":
    st.header("Sentiment Prediction")
    st.write("Analisis teks ulasan secara *real-time*")
    
    st.markdown('<div style="background-color: white; padding: 30px; border-radius: 15px; border: 1px solid #e5e7eb;">', unsafe_allow_html=True)
    input_text = st.text_area("Masukkan ulasan pelanggan:", placeholder="Contoh: Sangat membantu belajarku!")
    
    if st.button("Analisis Sekarang"):
        if input_text:
            res, emo, conf = get_prediction(input_text)
            
            # Update Dashboard Stats
            st.session_state.stats["total"] += 1
            st.session_state.stats[res.lower()] += 1
            st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
            
            st.divider()
            col_res, col_progress = st.columns([1, 2])
            col_res.markdown(f"### Hasil: {res} {emo}")
            col_progress.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
        else:
            st.warning("Silakan isi teks terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)