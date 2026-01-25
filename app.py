import streamlit as st
import pandas as pd
import time
import re
import random
import joblib  # Persiapkan library ini untuk load model nanti

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. CSS SAKTI (Clean Orange UI) ---
st.markdown("""
    <style>
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { 
        background-color: white !important; 
        border-right: 1px solid #e5e7eb !important; 
    }
    [data-testid="stSidebar"] * { color: #1f2937 !important; }
    [data-testid="stSidebar"] .stMarkdown p { color: #4b5563 !important; font-size: 0.95rem; font-weight: 500; }
    .stTextArea textarea { background-color: white !important; color: #1f2937 !important; border: 1px solid #d1d5db !important; }
    [data-testid="stFileUploader"] { background-color: white !important; border: 2px dashed #fb923c !important; border-radius: 12px !important; }
    [data-testid="stFileUploaderDropzone"] { background-color: #ffffff !important; }
    .metric-card {
        background-color: white; padding: 24px; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #f3f4f6;
        display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;
    }
    .metric-title { color: #6b7280 !important; font-size: 0.9rem; }
    .metric-value { color: #1f2937 !important; font-size: 2rem; font-weight: 700; margin-top: 8px; }
    .icon-box { width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px; }
    .bg-blue { background-color: #dbeafe; color: #3b82f6; }
    .bg-green { background-color: #d1fae5; color: #10b981; }
    .bg-red { background-color: #fee2e2; color: #ef4444; }
    .bg-gray { background-color: #f3f4f6; color: #6b7280; }
    .stButton>button {
        background: linear-gradient(135deg, #fb923c, #f97316) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING (TEMPAT TIM ML BEKERJA) ---
@st.cache_resource
def load_sentiment_model():
    """
    Fungsi ini untuk me-load model Machine Learning.
    Gunakan @st.cache_resource agar model tidak di-load berulang kali (berat).
    """
    # model = joblib.load('path_ke_model.pkl') # Nanti diisi ini
    # vectorizer = joblib.load('path_ke_vectorizer.pkl') # Nanti diisi ini
    return None # Sementara return None

model_ml = load_sentiment_model()

# --- 4. LOGIC PREPROCESSING & PREDICTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_prediction(text):
    # JIKA MODEL SUDAH ADA, GUNAKAN LOGIK INI:
    # if model_ml:
    #     vec_text = vectorizer.transform([clean_text(text)])
    #     pred = model_ml.predict(vec_text)[0]
    #     conf = model_ml.predict_proba(vec_text).max() * 100
    #     return pred, "Emoji", int(conf)

    # DUMMY LOGIC (UNTUK TESTING UI SEKARANG)
    pos_words = ["bagus", "puas", "mantap", "keren", "oke"]
    neg_words = ["buruk", "kecewa", "parah", "jelek", "lambat"]
    score = 0
    cleaned = clean_text(text)
    for w in pos_words: 
        if w in cleaned: score += 1
    for w in neg_words: 
        if w in cleaned: score -= 1
    
    if score > 0: return "Positif", "😀", random.randint(75, 98)
    if score < 0: return "Negatif", "😞", random.randint(70, 95)
    return "Netral", "😐", random.randint(50, 70)

# --- 5. SESSION STATE ---
if 'stats' not in st.session_state:
    st.session_state.stats = {"total": 0, "positif": 0, "negatif": 0, "netral": 0}
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    st.markdown("<p>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    st.write("")
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")
    st.markdown("<div style='margin-top: 200px;'></div>", unsafe_allow_html=True)
    st.divider()
    st.caption("Developed for Thesis Project © 2026")

# --- 7. PAGE ROUTING ---

# --- DASHBOARD PAGE ---
if menu == "Dashboard":
    st.markdown("<h2 style='color:#1f2937;'>Dashboard</h2>", unsafe_allow_html=True)
    st.write("Overview Statistik Real-time")
    
    s = st.session_state.stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif</div><div class="metric-value">{s["positif"]}</div></div><div class="icon-box bg-green">😊</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif</div><div class="metric-value">{s["negatif"]}</div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral</div><div class="metric-value">{s["netral"]}</div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Belum ada aktivitas analisis.")

# --- DATA MANAGEMENT PAGE ---
elif menu == "Data Management":
    st.markdown("<h2 style='color:#1f2937;'>Data Management</h2>", unsafe_allow_html=True)
    st.write("Proses dataset dalam jumlah besar (CSV/Excel)")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.write("Preview 5 Data Teratas:")
            st.dataframe(df.head(5), use_container_width=True)
            
            if st.button("Jalankan Batch Analysis"):
                with st.spinner("Sedang menganalisis dataset..."):
                    results = []
                    for i, row in df.iterrows():
                        txt = str(row.iloc[0]) # Mengasumsikan teks ada di kolom pertama
                        sentiment, _, _ = get_prediction(txt)
                        results.append({"Ulasan": txt, "Cleaned": clean_text(txt), "Sentimen": sentiment})
                    st.session_state.dataset = pd.DataFrame(results)
                    st.success("Batch Analysis Selesai!")

    if st.session_state.dataset is not None:
        with st.container(border=True):
            st.write("Hasil Analisis Keseluruhan:")
            st.dataframe(st.session_state.dataset, use_container_width=True)
            st.download_button("Download Hasil (.csv)", st.session_state.dataset.to_csv(index=False), "hasil_sentimen.csv")
            if st.button("Hapus Cache Data"):
                st.session_state.dataset = None
                st.rerun()

# --- PREDICTION PAGE ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2 style='color:#1f2937;'>Sentiment Prediction</h2>", unsafe_allow_html=True)
    st.write("Analisis teks tunggal secara real-time")
    
    with st.container(border=True):
        st.subheader("Sentiment Analysis")
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Aplikasi ini sangat membantu dalam belajar...", height=150)
        
        if st.button("Analisis Sentimen Sekarang"):
            if input_text:
                res, emo, conf = get_prediction(input_text)
                
                # Update Statistik di Dashboard
                st.session_state.stats["total"] += 1
                st.session_state.stats[res.lower()] += 1
                st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
                
                # Menampilkan Hasil ke User
                st.divider()
                col_res, col_conf = st.columns(2)
                col_res.markdown(f"### Hasil: {res} {emo}")
                col_conf.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
            else:
                st.warning("Silakan ketikkan ulasan terlebih dahulu.")