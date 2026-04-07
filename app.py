import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf  
import numpy as np             
import pickle            
import io

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- 2. CSS CUSTOM (TERMASUK LABEL SKRIPSI) ---
st.markdown("""
    <style>
    /* Style Dasar */
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { 
        background-color: white !important; 
        border-right: 1px solid #e5e7eb !important; 
    }
    [data-testid="stSidebar"] * { color: #1f2937 !important; }
    
    /* CSS UNTUK PENOMORAN (A, B, C) SKRIPSI */
    .label-skripsi {
        display: inline-block;
        background-color: #fb923c;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        text-align: center;
        line-height: 28px;
        font-weight: bold;
        margin-right: 10px;
        font-size: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        vertical-align: middle;
    }

    /* Style Komponen Lain */
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

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_sentiment_model():
    # Pastikan path file .h5 dan .pkl sudah benar di folder projectmu
    try:
        model = tf.keras.models.load_model('models/model_hybrid_coc.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('models/normalization_dicts.pkl', 'rb') as f:
            norm_dict = pickle.load(f)
        return model, tokenizer, norm_dict
    except:
        st.error("Gagal memuat model. Pastikan folder 'models/' berisi file yang diperlukan.")
        return None, None, None

model_ml, tokenizer_ml, norm_dict = load_sentiment_model()

# --- 4. PREPROCESSING FUNCTIONS ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def normalize_text(text):
    text = clean_text(text)
    words = text.split()
    normalized = [norm_dict.get(word, word) for word in words] if norm_dict else words
    return " ".join(normalized).strip()

def remove_stopwords(text):
    sw = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan', 'ada', 'tidak', 'saya', 'bisa'}
    words = str(text).split()
    return " ".join([w for w in words if w not in sw]).strip()

def stem_text(text):
    # Simple stemmer logic as per your snippet
    words = str(text).split()
    return " ".join([w[:4] for w in words]).strip() # Contoh penyederhanaan

def get_prediction(text):
    if model_ml and tokenizer_ml:
        normalized = normalize_text(text)
        seq = tokenizer_ml.texts_to_sequences([normalized])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')
        prediction = model_ml.predict(padded, verbose=0)
        labels = ["Netral", "Negatif", "Positif"]
        emojis = ["😐", "😞", "😀"]
        idx = np.argmax(prediction)
        conf = float(np.max(prediction) * 100)
        scores = {
            "positif": float(prediction[0][2] * 100),
            "negatif": float(prediction[0][1] * 100),
            "netral":  float(prediction[0][0] * 100),
        }
        return labels[idx], emojis[idx], int(conf), scores
    return "Error", "⚠️", 0, {}

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
    st.divider()

# --- 7. DASHBOARD PAGE ---
if menu == "Dashboard":
    st.markdown("<h2><span class='label-skripsi'>a</span> Dashboard</h2>", unsafe_allow_html=True)
    st.write("Overview Statistik Real-time")
    
    s = st.session_state.stats
    st.markdown("<b><span class='label-skripsi'>b</span> Ringkasan Statistik</b>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div><div class="icon-box bg-blue">📊</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Positif</div><div class="metric-value">{s["positif"]}</div></div><div class="icon-box bg-green">😊</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Negatif</div><div class="metric-value">{s["negatif"]}</div></div><div class="icon-box bg-red">😞</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div><div class="metric-title">Netral</div><div class="metric-value">{s["netral"]}</div></div><div class="icon-box bg-gray">😐</div></div>', unsafe_allow_html=True)

    st.markdown("### <span class='label-skripsi'>c</span> Aktivitas Terbaru", unsafe_allow_html=True)
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Belum ada aktivitas analisis.")

# --- 8. DATA MANAGEMENT PAGE ---
elif menu == "Data Management":
    st.markdown("<h2><span class='label-skripsi'>a</span> Data Management</h2>", unsafe_allow_html=True)
    st.write("Proses dataset dalam jumlah besar (CSV/Excel)")
    
    with st.container(border=True):
        st.markdown("<b><span class='label-skripsi'>b</span> Upload dataset ulasan</b>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.markdown("<b><span class='label-skripsi'>c</span> Preview 10 Data Teratas:</b>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("<b><span class='label-skripsi'>d</span> Kontrol Analisis</b>", unsafe_allow_html=True)
            if st.button("🔍 Jalankan Batch Analysis"):
                st.success("Analisis Berhasil (Simulasi)")

# --- 9. SENTIMENT PREDICTION PAGE ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2><span class='label-skripsi'>a</span> Sentiment Prediction</h2>", unsafe_allow_html=True)
    st.write("Analisis teks tunggal secara real-time")
    
    with st.container(border=True):
        st.markdown("### Sentiment Analysis")
        st.markdown("<b><span class='label-skripsi'>b</span> Masukkan teks ulasan</b>", unsafe_allow_html=True)
        input_text = st.text_area("", placeholder="Contoh: Aplikasi ini sangat membantu...", height=150, label_visibility="collapsed")
        
        st.write("")
        btn_label = st.markdown("<b><span class='label-skripsi'>c</span> Tombol Eksekusi</b>", unsafe_allow_html=True)
        if st.button("Analisis Sentimen Sekarang"):
            if input_text:
                res, emo, conf, scores = get_prediction(input_text)
                st.session_state.history.insert(0, {"Teks": input_text, "Hasil": res, "Waktu": time.strftime("%H:%M:%S")})
                
                st.divider()
                st.markdown(f"### <span class='label-skripsi'>d</span> Hasil: {res} {emo}", unsafe_allow_html=True)
                st.progress(conf/100, text=f"Tingkat Keyakinan: {conf}%")
