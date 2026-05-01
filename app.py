import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf
import numpy as np
import pickle
import io
import os
import firebase_admin
from firebase_admin import credentials, firestore

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- FIREBASE INITIALIZATION ---
if not firebase_admin._apps:
    try:
        fb_credentials = dict(st.secrets["firebase"])
        cred = credentials.Certificate(fb_credentials)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

db = firestore.client()

# --- CUSTOM CSS (PRESISI FIGMA & INTER FONT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp { 
        font-family: 'Inter', sans-serif !important; 
    }
    
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }

    /* SIDEBAR CONTAINER */
    [data-testid="stSidebar"] { 
        background-color: white !important; 
        border-right: 1px solid #e5e7eb !important; 
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-top: 2rem;
    }

    /* HEADER SIDEBAR */
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
    }
    
    .sidebar-subtitle {
        font-size: 16px;
        color: #1e293b;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* RADIO MENU STYLING */
    div.row-widget.stRadio > div {
        gap: 15px; 
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] { display: none; }

    [data-testid="stSidebar"] label {
        font-size: 18px !important;
        font-weight: 400 !important;
        color: #000000 !important;
    }

    /* DASHBOARD CARD */
    .metric-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); 
        border: 1px solid #f3f4f6; 
        margin-bottom: 1rem; 
    }
    .metric-title { color: #6b7280; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937; font-size: 1.75rem; font-weight: 700; }

    /* UI DATA MANAGEMENT (SIMPLE METRICS) */
    .simple-metric-label { color: #6b7280; font-size: 0.9rem; font-weight: 500; margin-bottom: -5px; }
    .simple-metric-value { color: #1f2937; font-size: 1.8rem; font-weight: 700; margin-bottom: 15px; }

    /* BUTTONS */
    .stButton>button { 
        background: #f97316 !important; 
        color: white !important; 
        border-radius: 8px !important; 
        font-weight: 600 !important; 
        border: none !important;
        width: 100%;
    }
    
    /* Tombol Download */
    [data-testid="stDownloadButton"] > button {
        background: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FIREBASE HELPERS ---
def save_to_firebase(text, result, confidence):
    try:
        db.collection("history_sentiment").add({
            "teks": text, "hasil": result, "keyakinan": confidence, "waktu": firestore.SERVER_TIMESTAMP
        })
    except: pass

def get_stats_firebase():
    try:
        docs = db.collection("history_sentiment").stream()
        total, pos, neg, net = 0, 0, 0, 0
        for doc in docs:
            total += 1
            res = doc.to_dict().get("hasil", "").lower()
            if res == "positif": pos += 1
            elif res == "negatif": neg += 1
            elif res == "netral": net += 1
        return {"total": total, "positif": pos, "negatif": neg, "netral": net}
    except: return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_history_firebase(limit=10):
    try:
        docs = db.collection("history_sentiment").order_by("waktu", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [{"Teks": d.to_dict().get("teks"), "Hasil": d.to_dict().get("hasil"), "Waktu": d.to_dict()['waktu'].strftime("%H:%M:%S") if d.to_dict().get('waktu') else "N/A"} for d in docs]
    except: return []

# --- MODEL LOADING ---
@st.cache_resource
def load_sentiment_model():
    try:
        model = tf.keras.models.load_model('models/model_hybrid_coc.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('models/normalization_dicts.pkl', 'rb') as f:
            norm_dict = pickle.load(f)
        return model, tokenizer, norm_dict
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

model_ml, tokenizer_ml, norm_dict = load_sentiment_model()

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    if norm_dict:
        normalized = [norm_dict.get(word, word) for word in words]
        return " ".join(normalized).strip()
    return text.strip()

def get_prediction(text):
    if model_ml and tokenizer_ml:
        normalized = normalize_text(text)
        seq = tokenizer_ml.texts_to_sequences([normalized])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100, padding='post')
        prediction = model_ml.predict(padded, verbose=0)
        labels, emojis = ["Netral", "Negatif", "Positif"], ["😐", "😞", "😀"]
        idx = np.argmax(prediction)
        return labels[idx], emojis[idx], int(np.max(prediction) * 100)
    return "Error", "⚠️", 0

# --- SESSION STATE ---
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'uploaded_df' not in st.session_state: st.session_state.uploaded_df = None
if 'uploaded_filename' not in st.session_state: st.session_state.uploaded_filename = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">Sentiment🙂</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Analisis Sentimen Ruangguru</div>', unsafe_allow_html=True)
    menu = st.radio("NAVIGATION", ["Dashboard", "Data Management", "Sentiment Prediction"])

# --- DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
    s = get_stats_firebase()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s["positif"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s["negatif"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s["netral"]}</div></div>', unsafe_allow_html=True)
    
    st.subheader("Aktivitas Terbaru")
    hist = get_history_firebase(10)
    if hist:
        st.dataframe(pd.DataFrame(hist), use_container_width=True)
        with st.expander("Lihat Riwayat Lengkap"):
            full_hist = get_history_firebase(100)
            st.table(full_hist)
    else: st.info("Belum ada data.")

# --- DATA MANAGEMENT ---
elif menu == "Data Management":
    st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
    if uploaded_file:
        if st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.uploaded_df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.dataset = None

    if st.session_state.uploaded_df is not None:
        df_view = st.session_state.uploaded_df
        st.write(f"📁 **{st.session_state.uploaded_filename}** — {len(df_view)} baris")
        st.dataframe(df_view.head(5), use_container_width=True)
        
        if st.button("🔍 Jalankan Batch Analysis"):
            with st.spinner("Menganalisis..."):
                text_col = next((c for c in ['text', 'ulasan', 'komentar', 'textDisplay'] if c in df_view.columns), df_view.columns[0])
                texts = df_view[text_col].astype(str).tolist()
                prog = st.progress(0)
                
                normalized = [normalize_text(t) for t in texts]
                prog.progress(0.4)
                
                seqs = tokenizer_ml.texts_to_sequences(normalized)
                padded = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=100, padding='post')
                preds = model_ml.predict(padded, batch_size=512, verbose=0)
                prog.progress(1.0)
                
                labels = ["Netral", "Negatif", "Positif"]
                res_list = [labels[np.argmax(p)] for p in preds]
                conf_list = [int(np.max(p)*100) for p in preds]
                
                st.session_state.dataset = pd.DataFrame({
                    "Text Asli": texts, 
                    "Sentimen": res_list,
                    "Keyakinan (%)": conf_list
                })

    if st.session_state.dataset is not None:
        st.divider()
        st.markdown("### Hasil Analisis")
        
        # --- SIMPLE METRICS BAR (SESUAI GAMBAR 3d15b3 - Tanpa Card Putih) ---
        res_df = st.session_state.dataset
        total_v = len(res_df)
        pos_v = len(res_df[res_df['Sentimen'] == "Positif"])
        neg_v = len(res_df[res_df['Sentimen'] == "Negatif"])
        net_v = len(res_df[res_df['Sentimen'] == "Netral"])
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<p class="simple-metric-label">Total</p><p class="simple-metric-value">{total_v}</p>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<p class="simple-metric-label">😀 Positif</p><p class="simple-metric-value">{pos_v}</p>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<p class="simple-metric-label">😞 Negatif</p><p class="simple-metric-value">{neg_v}</p>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<p class="simple-metric-label">😐 Netral</p><p class="simple-metric-value">{net_v}</p>', unsafe_allow_html=True)
        
        # --- TABEL ---
        st.dataframe(res_df, use_container_width=True)
        
        # --- FOOTER ACTIONS (DOWNLOAD KIRI, HAPUS KANAN) ---
        col_csv, col_excel, col_spacer, col_del = st.columns([0.8, 1, 6, 1.5])
        
        with col_csv:
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ CSV", csv_data, "hasil_sentimen.csv", "text/csv")
        
        with col_excel:
            # Excel diabaikan seperti instruksi (disiapkan buffer kosong atau teks saja)
            st.info("Excel tidak tersedia.", icon="ℹ️")
        
        with col_del:
            if st.button("🗑️ Hapus Hasil"):
                st.session_state.dataset = None
                st.rerun()

# --- PREDICTION ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2>Sentiment Prediction</h2>", unsafe_allow_html=True)
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Contoh: Keren banget!", height=150)
        if st.button("Analisis Sekarang"):
            if input_text.strip():
                res, emo, conf = get_prediction(input_text)
                save_to_firebase(input_text, res, conf)
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.write(f"Keyakinan: {conf}%")
                st.progress(conf/100)
