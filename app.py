import streamlit as st
import pandas as pd
import time
import re
import joblib
import tensorflow as tf  
import numpy as np        
import pickle             
import io
import firebase_admin
from firebase_admin import credentials, firestore

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Pro", page_icon="🙂", layout="wide")

# --- FIREBASE INITIALIZATION ---
# Menggunakan Streamlit Secrets untuk keamanan (bukan file JSON eksternal)
if not firebase_admin._apps:
    try:
        fb_credentials = dict(st.secrets["firebase"])
        cred = credentials.Certificate(fb_credentials)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Gagal inisialisasi Firebase: {e}")

db = firestore.client()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; }
    .stApp { background-color: #f7f9fc !important; color: #1f2937 !important; }
    [data-testid="stSidebar"] { background-color: white !important; border-right: 1px solid #e5e7eb !important; padding-top: 2rem; }
    .metric-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #f3f4f6; margin-bottom: 1rem; }
    .metric-title { color: #6b7280 !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #1f2937 !important; font-size: 1.75rem; font-weight: 700; }
    .stButton>button { background: #f97316 !important; color: white !important; border-radius: 8px !important; font-weight: 600 !important; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS FOR FIREBASE ---
def save_to_firebase(text, result, confidence):
    """Simpan hasil prediksi ke Firestore"""
    try:
        data = {
            "teks": text,
            "hasil": result,
            "keyakinan": confidence,
            "waktu": firestore.SERVER_TIMESTAMP
        }
        db.collection("history_sentiment").add(data)
    except Exception as e:
        st.error(f"Gagal menyimpan ke Firebase: {e}")

def get_stats_from_firebase():
    """Ambil statistik real-time dari Firestore"""
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
    except:
        return {"total": 0, "positif": 0, "negatif": 0, "netral": 0}

def get_history_from_firebase(limit_count=10):
    """Ambil riwayat data terbaru"""
    try:
        docs = db.collection("history_sentiment").order_by("waktu", direction=firestore.Query.DESCENDING).limit(limit_count).stream()
        history = []
        for doc in docs:
            d = doc.to_dict()
            waktu = d['waktu'].strftime("%H:%M:%S") if d.get('waktu') else "N/A"
            history.append({"Teks": d.get("teks"), "Hasil": d.get("hasil"), "Waktu": waktu})
        return history
    except:
        return []

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

# --- PREPROCESSING FUNCTIONS ---
def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    normalized = [norm_dict.get(word, word) for word in words] if norm_dict else words
    return " ".join(normalized).strip()

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
        return labels[idx], emojis[idx], int(conf)
    return "Error", "⚠️", 0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2>Sentiment<span style='color:#f97316;'>🙂</span></h2>", unsafe_allow_html=True)
    st.markdown("<p>Project Analisis Sentimen Ruangguru</p>", unsafe_allow_html=True)
    menu = st.radio("MAIN MENU", ["Dashboard", "Data Management", "Sentiment Prediction"], label_visibility="collapsed")

# --- PAGE: DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h2>Dashboard</h2>", unsafe_allow_html=True)
    
    with st.spinner("Sinkronisasi data database..."):
        s = get_stats_from_firebase()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Data</div><div class="metric-value">{s["total"]}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Positif 😊</div><div class="metric-value">{s["positif"]}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Negatif 😞</div><div class="metric-value">{s["negatif"]}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Netral 😐</div><div class="metric-value">{s["netral"]}</div></div>', unsafe_allow_html=True)

    st.subheader("Aktivitas Terbaru (Top 10)")
    recent_history = get_history_from_firebase(10)
    if recent_history:
        st.dataframe(pd.DataFrame(recent_history), use_container_width=True)
        
        with st.expander("Lihat Riwayat Lengkap"):
            full_history = get_history_from_firebase(100)
            st.table(full_history)
    else:
        st.info("Belum ada aktivitas di database.")

# --- PAGE: DATA MANAGEMENT (LOCAL BATCH) ---
elif menu == "Data Management":
    st.markdown("<h2>Data Management</h2>", unsafe_allow_html=True)
    st.info("Batch Analysis diproses secara lokal dan tidak disimpan ke Firebase.")
    
    uploaded_file = st.file_uploader("Upload dataset ulasan", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write(f"Preview Data: {len(df)} baris")
        st.dataframe(df.head(5), use_container_width=True)
        
        if st.button("Jalankan Batch Analysis"):
            with st.spinner("Menganalisis..."):
                text_col = next((c for c in ['text', 'ulasan', 'komentar', 'content'] if c in df.columns), df.columns[0])
                texts = df[text_col].astype(str).tolist()
                
                # Bulk prediction (tidak simpan ke firebase per baris)
                results = []
                for t in texts:
                    res, emo, conf = get_prediction(t)
                    results.append({"Teks": t, "Sentimen": res, "Skor": conf})
                
                st.session_state.batch_result = pd.DataFrame(results)
                st.success("Analisis Batch Selesai!")
                st.dataframe(st.session_state.batch_result, use_container_width=True)

# --- PAGE: SENTIMENT PREDICTION ---
elif menu == "Sentiment Prediction":
    st.markdown("<h2>Sentiment Prediction</h2>", unsafe_allow_html=True)
    
    with st.container(border=True):
        input_text = st.text_area("Masukkan teks ulasan", placeholder="Ketik di sini...", height=150)
        
        if st.button("Analisis Sentimen Sekarang"):
            if input_text.strip():
                res, emo, conf = get_prediction(input_text)
                
                # SIMPAN KE FIREBASE
                save_to_firebase(input_text, res, conf)
                
                st.divider()
                st.markdown(f"### Hasil: {res} {emo}")
                st.write(f"Tingkat Keyakinan: {conf}%")
                st.progress(conf/100)
            else:
                st.warning("Masukkan teks ulasan terlebih dahulu.")
